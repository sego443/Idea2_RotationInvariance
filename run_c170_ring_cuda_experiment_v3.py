"""Run one validation-only C170 v3 shared-scale ring CE experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_experiment as c170
import run_c170_ring_cuda_experiment_v2 as v2
from src.c170_shared_scale_preprocessing_v3 import prepare_model_inputs_shared_scale_v3
from src.c170_ring_representation import RingRepresentationClassifier


TRAIN_DISTANCE_MM = v2.TRAIN_DISTANCE_MM
VAL_DISTANCE_MM = v2.VAL_DISTANCE_MM
REGISTERED_TEST_DISTANCE_MM = v2.REGISTERED_TEST_DISTANCE_MM
EXCLUDED_DISTANCE_MM = v2.EXCLUDED_DISTANCE_MM
SEEDS = v2.SEEDS
N_ACTIONS = v2.N_ACTIONS
UPDATES_PER_EPOCH = v2.UPDATES_PER_EPOCH
LEARNING_RATE = v2.LEARNING_RATE
WEIGHT_DECAY = v2.WEIGHT_DECAY
DEFAULT_DATASET = v2.DEFAULT_DATASET
VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "shared_scale_ce": {
        "model_variant": "classification_only",
        "use_supcon": False,
        "shuffle_labels": False,
    },
    "shared_scale_shuffle": {
        "model_variant": "classification_only",
        "use_supcon": False,
        "shuffle_labels": True,
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one fixed C170 v3 shared-scale CE seed using train/validation only."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANT_CONFIG), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--confirm-physical-order",
        action="store_true",
        help="Confirm the frozen channel-map interpretation S0-S3 before training.",
    )
    return parser


def _save_numpy_state_dict(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    """Save only numeric tensor arrays in an NPZ archive (no pickle objects)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("safe state dict must map string names to tensors")
        if tensor.layout != torch.strided:
            raise TypeError(f"safe state dict tensor {name} must use strided layout")
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        if array.dtype.hasobject or array.dtype.kind not in "biuf":
            raise TypeError(f"safe state dict tensor {name} has unsupported dtype {array.dtype}")
        arrays[name] = np.array(array, copy=True)
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_numpy_state_dict(
    path: Path,
    expected_state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load numeric NPZ arrays with pickle disabled and exact state validation."""
    restored: dict[str, torch.Tensor] = {}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_state_dict):
            raise ValueError("safe state dict keys differ from the model state")
        for name, expected in expected_state_dict.items():
            array = np.array(archive[name], copy=True)
            if array.dtype.hasobject or array.dtype.kind not in "biuf":
                raise TypeError(f"safe state dict tensor {name} is not numeric")
            expected_array = expected.detach().cpu().numpy()
            if array.shape != tuple(expected.shape):
                raise ValueError(f"safe state dict tensor {name} has shape {array.shape}, expected {tuple(expected.shape)}")
            if array.dtype != expected_array.dtype:
                raise TypeError(f"safe state dict tensor {name} has dtype {array.dtype}, expected {expected_array.dtype}")
            value = torch.from_numpy(array).to(device=device)
            if value.dtype != expected.dtype:
                raise TypeError(f"safe state dict tensor {name} changed torch dtype")
            restored[name] = value
    return restored


def _collect_code_hashes() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "shared_scale_preprocessing": PROJECT_DIR / "src" / "c170_shared_scale_preprocessing_v3.py",
        "ring_representation": PROJECT_DIR / "src" / "c170_ring_representation.py",
        "frozen_v2_training_helpers": Path(v2.__file__).resolve(),
        "dataset_preprocessing_helpers": Path(c170.__file__).resolve(),
        "preprocessing_transitive_signal_processing": PROJECT_DIR / "src" / "rotation_poc.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing computation dependencies: {missing}")
    return {name: v2._file_sha256(path) for name, path in files.items()}


def _training_targets_v3(
    dataset: dict[str, Any],
    splits: dict[str, np.ndarray],
    variant: str,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    labels = np.asarray(dataset["labels"], dtype=np.int64)
    distances = np.asarray(dataset["distance_mm"], dtype=np.int64)
    blocks = np.asarray(dataset["block"], dtype=np.int64)
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    targets = labels.copy()
    if VARIANT_CONFIG[variant]["shuffle_labels"]:
        targets = c170.shuffle_training_labels_by_recording(
            targets,
            distances,
            blocks,
            train_indices,
            seed=int(seed) + v2.LABEL_SHUFFLE_SEED_OFFSET,
        )
        for block in np.unique(blocks[train_indices]):
            selected = train_indices[blocks[train_indices] == block]
            if len(np.unique(targets[selected])) != 1:
                raise RuntimeError(f"shuffled training block {int(block)} has multiple targets")
        for distance in TRAIN_DISTANCE_MM:
            selected = train_indices[distances[train_indices] == distance]
            if set(np.unique(targets[selected]).tolist()) != set(range(N_ACTIONS)):
                raise RuntimeError(f"block shuffle lost a direction target at {distance} mm")
    elif variant != "shared_scale_ce":
        raise ValueError(f"unsupported v3 variant: {variant}")

    outside_train = np.ones(len(labels), dtype=bool)
    outside_train[train_indices] = False
    if not np.array_equal(targets[outside_train], labels[outside_train]):
        raise RuntimeError("label shuffle changed labels outside the training split")
    shuffle_mapping = (
        v2._make_shuffle_mapping(dataset, train_indices, targets)
        if VARIANT_CONFIG[variant]["shuffle_labels"]
        else []
    )
    return targets, shuffle_mapping


def _validate_and_prepare_v3(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    dataset = c170.load_dataset(args.dataset)
    assignments = {
        "train": TRAIN_DISTANCE_MM,
        "val": VAL_DISTANCE_MM,
        "test": REGISTERED_TEST_DISTANCE_MM,
    }
    splits = c170.split_by_distance(dataset["distance_mm"], assignments)
    expected_positions = set(TRAIN_DISTANCE_MM + VAL_DISTANCE_MM + REGISTERED_TEST_DISTANCE_MM)
    expected_positions.update(EXCLUDED_DISTANCE_MM)
    observed_positions = set(int(value) for value in np.unique(dataset["distance_mm"]).tolist())
    if observed_positions != expected_positions:
        raise ValueError(
            f"observed distance positions {sorted(observed_positions)} differ from fixed split"
        )
    counts = {name: int(len(values)) for name, values in splits.items()}
    if counts != {"train": 879, "val": 320, "test": 320}:
        raise ValueError(f"expected train/val/test counts 879/320/320, got {counts}")

    train_val_indices = np.concatenate((splits["train"], splits["val"]))
    train_val_metadata = {
        name: dataset[name][train_val_indices]
        for name in ("block", "distance_mm", "labels", "channel_map")
    }
    c170.assert_recording_groups_intact(train_val_metadata)
    channel_map_confirmation = c170.validate_channel_maps(
        dataset["channel_map"][train_val_indices],
        confirm_physical_order=bool(args.confirm_physical_order),
    )

    normalized, shared_scale = prepare_model_inputs_shared_scale_v3(
        dataset["X_raw"], {"train": splits["train"], "val": splits["val"]}
    )
    train_np = np.asarray(normalized["train"], dtype=np.float32)
    val_np = np.asarray(normalized["val"], dtype=np.float32)
    if not np.isfinite(train_np).all() or not np.isfinite(val_np).all():
        raise ValueError("preprocessed train/validation inputs contain non-finite values")
    train_inputs = torch.from_numpy(train_np).to(device=device, dtype=torch.float32)
    val_inputs = torch.from_numpy(val_np).to(device=device, dtype=torch.float32)

    labels = np.asarray(dataset["labels"], dtype=np.int64)
    distances = np.asarray(dataset["distance_mm"], dtype=np.int64)
    blocks = np.asarray(dataset["block"], dtype=np.int64)
    original_groups = v2._training_groups(splits["train"], labels, distances)
    position_weights, sample_weights = v2._training_sample_weights(original_groups)
    weight_array = np.asarray(
        [sample_weights[int(index)] for index in splits["train"]], dtype=np.float32
    )
    if not math.isclose(float(weight_array.sum()), 1.0, rel_tol=0.0, abs_tol=2e-7):
        raise ValueError("training sample weights must sum to one")

    all_training_targets, shuffle_mapping = _training_targets_v3(
        dataset, splits, args.variant, args.seed
    )
    train_targets = torch.as_tensor(
        all_training_targets[splits["train"]], dtype=torch.long, device=device
    )
    val_targets = torch.as_tensor(labels[splits["val"]], dtype=torch.long, device=device)
    train_weights = torch.as_tensor(weight_array, dtype=torch.float32, device=device)

    split_metadata = {
        "split_counts": counts,
        "split_positions_mm": {
            "train": list(TRAIN_DISTANCE_MM),
            "validation": list(VAL_DISTANCE_MM),
            "registered_test": list(REGISTERED_TEST_DISTANCE_MM),
            "excluded": list(EXCLUDED_DISTANCE_MM),
        },
        "excluded_70mm_count": int(np.count_nonzero(distances == 70)),
        "duplicate_rows_excluded_by_loader": int(dataset["duplicate_rows_excluded"]),
        "channel_map_confirmation": channel_map_confirmation,
        "channel_map_counts": {
            split: {
                ",".join(str(int(value)) for value in mapping): int(
                    np.count_nonzero(
                        np.all(dataset["channel_map"][indices] == np.asarray(mapping), axis=1)
                    )
                )
                for mapping in sorted(
                    {
                        tuple(int(value) for value in row)
                        for row in dataset["channel_map"][indices].tolist()
                    }
                )
            }
            for split, indices in (("train", splits["train"]), ("validation", splits["val"]))
        },
        "training_original_group_sizes": [
            {"distance_mm": key[0], "direction": c170.DIRECTIONS[key[1]], "count": len(value)}
            for key, value in original_groups.items()
        ],
        "position_weights": {str(key): value for key, value in position_weights.items()},
        "label_shuffle_seed": (
            args.seed + v2.LABEL_SHUFFLE_SEED_OFFSET
            if VARIANT_CONFIG[args.variant]["shuffle_labels"]
            else None
        ),
        "label_shuffle_mapping": shuffle_mapping,
    }
    preprocessing = {
        "sample_rate_hz": 250,
        "filter_band_hz": [0.5, 40.0],
        "filter_type": "butterworth_bandpass_sosfiltfilt_order_3",
        "dc_removal": "subtract_per_trial_per_channel_temporal_mean",
        "sensor_scale_source": "train_filtered_demeaned_global_median_absolute_across_samples_time_channels",
        "sensor_scale_axes": ["train_samples", "time", "channels"],
        "shared_sensor_scale": float(shared_scale),
        "normalization_fit_splits": ["train"],
        "processed_splits": ["train", "validation"],
        "channel_order": ["S0", "S1", "S2", "S3"],
        "channel_map_as_model_input": False,
        "test_preprocessed": False,
    }
    tensors = {
        "train_inputs": train_inputs,
        "validation_inputs": val_inputs,
        "train_targets": train_targets,
        "validation_targets": val_targets,
        "train_weights": train_weights,
        "train_indices": splits["train"].astype(np.int64, copy=True),
        "validation_indices": splits["val"].astype(np.int64, copy=True),
        "train_distances": distances[splits["train"]].copy(),
        "validation_distances": distances[splits["val"]].copy(),
        "train_blocks": blocks[splits["train"]].copy(),
        "validation_channel_maps": np.asarray(dataset["channel_map"][splits["val"]], dtype=np.int64),
        "train_channel_maps": np.asarray(dataset["channel_map"][splits["train"]], dtype=np.int64),
        "original_groups": original_groups,
    }
    return dataset, splits, split_metadata, {"preprocessing": preprocessing, "tensors": tensors}


def _manifest_base_v3(
    config: dict[str, Any],
    config_sha256: str,
    started_utc: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "format": "c170-ring-cuda-v3-run",
        "format_version": 1,
        "status": "running",
        "config": config,
        "config_sha256": config_sha256,
        "metadata": {
            "variant": args.variant,
            "seed": int(args.seed),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "scipy_version": __import__("scipy").__version__,
            "torch_cuda_build": torch.version.cuda,
            "actual_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "test_metrics_computed": False,
        },
        "timing": {
            "started_utc": started_utc,
            "preprocessing_seconds": None,
            "training_seconds": None,
            "total_elapsed_seconds": None,
        },
        "split": {},
        "preprocessing": {},
        "selection": {},
        "selected_checkpoint_train_evaluation": {},
        "provenance": {"code_sha256": _collect_code_hashes()},
        "artifacts": {},
        "test_access": {
            "loaded_as_part_of_complete_npz": True,
            "used_as_model_input": False,
            "preprocessed": False,
            "used_to_estimate_statistics": False,
            "predicted_or_scored": False,
        },
        "test_metrics_computed": False,
    }


def _run(args: argparse.Namespace) -> int:
    if args.max_epochs < 1 or args.patience < 1 or args.cpu_threads < 1:
        raise ValueError("max-epochs, patience, and cpu-threads must be positive")
    if not args.confirm_physical_order:
        raise ValueError("pass --confirm-physical-order after confirming the S0-S3 order")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    dataset_path = args.dataset.resolve()
    dataset_sha256 = v2._file_sha256(dataset_path)
    variant_config = VARIANT_CONFIG[args.variant]
    config = {
        "variant": args.variant,
        "model_variant": variant_config["model_variant"],
        "use_supcon": False,
        "shuffle_labels": variant_config["shuffle_labels"],
        "seed": int(args.seed),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "cpu_threads": int(args.cpu_threads),
        "device": str(device),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "train_distance_mm": list(TRAIN_DISTANCE_MM),
        "validation_distance_mm": list(VAL_DISTANCE_MM),
        "registered_test_distance_mm": list(REGISTERED_TEST_DISTANCE_MM),
        "excluded_distance_mm": list(EXCLUDED_DISTANCE_MM),
        "expected_split_counts": {"train": 879, "validation": 320, "registered_test": 320},
        "batch_sizes_per_epoch": [44] * 19 + [43],
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "sampler_group_key": "original_distance_mm_and_direction_label",
        "sampler_rng": "numpy_seedsequence_seed_epoch_independent_of_model_rng",
        "sample_weight_rule": "position_voronoi_weight/(4*original_group_size)",
        "position_weight_period_mm": v2.ANGLE_PERIOD_MM,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "cross_entropy_label_smoothing": 0.0,
        "batch_ce_objective": "20*sum(sample_weight_i*per_sample_ce_i)",
        "supcon_weight": 0.0,
        "preprocessing_scale": "max(median(abs(filtered_per_trial_demeaned_train)),float32_epsilon)",
        "preprocessing_scale_axes": ["train_samples", "time", "channels"],
        "preprocessing_scale_scope": "one_scalar_shared_by_S0_S1_S2_S3",
        "preprocessing_scale_fit_split": "train",
        "checkpoint_selection": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
        "safe_state_dict_format": "numpy_npz_numeric_arrays",
        "safe_state_dict_reload": "numpy_load_allow_pickle_false_then_strict_state_dict",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
        "physical_order_confirmed_by_operator": True,
    }
    config_sha256 = v2._canonical_json_sha256(config)
    output_dir = args.output_dir.resolve()
    c170.create_output_directory(output_dir)
    manifest = _manifest_base_v3(config, config_sha256, started_utc, args, device)
    manifest_path = output_dir / "run_manifest.json"
    v2._atomic_write_json(manifest_path, manifest)

    try:
        torch.set_num_threads(args.cpu_threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats(device)

        preprocessing_started = time.monotonic()
        dataset, splits, split_metadata, prepared = _validate_and_prepare_v3(args, device)
        tensors = prepared["tensors"]
        preprocessing_seconds = time.monotonic() - preprocessing_started
        manifest["split"] = split_metadata
        manifest["preprocessing"] = prepared["preprocessing"]
        manifest["timing"]["preprocessing_seconds"] = preprocessing_seconds
        v2._atomic_write_json(manifest_path, manifest)

        model = RingRepresentationClassifier(
            variant=variant_config["model_variant"], n_classes=N_ACTIONS
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        train_indices = tensors["train_indices"]
        val_indices = tensors["validation_indices"]
        original_groups = tensors["original_groups"]
        global_to_local = {int(index): offset for offset, index in enumerate(train_indices)}
        best_macro_f1 = -1.0
        best_cross_entropy = float("inf")
        best_epoch = -1
        bad_epochs = 0
        best_metrics: dict[str, Any] | None = None
        best_probabilities: np.ndarray | None = None
        best_predictions: np.ndarray | None = None
        curve_rows: list[dict[str, object]] = []
        training_started = time.monotonic()
        best_checkpoint_path = output_dir / "best_checkpoint.pt"
        safe_state_dict_path = output_dir / "best_weights.npz"

        for epoch in range(args.max_epochs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_started = time.monotonic()
            model.train()
            batches = v2._batches_for_epoch(original_groups, args.seed, epoch)
            weighted_ce_epoch = torch.zeros((), dtype=torch.float32, device=device)

            for batch_global in batches:
                local_np = np.asarray(
                    [global_to_local[int(index)] for index in batch_global], dtype=np.int64
                )
                local_indices = torch.as_tensor(local_np, dtype=torch.long, device=device)
                samples = tensors["train_inputs"][local_indices]
                targets = tensors["train_targets"][local_indices]
                weights = tensors["train_weights"][local_indices]
                logits = model(samples)
                per_sample_ce = F.cross_entropy(
                    logits.float(), targets, reduction="none", label_smoothing=0.0
                )
                loss = v2.batch_objective(
                    per_sample_ce,
                    weights,
                    None,
                    use_supcon=False,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                weighted_ce_epoch = weighted_ce_epoch + torch.sum(weights * per_sample_ce).detach()

            val_metrics, val_probabilities, val_predictions, val_cross_entropy = v2._evaluate(
                model,
                tensors["validation_inputs"],
                tensors["validation_targets"],
                tensors["validation_distances"],
                tensors["validation_channel_maps"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.monotonic() - epoch_started
            training_seconds = time.monotonic() - training_started
            macro_f1 = float(val_metrics["overall"]["macro_f1"])
            improved = macro_f1 > best_macro_f1 or (
                macro_f1 == best_macro_f1 and val_cross_entropy < best_cross_entropy
            )
            if improved:
                best_macro_f1 = macro_f1
                best_cross_entropy = val_cross_entropy
                best_epoch = epoch
                bad_epochs = 0
                best_metrics = val_metrics
                best_probabilities = val_probabilities.copy()
                best_predictions = val_predictions.copy()
                state_dict = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                checkpoint_payload = {
                    "format": "c170-ring-cuda-v3-checkpoint",
                    "format_version": 1,
                    "variant": args.variant,
                    "model_variant": variant_config["model_variant"],
                    "seed": int(args.seed),
                    "best_epoch_zero_based": int(epoch),
                    "state_dict": state_dict,
                    "class_order": list(c170.DIRECTIONS),
                    "preprocessing": prepared["preprocessing"],
                    "experiment_config": config,
                    "experiment_config_sha256": config_sha256,
                    "source_code_sha256": manifest["provenance"]["code_sha256"],
                    "label_shuffle_mapping": split_metadata["label_shuffle_mapping"],
                    "selection": {
                        "criterion": config["checkpoint_selection"],
                        "validation_macro_f1": best_macro_f1,
                        "validation_cross_entropy": best_cross_entropy,
                        "best_epoch_zero_based": best_epoch,
                    },
                    "test_metrics_computed": False,
                }
                _save_numpy_state_dict(safe_state_dict_path, state_dict)
                # Preserve the v2-style metadata-rich checkpoint. It is never loaded by v3.
                v2._atomic_torch_save(best_checkpoint_path, checkpoint_payload)
            else:
                bad_epochs += 1

            weighted_ce_value = float(weighted_ce_epoch.detach().cpu())
            val_by_position_f1 = {
                key: value["macro_f1"] for key, value in val_metrics["by_position_mm"].items()
            }
            recalls = val_metrics["overall"]["per_class_recall"]
            curve_rows.append(
                {
                    "epoch": epoch + 1,
                    "optimizer_updates": UPDATES_PER_EPOCH,
                    "epoch_seconds": epoch_seconds,
                    "elapsed_seconds": training_seconds,
                    "train_weighted_ce_online": weighted_ce_value,
                    "train_supcon_batch_mean_online": None,
                    "train_objective_online": weighted_ce_value,
                    "val_cross_entropy": val_cross_entropy,
                    "val_macro_f1": macro_f1,
                    "val_recall_up": recalls[0],
                    "val_recall_down": recalls[1],
                    "val_recall_left": recalls[2],
                    "val_recall_right": recalls[3],
                    "val_unique_predicted_classes": val_metrics["overall"][
                        "prediction_unique_class_count"
                    ],
                    "val_f1_by_position_json": json.dumps(val_by_position_f1, sort_keys=True),
                    "val_position_channel_map_json": json.dumps(
                        val_metrics["by_position_mm_and_channel_map"], sort_keys=True
                    ),
                }
            )
            v2._atomic_write_csv(
                output_dir / "learning_curve.csv", v2.CURVE_FIELDS, curve_rows
            )
            manifest["timing"].update(
                {
                    "training_seconds": training_seconds,
                    "total_elapsed_seconds": time.monotonic() - started,
                }
            )
            manifest["selection"] = {
                "best_epoch_zero_based": best_epoch,
                "best_epoch_one_based": best_epoch + 1,
                "validation_macro_f1": best_macro_f1,
                "validation_cross_entropy": best_cross_entropy,
                "bad_epochs_since_improvement": bad_epochs,
            }
            manifest["artifacts"] = {
                "best_checkpoint": {"path": best_checkpoint_path.name, "sha256": None},
                "safe_state_dict": {"path": safe_state_dict_path.name, "sha256": None},
                "learning_curve": {"path": "learning_curve.csv", "sha256": None},
            }
            v2._atomic_write_json(manifest_path, manifest)
            print(
                f"variant={args.variant} seed={args.seed} epoch={epoch + 1} "
                f"updates={UPDATES_PER_EPOCH} val_macro_f1={macro_f1:.6f} "
                f"val_ce={val_cross_entropy:.6f} epoch_seconds={epoch_seconds:.3f}",
                flush=True,
            )
            if bad_epochs >= args.patience:
                break

        if best_epoch < 0 or best_metrics is None or best_probabilities is None or best_predictions is None:
            raise RuntimeError("run ended without selecting a validation checkpoint")

        safe_state_dict = _load_numpy_state_dict(safe_state_dict_path, model.state_dict(), device)
        model.load_state_dict(safe_state_dict, strict=True)
        train_checkpoint_metrics = v2._train_checkpoint_metrics(
            model,
            tensors["train_inputs"],
            tensors["train_targets"],
            tensors["train_weights"],
            tensors["train_distances"],
            tensors["train_channel_maps"],
        )
        final_val_metrics, final_val_probabilities, final_val_predictions, final_val_ce = v2._evaluate(
            model,
            tensors["validation_inputs"],
            tensors["validation_targets"],
            tensors["validation_distances"],
            tensors["validation_channel_maps"],
        )
        if not np.allclose(final_val_probabilities, best_probabilities, rtol=0.0, atol=1e-6):
            raise RuntimeError("safe selected-weight reload changed validation probabilities")
        if not np.array_equal(final_val_predictions, best_predictions):
            raise RuntimeError("safe selected-weight reload changed validation predictions")
        if not math.isclose(final_val_ce, best_cross_entropy, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError("safe selected-weight reload changed validation cross-entropy")

        prediction_rows = v2._prediction_rows(
            dataset,
            val_indices,
            final_val_predictions,
            final_val_probabilities,
            args.variant,
            args.seed,
        )
        v2._atomic_write_csv(
            output_dir / "validation_predictions.csv",
            tuple(prediction_rows[0].keys()),
            prediction_rows,
        )
        artifact_hashes = {
            "best_checkpoint": v2._file_sha256(best_checkpoint_path),
            "safe_state_dict": v2._file_sha256(safe_state_dict_path),
            "learning_curve": v2._file_sha256(output_dir / "learning_curve.csv"),
            "validation_predictions": v2._file_sha256(output_dir / "validation_predictions.csv"),
        }
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        manifest["status"] = "completed"
        manifest["timing"].update(
            {
                "training_seconds": time.monotonic() - training_started,
                "total_elapsed_seconds": time.monotonic() - started,
                "peak_cuda_memory_allocated_bytes": peak_memory,
                "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
                    "+00:00", "Z"
                ),
            }
        )
        manifest["selection"] = {
            "best_epoch_zero_based": int(best_epoch),
            "best_epoch_one_based": int(best_epoch + 1),
            "actual_epochs": len(curve_rows),
            "optimizer_updates": len(curve_rows) * UPDATES_PER_EPOCH,
            "validation_macro_f1": float(best_macro_f1),
            "validation_cross_entropy": float(best_cross_entropy),
            "validation_metrics": final_val_metrics,
        }
        manifest["selected_checkpoint_train_evaluation"] = train_checkpoint_metrics
        manifest["artifacts"] = {
            name: {"path": path, "sha256": artifact_hashes[name]}
            for name, path in (
                ("best_checkpoint", best_checkpoint_path.name),
                ("safe_state_dict", safe_state_dict_path.name),
                ("learning_curve", "learning_curve.csv"),
                ("validation_predictions", "validation_predictions.csv"),
            )
        }
        manifest["test_metrics_computed"] = False
        v2._atomic_write_json(manifest_path, manifest)
        manifest_sha256 = v2._file_sha256(manifest_path)
        v2._atomic_write_json(
            output_dir / "RUN_COMPLETE.json",
            {
                "format": "c170-ring-cuda-v3-complete",
                "format_version": 1,
                "status": "completed",
                "config_sha256": config_sha256,
                "manifest_sha256": manifest_sha256,
                "artifact_sha256": artifact_hashes,
                "selected_epoch_one_based": best_epoch + 1,
                "test_metrics_computed": False,
            },
        )
        print(
            f"RUN_COMPLETE variant={args.variant} seed={args.seed} "
            f"best_epoch={best_epoch + 1} val_macro_f1={best_macro_f1:.6f} "
            f"train_macro_f1={train_checkpoint_metrics['overall']['macro_f1']:.6f} "
            f"updates={len(curve_rows) * UPDATES_PER_EPOCH}",
            flush=True,
        )
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["timing"]["total_elapsed_seconds"] = time.monotonic() - started
        try:
            v2._atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
