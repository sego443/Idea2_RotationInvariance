"""Run one fixed C170 harmonic-phase v4 validation-only seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import random
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_experiment as c170
import run_c170_ring_cuda_experiment_v2 as v2
import run_c170_ring_cuda_experiment_v3 as v3
from src.c170_harmonic_phase_model_v4 import (
    ACTIVE_FEATURE_INDICES,
    FEATURE_NAMES,
    HarmonicPhaseClassifierV4,
    VARIANTS,
)


SEEDS = (17, 29, 43, 61, 89)
EXPECTED_SHARED_SCALE = 11123.037109375
TRAIN_DISTANCE_MM = v2.TRAIN_DISTANCE_MM
VAL_DISTANCE_MM = v2.VAL_DISTANCE_MM
REGISTERED_TEST_DISTANCE_MM = v2.REGISTERED_TEST_DISTANCE_MM
EXCLUDED_DISTANCE_MM = v2.EXCLUDED_DISTANCE_MM
UPDATES_PER_EPOCH = v2.UPDATES_PER_EPOCH
LEARNING_RATE = v2.LEARNING_RATE
WEIGHT_DECAY = v2.WEIGHT_DECAY
DEFAULT_DATASET = v2.DEFAULT_DATASET
VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    variant: {
        "model_variant": "harmonic_phase_classifier_v4",
        "active_feature_indices": list(ACTIVE_FEATURE_INDICES[variant]),
        "shuffle_labels": variant == "harmonic_shuffle",
        "preprocessing_variant": "shared_scale_shuffle" if variant == "harmonic_shuffle" else "shared_scale_ce",
    }
    for variant in VARIANTS
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one fixed C170 v4 harmonic-phase seed with validation-only selection."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--confirm-physical-order",
        action="store_true",
        help="Confirm the frozen S0-S3 physical channel order before training.",
    )
    return parser


def _variant_config(variant: str) -> dict[str, Any]:
    if variant not in VARIANT_CONFIG:
        raise ValueError(f"unknown v4 variant: {variant}")
    return dict(VARIANT_CONFIG[variant])


def _batches_for_epoch(
    groups: dict[tuple[int, int], np.ndarray],
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    """Use the frozen v2 exact no-replacement sampler."""
    return v2._batches_for_epoch(groups, seed, epoch)


def _save_numpy_state_dict(path: Path, state_dict: dict[str, torch.Tensor]) -> None:
    """Reuse the already-verified numeric NPZ writer with no pickle payload."""
    v3._save_numpy_state_dict(path, state_dict)


def _load_numpy_state_dict(
    path: Path,
    expected_state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Reuse v3 allow_pickle=False and exact key/shape/dtype validation."""
    return v3._load_numpy_state_dict(path, expected_state_dict, device)


def _collect_code_hashes() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "model": PROJECT_DIR / "src" / "c170_harmonic_phase_model_v4.py",
        "shared_scale_v3_runner_helpers": Path(v3.__file__).resolve(),
        "shared_scale_preprocessing": PROJECT_DIR / "src" / "c170_shared_scale_preprocessing_v3.py",
        "ring_representation_import_dependency": PROJECT_DIR / "src" / "c170_ring_representation.py",
        "frozen_v2_training_helpers": Path(v2.__file__).resolve(),
        "dataset_preprocessing_helpers": Path(c170.__file__).resolve(),
        "preprocessing_transitive_signal_processing": PROJECT_DIR / "src" / "rotation_poc.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing computation dependencies: {missing}")
    return {name: v2._file_sha256(path) for name, path in files.items()}


def _model_parameter_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().to(device="cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_and_prepare_v4(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    data_variant = _variant_config(args.variant)["preprocessing_variant"]
    prep_args = SimpleNamespace(
        dataset=args.dataset,
        confirm_physical_order=args.confirm_physical_order,
        variant=data_variant,
        seed=int(args.seed),
    )
    dataset, splits, split_metadata, prepared = v3._validate_and_prepare_v3(prep_args, device)
    actual_scale = float(prepared["preprocessing"]["shared_sensor_scale"])
    if actual_scale != EXPECTED_SHARED_SCALE:
        raise ValueError(
            f"training-fitted shared scale changed: expected {EXPECTED_SHARED_SCALE!r}, got {actual_scale!r}"
        )
    split_metadata["v4_variant"] = args.variant
    prepared["preprocessing"]["shared_sensor_scale_required_by_v4"] = EXPECTED_SHARED_SCALE
    return dataset, splits, split_metadata, prepared


def _manifest_base(
    config: dict[str, Any],
    config_sha256: str,
    started_utc: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "format": "c170-ring-cuda-v4-run",
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
        "initialization": {
            "rule": "torch_manual_seed_seed_before_identical_architecture_construction",
            "trainable_parameter_sha256": None,
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
    variant = _variant_config(args.variant)
    config = {
        "variant": args.variant,
        "model_variant": variant["model_variant"],
        "active_feature_indices": variant["active_feature_indices"],
        "feature_names": list(FEATURE_NAMES),
        "shuffle_labels": variant["shuffle_labels"],
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
        "preprocessing": "v3_filter_per_trial_demean_one_shared_train_median_scale",
        "preprocessing_scale_required": EXPECTED_SHARED_SCALE,
        "preprocessing_scale_fit_split": "train",
        "feature_lags": [1, 4, 16, 64],
        "lag_boundary": "zero_prefix_no_circular_wrap",
        "architecture": {
            "stem": "conv1d_12_to_24_k9_stride2_pad4_biasfalse_gn4_silu",
            "residual_dilations": [1, 2, 4],
            "residual_block": "depthwise24_k7_gn4_silu_pointwise24_gn4_residual_silu",
            "temporal_pooling": "adaptive_avg_lengths_1_2_4_plus_population_std_unbiased_false",
            "dropout": 0.2,
            "classifier": "linear_192_to_4",
            "trainable_parameters": 5932,
            "initialization": "same_torch_manual_seed_and_parameter_shapes_for_all_variants",
        },
        "checkpoint_selection": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
        "safe_state_dict_format": "numpy_npz_numeric_arrays",
        "safe_state_dict_reload": "numpy_load_allow_pickle_false_then_strict_state_dict",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
        "physical_order_confirmed_by_operator": True,
    }
    config_sha256 = v2._canonical_json_sha256(config)
    output_dir = args.output_dir.resolve()
    c170.create_output_directory(output_dir)
    manifest = _manifest_base(config, config_sha256, started_utc, args, device)
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
        dataset, splits, split_metadata, prepared = _validate_and_prepare_v4(args, device)
        tensors = prepared["tensors"]
        preprocessing_seconds = time.monotonic() - preprocessing_started
        manifest["split"] = split_metadata
        manifest["preprocessing"] = prepared["preprocessing"]
        manifest["timing"]["preprocessing_seconds"] = preprocessing_seconds
        v2._atomic_write_json(manifest_path, manifest)

        # Reset the seed at model construction so all four fixed masks receive
        # identical trainable parameters for each registered seed.
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        model = HarmonicPhaseClassifierV4(args.variant, n_classes=4).to(device)
        manifest["initialization"]["trainable_parameter_sha256"] = _model_parameter_sha256(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
        curve_rows: list[dict[str, Any]] = []
        training_started = time.monotonic()
        best_checkpoint_path = output_dir / "best_checkpoint.pt"
        safe_state_dict_path = output_dir / "best_weights.npz"

        for epoch in range(args.max_epochs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_started = time.monotonic()
            model.train()
            batches = _batches_for_epoch(original_groups, args.seed, epoch)
            weighted_ce_epoch = torch.zeros((), dtype=torch.float32, device=device)
            for batch_global in batches:
                local_np = np.asarray([global_to_local[int(index)] for index in batch_global], dtype=np.int64)
                local_indices = torch.as_tensor(local_np, dtype=torch.long, device=device)
                samples = tensors["train_inputs"][local_indices]
                targets = tensors["train_targets"][local_indices]
                weights = tensors["train_weights"][local_indices]
                logits = model(samples)
                per_sample_ce = F.cross_entropy(
                    logits.float(), targets, reduction="none", label_smoothing=0.0
                )
                loss = v2.batch_objective(per_sample_ce, weights, None, use_supcon=False)
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
                state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                checkpoint_payload = {
                    "format": "c170-ring-cuda-v4-checkpoint",
                    "format_version": 1,
                    "variant": args.variant,
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
                # This metadata archive is preserved for provenance and never
                # loaded; selected weights are reloaded only from the safe NPZ.
                v2._atomic_torch_save(best_checkpoint_path, checkpoint_payload)
            else:
                bad_epochs += 1

            weighted_ce_value = float(weighted_ce_epoch.detach().cpu())
            position_f1 = {key: value["macro_f1"] for key, value in val_metrics["by_position_mm"].items()}
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
                    "val_unique_predicted_classes": val_metrics["overall"]["prediction_unique_class_count"],
                    "val_f1_by_position_json": json.dumps(position_f1, sort_keys=True),
                    "val_position_channel_map_json": json.dumps(
                        val_metrics["by_position_mm_and_channel_map"], sort_keys=True
                    ),
                }
            )
            v2._atomic_write_csv(output_dir / "learning_curve.csv", v2.CURVE_FIELDS, curve_rows)
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
                f"variant={args.variant} seed={args.seed} epoch={epoch + 1} updates={UPDATES_PER_EPOCH} "
                f"val_macro_f1={macro_f1:.6f} val_ce={val_cross_entropy:.6f} "
                f"epoch_seconds={epoch_seconds:.3f}",
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
            dataset, val_indices, final_val_predictions, final_val_probabilities, args.variant, args.seed
        )
        v2._atomic_write_csv(
            output_dir / "validation_predictions.csv", tuple(prediction_rows[0].keys()), prediction_rows
        )
        artifact_hashes = {
            "best_checkpoint": v2._file_sha256(best_checkpoint_path),
            "safe_state_dict": v2._file_sha256(safe_state_dict_path),
            "learning_curve": v2._file_sha256(output_dir / "learning_curve.csv"),
            "validation_predictions": v2._file_sha256(output_dir / "validation_predictions.csv"),
        }
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        manifest["status"] = "completed"
        manifest["timing"].update(
            {
                "training_seconds": time.monotonic() - training_started,
                "total_elapsed_seconds": time.monotonic() - started,
                "peak_cuda_memory_allocated_bytes": peak_memory,
                "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
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
                "format": "c170-ring-cuda-v4-complete",
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
            f"RUN_COMPLETE variant={args.variant} seed={args.seed} best_epoch={best_epoch + 1} "
            f"val_macro_f1={best_macro_f1:.6f} "
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
