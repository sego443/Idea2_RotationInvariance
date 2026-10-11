from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from scripts.runners import run_c170_experiment as c170
from c170_ring_representation import (
    RingRepresentationClassifier,
    build_balanced_batches,
    cross_position_supervised_contrastive_loss,
    projection_diagnostics,
)


TRAIN_DISTANCE_MM = (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)
VAL_DISTANCE_MM = (5, 40, 90, 130)
TEST_DISTANCE_MM = (15, 25, 110, 150)
VARIANTS = (
    "main_true",
    "main_block_permutation",
    "classification_only",
    "slot_readout",
)
MODEL_VARIANT_BY_EXPERIMENT = {
    "main_true": "main",
    "main_block_permutation": "main",
    "classification_only": "classification_only",
    "slot_readout": "slot_readout",
}
SEEDS = (17, 29, 43, 61, 89)
MAX_EPOCHS = 160
PATIENCE = 25
CPU_THREADS = 2
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.001
LABEL_SMOOTHING = 0.02
CONTRASTIVE_WEIGHT = 0.1
CONTRASTIVE_TEMPERATURE = 0.1


def build_experiment_plan(speed_probe: bool = False) -> dict[str, object]:
    return {
        "phase": "validation-only",
        "run_mode": "speed-probe" if speed_probe else "fixed-budget-candidate",
        "train_distance_mm": list(TRAIN_DISTANCE_MM),
        "val_distance_mm": list(VAL_DISTANCE_MM),
        "test_distance_mm": list(TEST_DISTANCE_MM),
        "excluded_distance_mm": [70],
        "variants": list(VARIANTS),
        "seeds": [17] if speed_probe else list(SEEDS),
        "max_epochs": 3 if speed_probe else MAX_EPOCHS,
        "patience": 3 if speed_probe else PATIENCE,
        "batch_size": 44,
        "cpu_threads": CPU_THREADS,
        "updates_per_epoch": 20,
        "max_updates_per_candidate": 60 if speed_probe else 3200,
        "max_total_updates": 240 if speed_probe else 64000,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "cross_entropy_label_smoothing": LABEL_SMOOTHING,
        "contrastive_weight": CONTRASTIVE_WEIGHT,
        "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
        "selection_criterion": [
            "highest validation macro-F1",
            "lowest validation cross-entropy when macro-F1 ties",
            "lowest seed when both validation measures tie",
        ],
        "test_metrics_computed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C170 ring temporal representation validation-only experiment")
    parser.add_argument("--phase", choices=("validation-only",), required=True)
    parser.add_argument("--dataset", type=Path, default=c170.DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-physical-order", action="store_true")
    parser.add_argument("--speed-probe", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_variant_for_experiment(experiment_name: str) -> str:
    try:
        return MODEL_VARIANT_BY_EXPERIMENT[experiment_name]
    except KeyError as error:
        raise ValueError(f"unknown C170 ring experiment: {experiment_name}") from error


def _ring_architecture(variant: str) -> dict[str, object]:
    return {
        "class": "RingRepresentationClassifier",
        "variant": variant,
        "input_shape": [500, 4],
        "shared_encoder_output": [4, 32, 125],
        "ring_dilations": [1, 2],
        "oriented_neighbor_transforms": {
            "+1": "independent shared-across-nodes temporal Conv1d",
            "-1": "independent shared-across-nodes temporal Conv1d",
        },
        "readout": "fixed_sensor_slots" if variant == "slot_readout" else "nodes_and_ordered_edges",
        "class_count": 4,
        "projection_dimension": 32,
    }


def _checkpoint_payload(
    model: RingRepresentationClassifier,
    variant: str,
    seed: int,
    label_shuffle_seed: int | None,
    best_epoch: int,
    best_macro_f1: float,
    best_cross_entropy: float,
    sensor_scales: np.ndarray,
    channel_map_confirmation: dict[str, list[str]],
    split_distance_mm: dict[str, list[int]],
    dataset_path: Path,
    dataset_hash: str,
    plan: dict[str, object],
) -> dict[str, object]:
    model_variant = _model_variant_for_experiment(variant)
    return {
        "format": "c170-ring-checkpoint",
        "format_version": 1,
        "model_kind": model_variant,
        "model_name": f"c170_ring_{model_variant}",
        "architecture": _ring_architecture(model_variant),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "class_order": list(c170.DIRECTIONS),
        "training_hyperparameters": {
            "seed": int(seed),
            "numpy_seed": int(seed),
            "torch_seed": int(seed),
            "label_shuffle_seed": label_shuffle_seed,
            "max_epochs": int(plan["max_epochs"]),
            "patience": int(plan["patience"]),
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "contrastive_weight": CONTRASTIVE_WEIGHT if variant.startswith("main_") else 0.0,
            "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
            "batch_size": 44,
        },
        "preprocessing": {
            "sample_rate_hz": 250,
            "filter_band_hz": [0.5, 40.0],
            "filter_type": "butterworth_bandpass_sosfiltfilt_order_3",
            "dc_removal": "subtract_per_trial_per_channel_temporal_mean",
            "sensor_scales_source": "training_split_median_absolute_centered_signal",
            "training_derived_sensor_scales": [float(value) for value in sensor_scales],
            "model_input_transform": "identity",
        },
        "experiment_name": variant,
        "selection_basis": {
            "criterion": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
            "validation_macro_f1": float(best_macro_f1),
            "validation_cross_entropy": float(best_cross_entropy),
            "best_epoch_zero_based": int(best_epoch),
        },
        "channel_map_confirmation": channel_map_confirmation,
        "physical_order_confirmation_source": (
            "Both channel maps correspond to S0,S1,S2,S3, as confirmed in the C170 acceptance record"
        ),
        "split_distance_mm": split_distance_mm,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_hash,
    }


def load_ring_checkpoint(path: Path) -> tuple[RingRepresentationClassifier, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "c170-ring-checkpoint" or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if payload.get("class_order") != list(c170.DIRECTIONS):
        raise ValueError(f"Checkpoint class order mismatch: {path}")
    variant = str(payload["model_kind"])
    if variant not in {"main", "classification_only", "slot_readout"}:
        raise ValueError(f"Unsupported checkpoint model kind: {path}")
    if payload.get("architecture") != _ring_architecture(variant):
        raise ValueError(f"Checkpoint architecture record mismatch: {path}")
    model = RingRepresentationClassifier(variant)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def _full_metrics(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    target_labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    return c170.evaluation_metrics(
        target_labels,
        predictions,
        dataset["distance_mm"][indices],
        dataset["channel_map"][indices],
    )


def _prediction_rows(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    predictions: np.ndarray,
    variant: str,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    model_variant = _model_variant_for_experiment(variant)
    for index, prediction in zip(indices, predictions, strict=True):
        rows.append(
            {
                "experiment": variant,
                "model_kind": model_variant,
                "model_name": f"c170_ring_{model_variant}",
                "seed": int(seed),
                "split": "val",
                "sample_index": int(dataset["sample_index"][index]),
                "block": int(dataset["block"][index]),
                "repetition": int(dataset["repetition"][index]),
                "source_repetition": int(dataset["source_repetition"][index]),
                "distance_mm": int(dataset["distance_mm"][index]),
                "channel_map": ",".join(
                    str(int(channel)) for channel in dataset["channel_map"][index]
                ),
                "truth": str(dataset["direction"][index]),
                "prediction": c170.DIRECTIONS[int(prediction)],
            }
        )
    return rows


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "experiment",
        "model_kind",
        "model_name",
        "seed",
        "split",
        "sample_index",
        "block",
        "repetition",
        "source_repetition",
        "distance_mm",
        "channel_map",
        "truth",
        "prediction",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate(
    model: RingRepresentationClassifier,
    inputs: torch.Tensor,
    labels: np.ndarray,
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, dict[str, object]]:
    model.eval()
    target = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    with torch.no_grad():
        logits = model(inputs)
        loss = float(criterion(logits, target))
    predictions = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    return loss, predictions, _full_metrics(dataset, indices, labels, predictions)


def _build_training_schedule(
    dataset: dict[str, np.ndarray],
    train_indices: np.ndarray,
    variant: str,
    seed: int,
    epoch_count: int,
) -> tuple[np.ndarray, list[list[np.ndarray]], int | None]:
    train_targets_all = dataset["labels"].copy()
    label_shuffle_seed = None
    if variant == "main_block_permutation":
        label_shuffle_seed = seed + 10_000
        train_targets_all = c170.shuffle_training_labels_by_recording(
            dataset["labels"],
            dataset["distance_mm"],
            dataset["block"],
            train_indices,
            seed=label_shuffle_seed,
        )
    elif variant not in MODEL_VARIANT_BY_EXPERIMENT:
        raise ValueError(f"unknown C170 ring experiment: {variant}")

    batches_by_epoch = [
        build_balanced_batches(
            train_indices,
            train_targets_all,
            dataset["distance_mm"],
            dataset["block"],
            seed=seed + epoch,
        )
        for epoch in range(epoch_count)
    ]
    return train_targets_all, batches_by_epoch, label_shuffle_seed


def _training_losses(
    logits: torch.Tensor,
    projection: torch.Tensor | None,
    batch_targets: torch.Tensor,
    batch_distances: np.ndarray,
    batch_blocks: np.ndarray,
    criterion: nn.Module,
    *,
    contrastive_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    ce_loss = criterion(logits, batch_targets)
    if not contrastive_enabled:
        if projection is not None:
            raise ValueError("classification-only training must not receive projections")
        return ce_loss, ce_loss, None
    if projection is None:
        raise ValueError("contrastive training requires model projections")
    contrastive_loss = cross_position_supervised_contrastive_loss(
        projection,
        batch_targets,
        batch_distances,
        batch_blocks,
        temperature=CONTRASTIVE_TEMPERATURE,
    )
    return ce_loss, ce_loss + CONTRASTIVE_WEIGHT * contrastive_loss, contrastive_loss


def _train_candidate(
    *,
    dataset: dict[str, np.ndarray],
    splits: dict[str, np.ndarray],
    normalized: dict[str, np.ndarray],
    sensor_scales: np.ndarray,
    channel_map_confirmation: dict[str, list[str]],
    dataset_path: Path,
    dataset_hash: str,
    output_dir: Path,
    plan: dict[str, object],
    variant: str,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(plan["cpu_threads"]))
    train_indices = splits["train"]
    val_indices = splits["val"]
    train_targets_all, batches_by_epoch, label_shuffle_seed = _build_training_schedule(
        dataset,
        train_indices,
        variant=variant,
        seed=seed,
        epoch_count=int(plan["max_epochs"]),
    )
    train_offset = {int(index): offset for offset, index in enumerate(train_indices)}
    train_inputs = torch.from_numpy(normalized["train"])
    val_inputs = torch.from_numpy(normalized["val"])
    train_targets = train_targets_all[train_indices].astype(np.int64)
    val_targets = dataset["labels"][val_indices].astype(np.int64)
    model = RingRepresentationClassifier(_model_variant_for_experiment(variant))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    candidate_dir = output_dir / "runs" / f"{variant}__seed-{seed}"
    candidate_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = candidate_dir / "checkpoint.pt"
    curve: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    best_cross_entropy = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    update_count = 0
    contrastive_enabled = variant.startswith("main_")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    started = time.perf_counter()

    for epoch in range(int(plan["max_epochs"])):
        model.train()
        epoch_ce: list[float] = []
        epoch_contrastive: list[float] = []
        for global_batch in batches_by_epoch[epoch]:
            offsets = np.asarray(
                [train_offset[int(index)] for index in global_batch], dtype=np.int64
            )
            batch_inputs = train_inputs[offsets]
            batch_targets = torch.from_numpy(
                train_targets_all[global_batch].astype(np.int64)
            )
            if contrastive_enabled:
                logits, projection = model.forward_with_projection(batch_inputs)
            else:
                logits = model(batch_inputs)
                projection = None
            ce_loss, total_loss, contrastive_loss = _training_losses(
                logits,
                projection,
                batch_targets,
                dataset["distance_mm"][global_batch],
                dataset["block"][global_batch],
                criterion,
                contrastive_enabled=contrastive_enabled,
            )
            contrastive_loss_value = (
                float(contrastive_loss.detach()) if contrastive_loss is not None else 0.0
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()
            update_count += 1
            epoch_ce.append(float(ce_loss.detach()))
            epoch_contrastive.append(contrastive_loss_value)

        train_ce, train_predictions, train_metrics = _evaluate(
            model, train_inputs, train_targets, dataset, train_indices, criterion
        )
        val_ce, val_predictions, val_metrics = _evaluate(
            model, val_inputs, val_targets, dataset, val_indices, criterion
        )
        val_macro_f1 = float(val_metrics["overall"]["macro_f1"])
        improved = val_macro_f1 > best_macro_f1 or (
            val_macro_f1 == best_macro_f1 and val_ce < best_cross_entropy
        )
        row = {
            "epoch": epoch,
            "updates_completed": update_count,
            "train_batch_cross_entropy_mean": float(np.mean(epoch_ce)),
            "train_batch_contrastive_loss_mean": float(np.mean(epoch_contrastive)),
            "train_full_cross_entropy": train_ce,
            "train_target_metrics": train_metrics,
            "validation_cross_entropy": val_ce,
            "validation_metrics": val_metrics,
            "best_epoch_so_far": epoch if improved else best_epoch,
        }
        curve.append(row)
        _write_json(candidate_dir / "learning_curve.json", curve)

        if improved:
            best_macro_f1 = val_macro_f1
            best_cross_entropy = val_ce
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            checkpoint_payload = _checkpoint_payload(
                model=model,
                variant=variant,
                seed=seed,
                label_shuffle_seed=label_shuffle_seed,
                best_epoch=best_epoch,
                best_macro_f1=best_macro_f1,
                best_cross_entropy=best_cross_entropy,
                sensor_scales=sensor_scales,
                channel_map_confirmation=channel_map_confirmation,
                split_distance_mm={
                    name: sorted(
                        np.unique(dataset["distance_mm"][indices]).astype(int).tolist()
                    )
                    for name, indices in splits.items()
                },
                dataset_path=dataset_path,
                dataset_hash=dataset_hash,
                plan=plan,
            )
            torch.save(checkpoint_payload, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(plan["patience"]):
            break

    elapsed_seconds = time.perf_counter() - started
    reloaded_model, checkpoint_payload = load_ring_checkpoint(checkpoint_path)
    reloaded_val_logits = reloaded_model(val_inputs)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    with torch.no_grad():
        in_memory_val_logits = model(val_inputs)
    torch.testing.assert_close(
        reloaded_val_logits,
        in_memory_val_logits,
        rtol=0.0,
        atol=0.0,
    )
    val_predictions = reloaded_val_logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    validation_metrics = _full_metrics(
        dataset, val_indices, val_targets, val_predictions
    )
    prediction_rows = _prediction_rows(dataset, val_indices, val_predictions, variant, seed)
    _write_predictions(candidate_dir / "validation_predictions.csv", prediction_rows)

    diagnostics: dict[str, object] | None = None
    if contrastive_enabled:
        with torch.no_grad():
            _, train_projection = reloaded_model.forward_with_projection(train_inputs)
        diagnostics = projection_diagnostics(train_projection, train_targets)

    checkpoint_hash = _sha256(checkpoint_path)
    run_record = {
        "variant": variant,
        "seed": int(seed),
        "label_shuffle_seed": label_shuffle_seed,
        "best_epoch_zero_based": int(best_epoch),
        "best_validation_macro_f1": float(best_macro_f1),
        "best_validation_cross_entropy": float(best_cross_entropy),
        "actual_epochs": len(curve),
        "actual_optimizer_updates": update_count,
        "parameter_count": int(parameter_count),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_optimizer_update": elapsed_seconds / update_count,
        "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_reload_logits_exact": True,
        "checkpoint_reload_max_abs_logit_difference": float(
            torch.max(torch.abs(reloaded_val_logits - in_memory_val_logits))
        ),
        "validation_metrics": validation_metrics,
        "projection_diagnostics": diagnostics,
        "train_curve_path": str((candidate_dir / "learning_curve.json").relative_to(output_dir)),
        "validation_predictions_path": str(
            (candidate_dir / "validation_predictions.csv").relative_to(output_dir)
        ),
    }
    _write_json(candidate_dir / "run.json", run_record)
    return run_record


def run_validation_phase(
    dataset_path: Path,
    output_dir: Path,
    confirm_physical_order: bool,
    speed_probe: bool,
) -> dict[str, object]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    plan = build_experiment_plan(speed_probe=speed_probe)
    dataset_hash = c170.dataset_sha256(dataset_path)
    dataset = c170.load_dataset(dataset_path)
    if c170.dataset_sha256(dataset_path) != dataset_hash:
        raise RuntimeError("C170 dataset changed while being read")
    assignments = {
        "train": plan["train_distance_mm"],
        "val": plan["val_distance_mm"],
        "test": plan["test_distance_mm"],
    }
    splits = c170.split_by_distance(dataset["distance_mm"], assignments)
    c170.assert_recording_groups_intact(dataset)
    c170.assert_no_distance_crossing(dataset["distance_mm"], splits)
    labels = dataset["labels"]
    for split_name in ("train", "val"):
        if not np.array_equal(np.unique(labels[splits[split_name]]), np.arange(4)):
            raise ValueError(f"{split_name} split does not contain all four direction classes")

    train_val_indices = np.concatenate([splits["train"], splits["val"]])
    channel_map_confirmation = c170.validate_channel_maps(
        dataset["channel_map"][train_val_indices],
        confirm_physical_order=confirm_physical_order,
    )
    normalized, sensor_scales = c170.prepare_model_inputs(dataset["X_raw"], splits)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "runs").mkdir(exist_ok=False)
    split_distance_mm = {
        "train": list(TRAIN_DISTANCE_MM),
        "val": list(VAL_DISTANCE_MM),
        "test": list(TEST_DISTANCE_MM),
    }
    config = {
        **plan,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_hash,
        "sample_count_after_duplicate_exclusion": int(len(dataset["X_raw"])),
        "duplicate_rows_excluded": int(dataset["duplicate_rows_excluded"]),
        "split_distance_mm": split_distance_mm,
        "split_sample_counts": {
            "train": int(len(splits["train"])),
            "val": int(len(splits["val"])),
        },
        "channel_map_confirmation": channel_map_confirmation,
        "confirm_physical_order_flag": bool(confirm_physical_order),
        "direction_order": list(c170.DIRECTIONS),
        "sample_rate_hz": 250,
        "filter_band_hz": [0.5, 40.0],
        "training_derived_sensor_scales": [float(value) for value in sensor_scales],
        "tensor_shapes": {
            "input": ["B", 500, 4],
            "shared_encoder_and_ring": ["B", 4, 32, 125],
        },
        "results": [],
        "validation_metrics_only": True,
    }
    _write_json(output_dir / "config.json", config)
    progress_path = output_dir / "progress.jsonl"
    progress_path.touch(exist_ok=False)
    candidate_count = len(plan["variants"]) * len(plan["seeds"])
    completed = 0
    run_records: list[dict[str, object]] = []
    for variant in plan["variants"]:
        for seed in plan["seeds"]:
            run_record = _train_candidate(
                dataset=dataset,
                splits=splits,
                normalized=normalized,
                sensor_scales=sensor_scales,
                channel_map_confirmation=channel_map_confirmation,
                dataset_path=dataset_path,
                dataset_hash=dataset_hash,
                output_dir=output_dir,
                plan=plan,
                variant=str(variant),
                seed=int(seed),
            )
            run_records.append(run_record)
            completed += 1
            progress_record = {
                "completed_candidates": completed,
                "candidate_count": candidate_count,
                "variant": variant,
                "seed": int(seed),
                "actual_optimizer_updates": run_record["actual_optimizer_updates"],
                "elapsed_seconds": run_record["elapsed_seconds"],
                "checkpoint_path": run_record["checkpoint_path"],
            }
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(json.dumps(progress_record, ensure_ascii=False) + "\n")
                progress.flush()
            config["results"] = run_records
            _write_json(output_dir / "config.json", config)
            print(json.dumps(progress_record, ensure_ascii=False), flush=True)

    if speed_probe:
        seconds_per_update = {
            str(record["variant"]): float(record["seconds_per_optimizer_update"])
            for record in run_records
        }
        estimated_full_seconds = sum(
            seconds_per_update[variant] * len(SEEDS) * 3200
            for variant in VARIANTS
        )
        config["formal_budget_estimate"] = {
            "max_variants": len(VARIANTS),
            "seeds_per_variant": len(SEEDS),
            "max_epochs_per_candidate": MAX_EPOCHS,
            "updates_per_epoch": 20,
            "maximum_optimizer_updates": len(VARIANTS) * len(SEEDS) * MAX_EPOCHS * 20,
            "estimated_seconds_by_variant": {
                variant: seconds_per_update[variant] * len(SEEDS) * 3200
                for variant in VARIANTS
            },
            "estimated_maximum_wall_seconds": estimated_full_seconds,
            "estimated_maximum_wall_hours": estimated_full_seconds / 3600.0,
            "basis": "Measured time per optimizer update in a real C170 train/validation speed probe; early-stopping savings are excluded",
        }
    _write_json(output_dir / "config.json", config)
    return {"config": config, "runs": run_records}


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_validation_phase(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        confirm_physical_order=args.confirm_physical_order,
        speed_probe=args.speed_probe,
    )
    print(
        json.dumps(
            {
                "phase": "validation-only",
                "output_dir": str(args.output_dir),
                "candidate_count": len(result["runs"]),
                "maximum_optimizer_updates": result["config"]["max_total_updates"],
                "formal_budget_estimate": result["config"].get("formal_budget_estimate"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
