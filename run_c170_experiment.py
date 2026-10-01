from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rotation_poc import (
    OrbitCNN,
    TinyTemporalCNN,
    calibrate_windows,
    filter_windows,
    harmonic_invariants,
    macro_metrics,
    predict_temporal_model,
    train_temporal_model,
)


DEFAULT_DATASET_PATH = PROJECT_DIR / "data" / "processed_c170_20260926" / "dataset.npz"
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DIRECTION_TO_ID = {direction: index for index, direction in enumerate(DIRECTIONS)}
DEFAULT_SEEDS = (17, 29, 43, 61, 89)
MODEL_KINDS = ("raw", "orbit", "harmonic", "orbit_prob")
DEFAULT_MODEL_KINDS = ("raw", "orbit")
SPLIT_NAMES = ("train", "val", "test")
PHYSICAL_SENSOR_ORDER = ("S0", "S1", "S2", "S3")
CONFIRMED_CHANNEL_MAPS = {
    (0, 2, 4, 6): PHYSICAL_SENSOR_ORDER,
    (0, 1, 2, 3): PHYSICAL_SENSOR_ORDER,
}


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C170 position-holdout movement recognition experiment")
    parser.add_argument("--phase", choices=("validation-only", "final-test"), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-distance-mm", type=int, nargs="+")
    parser.add_argument("--val-distance-mm", type=int, nargs="+")
    parser.add_argument("--test-distance-mm", type=int, nargs="+")
    parser.add_argument("--confirm-physical-order", action="store_true")
    parser.add_argument("--model-kinds", nargs="+", choices=MODEL_KINDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    return parser.parse_args(argv)


def validate_phase_args(args: argparse.Namespace) -> None:
    positions = (args.train_distance_mm, args.val_distance_mm, args.test_distance_mm)
    if args.phase == "validation-only" and any(value is None for value in positions):
        raise ValueError("validation-only requires explicit train, validation, and test positions")
    if args.phase == "final-test" and any(value is not None for value in positions):
        raise ValueError("final-test reads positions from the validation-only configuration and does not accept position arguments")
    if args.phase == "final-test" and args.confirm_physical_order:
        raise ValueError("final-test reads the channel_map confirmation from the validation-only configuration")
    if args.model_kinds is not None and len(set(args.model_kinds)) != len(args.model_kinds):
        raise ValueError("model-kinds must be unique")


def load_dataset(path: Path) -> dict[str, np.ndarray | int]:
    required = (
        "X_raw",
        "block",
        "distance_mm",
        "direction",
        "repetition",
        "source_repetition",
        "is_duplicate",
        "channel_map",
    )
    with np.load(path) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            raise ValueError(f"dataset.npz is missing required fields: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}

    sample_count = len(arrays["X_raw"])
    if arrays["X_raw"].ndim != 3 or arrays["X_raw"].shape[1:] != (500, 4):
        raise ValueError(f"X_raw must have shape (samples, 500, 4); actual shape is {arrays['X_raw'].shape}")
    if any(len(values) != sample_count for values in arrays.values()):
        raise ValueError("dataset.npz fields have inconsistent sample counts")
    if arrays["channel_map"].shape != (sample_count, 4):
        raise ValueError("channel_map must contain four original EXG column indices per sample")
    if not np.isfinite(arrays["X_raw"]).all():
        raise ValueError("X_raw contains non-finite values")

    keep = ~arrays["is_duplicate"].astype(bool)
    dataset = {name: values[keep].copy() for name, values in arrays.items()}
    dataset["is_duplicate"] = np.zeros(int(keep.sum()), dtype=bool)
    dataset["sample_index"] = np.flatnonzero(keep).astype(np.int64)
    dataset["labels"] = encode_directions(dataset["direction"])
    dataset["duplicate_rows_excluded"] = int((~keep).sum())
    return dataset


def encode_directions(directions: np.ndarray) -> np.ndarray:
    values = np.asarray(directions).astype(str)
    unknown = sorted(set(values.tolist()) - set(DIRECTION_TO_ID))
    if unknown:
        raise ValueError(f"direction contains undefined classes: {unknown}")
    return np.asarray([DIRECTION_TO_ID[value] for value in values], dtype=np.int64)


def split_by_distance(
    distance_mm: np.ndarray,
    assignments: dict[str, Sequence[int]],
) -> dict[str, np.ndarray]:
    distances = np.asarray(distance_mm, dtype=np.int64)
    if distances.ndim != 1:
        raise ValueError("distance_mm must be one-dimensional")
    if set(assignments) != set(SPLIT_NAMES):
        raise ValueError(f"Position splits must contain exactly {SPLIT_NAMES}")

    normalized = {name: {int(position) for position in assignments[name]} for name in SPLIT_NAMES}
    if any(not positions for positions in normalized.values()):
        raise ValueError("Train, validation, and test must each contain at least one position")
    assigned: dict[int, str] = {}
    for split_name in SPLIT_NAMES:
        for position in normalized[split_name]:
            if position in assigned:
                raise ValueError(f"Position {position} mm is assigned to multiple splits")
            assigned[position] = split_name

    available = set(np.unique(distances).tolist())
    missing = sorted(set(assigned) - available)
    if missing:
        raise ValueError(f"dataset.npz does not contain the requested positions: {missing}")

    splits = {
        split_name: np.flatnonzero(np.isin(distances, sorted(normalized[split_name])))
        for split_name in SPLIT_NAMES
    }
    assert_no_distance_crossing(distances, splits)
    return splits


def assert_no_distance_crossing(
    distance_mm: np.ndarray,
    splits: dict[str, np.ndarray],
) -> None:
    distances = np.asarray(distance_mm, dtype=np.int64)
    split_positions: dict[str, set[int]] = {}
    split_indices: dict[str, set[int]] = {}
    for split_name in SPLIT_NAMES:
        indices = np.asarray(splits[split_name], dtype=np.int64)
        if indices.ndim != 1 or not len(indices):
            raise ValueError(f"{split_name} split has no samples")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{split_name} split contains duplicate samples")
        if np.any(indices < 0) or np.any(indices >= len(distances)):
            raise ValueError(f"{split_name} split contains invalid sample indices")
        positions = set(distances[indices].tolist())
        complete_indices = np.flatnonzero(np.isin(distances, sorted(positions)))
        if not np.array_equal(np.sort(indices), complete_indices):
            raise ValueError(f"{split_name} split does not contain complete distance_mm groups")
        split_positions[split_name] = positions
        split_indices[split_name] = set(indices.tolist())

    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1 :]:
            if split_positions[left_name] & split_positions[right_name]:
                raise ValueError(f"{left_name} and {right_name} have overlapping distance_mm positions")
            if split_indices[left_name] & split_indices[right_name]:
                raise ValueError(f"{left_name} and {right_name} contain overlapping samples")


def validate_channel_maps(
    channel_maps: np.ndarray,
    confirm_physical_order: bool = False,
) -> dict[str, list[str]]:
    values = np.asarray(channel_maps)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise ValueError("Selected channel_map must contain four EXG column indices")
    observed = sorted({tuple(int(channel) for channel in row) for row in values.tolist()})
    unknown = [mapping for mapping in observed if mapping not in CONFIRMED_CHANNEL_MAPS]
    if unknown:
        raise ValueError(f"channel_map has not been verified against physical sensor order: {unknown}")
    if len(observed) > 1 and not confirm_physical_order:
        raise ValueError(
            "Selected positions contain multiple channel maps; verify physical sensor order and pass --confirm-physical-order"
        )
    return {
        ",".join(str(channel) for channel in mapping): list(CONFIRMED_CHANNEL_MAPS[mapping])
        for mapping in observed
    }


def assert_recording_groups_intact(dataset: dict[str, np.ndarray]) -> None:
    blocks = dataset["block"]
    distances = dataset["distance_mm"]
    labels = dataset["labels"]
    channel_maps = dataset["channel_map"]
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        if len(np.unique(distances[indices])) != 1:
            raise ValueError(f"block {int(block)} spans multiple distance_mm positions")
        if len(np.unique(labels[indices])) != 1:
            raise ValueError(f"block {int(block)} contains multiple direction classes")
        if len(np.unique(channel_maps[indices], axis=0)) != 1:
            raise ValueError(f"block {int(block)} contains multiple channel maps")


def shuffle_training_labels_by_recording(
    labels: np.ndarray,
    distance_mm: np.ndarray,
    blocks: np.ndarray,
    training_indices: np.ndarray,
    seed: int,
) -> np.ndarray:
    true_labels = np.asarray(labels, dtype=np.int64)
    distances = np.asarray(distance_mm, dtype=np.int64)
    block_ids = np.asarray(blocks, dtype=np.int64)
    indices = np.asarray(training_indices, dtype=np.int64)
    if true_labels.ndim != 1 or distances.shape != true_labels.shape or block_ids.shape != true_labels.shape:
        raise ValueError("labels, distance_mm, and block must have the same one-dimensional shape")
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices):
        raise ValueError("Training sample indices must be one-dimensional and unique")
    if np.any(indices < 0) or np.any(indices >= len(true_labels)):
        raise ValueError("Training sample indices are out of range")

    shuffled = true_labels.copy()
    rng = np.random.default_rng(seed)
    for position in np.unique(distances[indices]):
        position_indices = indices[distances[indices] == position]
        groups: list[tuple[int, np.ndarray, int]] = []
        for block in sorted(np.unique(block_ids[position_indices]).tolist()):
            group_indices = position_indices[block_ids[position_indices] == block]
            complete_group = np.flatnonzero(block_ids == block)
            if not np.array_equal(np.sort(group_indices), complete_group):
                raise ValueError(f"Training split does not contain the complete recording group for block {int(block)}")
            group_labels = np.unique(true_labels[group_indices])
            group_positions = np.unique(distances[group_indices])
            if len(group_labels) != 1 or len(group_positions) != 1:
                raise ValueError(f"block {int(block)} has non-unique labels or positions")
            groups.append((int(block), group_indices, int(group_labels[0])))
        if len(groups) != len(DIRECTIONS):
            raise ValueError(f"Position {int(position)} mm must contain four complete recording blocks")

        source_labels = np.asarray([group[2] for group in groups], dtype=np.int64)
        if not np.array_equal(np.sort(source_labels), np.arange(len(DIRECTIONS))):
            raise ValueError(f"The four blocks at position {int(position)} mm must each correspond to one direction")
        permutation = rng.permutation(len(groups))
        if np.any(permutation == np.arange(len(groups))):
            permutation = np.roll(np.arange(len(groups)), int(rng.integers(1, len(groups))))
        shuffled_labels = source_labels[permutation]
        for (_, group_indices, _), assigned_label in zip(groups, shuffled_labels, strict=True):
            shuffled[group_indices] = assigned_label

        before = np.bincount(source_labels, minlength=len(DIRECTIONS))
        after = np.bincount(shuffled_labels, minlength=len(DIRECTIONS))
        if not np.array_equal(before, after):
            raise RuntimeError(f"Direction-block counts changed at position {int(position)} mm")
        for block, group_indices, original_label in groups:
            if np.unique(shuffled[group_indices]).item() == original_label:
                raise RuntimeError(f"Block {block} at position {int(position)} mm did not change direction")

    if np.array_equal(shuffled[indices], true_labels[indices]):
        raise ValueError("No recording groups can be permuted at the training positions")
    return shuffled


def prepare_model_inputs(
    windows: np.ndarray,
    splits: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train_indices = splits["train"]
    val_indices = splits["val"]
    filtered = filter_windows(windows[np.concatenate([train_indices, val_indices])], sample_rate=250)
    train_filtered = filtered[: len(train_indices)]
    val_filtered = filtered[len(train_indices) :]
    train_normalized, sensor_scales = calibrate_windows(train_filtered)
    val_normalized, _ = calibrate_windows(val_filtered, scale=sensor_scales)
    return {
        "train": train_normalized.astype(np.float32),
        "val": val_normalized.astype(np.float32),
    }, sensor_scales


def prepare_candidate_inputs(samples: np.ndarray, model_kind: str) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("Candidate model inputs must contain four sensor channels")
    if model_kind == "harmonic":
        return harmonic_invariants(values)
    if model_kind in {"raw", "orbit", "orbit_prob"}:
        return values
    raise ValueError(f"Unknown model_kind: {model_kind}")


def model_display_name(model_kind: str) -> str:
    names = {
        "raw": "cnn",
        "orbit": "orbit_cnn",
        "harmonic": "harmonic_cnn",
        "orbit_prob": "orbit_probability_cnn",
    }
    try:
        return names[model_kind]
    except KeyError as error:
        raise ValueError(f"Unknown model_kind: {model_kind}") from error


def _metric_bundle(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    metrics = macro_metrics(labels, predictions, n_classes=len(DIRECTIONS))
    metrics["prediction_unique_class_count"] = int(
        np.count_nonzero(metrics["prediction_counts"])
    )
    return metrics


def evaluation_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    distance_mm: np.ndarray,
    channel_maps: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    distances = np.asarray(distance_mm, dtype=np.int64)
    maps = np.asarray(channel_maps, dtype=np.int64)
    if truth.ndim != 1 or predicted.shape != truth.shape or distances.shape != truth.shape:
        raise ValueError("Evaluation labels, predictions, and positions must have the same one-dimensional shape")
    if maps.shape != (len(truth), 4) or not len(truth):
        raise ValueError("Evaluation channel_map must provide four EXG column indices per sample")

    by_distance: dict[str, object] = {}
    by_distance_and_map: dict[str, dict[str, object]] = {}
    for position in sorted(np.unique(distances).tolist()):
        position_mask = distances == position
        by_distance[str(int(position))] = _metric_bundle(
            truth[position_mask], predicted[position_mask]
        )
        maps_at_position: dict[str, object] = {}
        for mapping in sorted({tuple(row.tolist()) for row in maps[position_mask]}):
            map_mask = position_mask & np.all(maps == np.asarray(mapping), axis=1)
            map_name = ",".join(str(int(channel)) for channel in mapping)
            maps_at_position[map_name] = _metric_bundle(truth[map_mask], predicted[map_mask])
        by_distance_and_map[str(int(position))] = maps_at_position

    return {
        "overall": _metric_bundle(truth, predicted),
        "by_distance_mm": by_distance,
        "by_distance_mm_and_channel_map": by_distance_and_map,
    }


def create_output_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    return path


def _prediction_rows(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    predictions: np.ndarray,
    experiment: str,
    model_kind: str,
    seed: int,
    split_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, prediction in zip(indices, predictions, strict=True):
        map_name = ",".join(str(int(channel)) for channel in dataset["channel_map"][index])
        rows.append(
            {
                "experiment": experiment,
                "model_kind": model_kind,
                "model_name": model_display_name(model_kind),
                "seed": seed,
                "split": split_name,
                "sample_index": int(dataset["sample_index"][index]),
                "block": int(dataset["block"][index]),
                "repetition": int(dataset["repetition"][index]),
                "source_repetition": int(dataset["source_repetition"][index]),
                "distance_mm": int(dataset["distance_mm"][index]),
                "channel_map": map_name,
                "truth": str(dataset["direction"][index]),
                "prediction": DIRECTIONS[int(prediction)],
            }
        )
    return rows


def _metric_row(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    predictions: np.ndarray,
    experiment: str,
    model_kind: str,
    seed: int,
    split_name: str,
    best_epoch: int,
    label_shuffle_seed: int | None,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "model_kind": model_kind,
        "model_name": model_display_name(model_kind),
        "seed": seed,
        "split": split_name,
        "best_epoch": best_epoch,
        "label_shuffle_seed": label_shuffle_seed,
        **evaluation_metrics(
            dataset["labels"][indices],
            predictions,
            dataset["distance_mm"][indices],
            dataset["channel_map"][indices],
        ),
    }


def _mean_bundle(bundles: list[dict[str, object]]) -> dict[str, object]:
    recalls = np.asarray([bundle["per_class_recall"] for bundle in bundles], dtype=float)
    f1_values = np.asarray([bundle["macro_f1"] for bundle in bundles], dtype=float)
    accuracy_values = np.asarray([bundle["accuracy"] for bundle in bundles], dtype=float)
    unique_values = np.asarray([bundle["prediction_unique_class_count"] for bundle in bundles], dtype=float)
    return {
        "accuracy_mean": float(accuracy_values.mean()),
        "macro_f1_mean": float(f1_values.mean()),
        "macro_f1_std": float(f1_values.std(ddof=1)) if len(f1_values) > 1 else 0.0,
        "per_class_recall_mean": recalls.mean(axis=0).tolist(),
        "prediction_unique_class_count_mean": float(unique_values.mean()),
    }


def summarize_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    keys = sorted(
        {
            (str(row["experiment"]), str(row["model_kind"]), str(row["split"]))
            for row in rows
        }
    )
    for experiment, model_kind, split_name in keys:
        selected = [
            row
            for row in rows
            if row["experiment"] == experiment
            and row["model_kind"] == model_kind
            and row["split"] == split_name
        ]
        first = selected[0]
        by_distance = {
            position: _mean_bundle([row["by_distance_mm"][position] for row in selected])
            for position in first["by_distance_mm"]
        }
        by_distance_and_map = {
            position: {
                mapping: _mean_bundle(
                    [row["by_distance_mm_and_channel_map"][position][mapping] for row in selected]
                )
                for mapping in first["by_distance_mm_and_channel_map"][position]
            }
            for position in first["by_distance_mm_and_channel_map"]
        }
        summaries.append(
            {
                "experiment": experiment,
                "model_kind": model_kind,
                "model_name": first["model_name"],
                "split": split_name,
                "seeds": [int(row["seed"]) for row in selected],
                "overall": _mean_bundle([row["overall"] for row in selected]),
                "by_distance_mm": by_distance,
                "by_distance_mm_and_channel_map": by_distance_and_map,
            }
        )
    return summaries


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
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def build_model(model_kind: str) -> nn.Module:
    if model_kind in {"raw", "harmonic"}:
        in_channels = 3 if model_kind == "harmonic" else 4
        return TinyTemporalCNN(in_channels=in_channels, n_classes=len(DIRECTIONS))
    if model_kind in {"orbit", "orbit_prob"}:
        base = TinyTemporalCNN(in_channels=4, n_classes=len(DIRECTIONS))
        aggregation = "mean_logits" if model_kind == "orbit" else "mean_probabilities"
        return OrbitCNN(base, aggregation=aggregation)
    raise ValueError(f"Unknown model_kind: {model_kind}")


def model_architecture(model_kind: str) -> dict[str, object]:
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"Unknown model_kind: {model_kind}")
    architecture: dict[str, object] = {
        "class": "OrbitCNN" if model_kind in {"orbit", "orbit_prob"} else "TinyTemporalCNN",
        "model_kind": model_kind,
        "in_channels": 3 if model_kind == "harmonic" else 4,
        "n_classes": len(DIRECTIONS),
        "temporal_cnn": {
            "conv_channels": [12, 20, 20],
            "kernel_sizes": [15, 9, 5],
            "pool_sizes": [2, 2],
            "adaptive_pool_length": 8,
            "classifier_width": 32,
            "dropout": 0.1,
        },
        "orbit_shifts": [0, 1, 2, 3] if model_kind in {"orbit", "orbit_prob"} else [],
        "orbit_aggregation": (
            "mean_logits" if model_kind == "orbit"
            else "mean_probabilities" if model_kind == "orbit_prob"
            else None
        ),
    }
    if model_kind == "harmonic":
        architecture["input_transform"] = "harmonic_c4_invariants"
    return architecture


def create_checkpoint_payload(
    model: nn.Module,
    model_kind: str,
    experiment_name: str,
    seed: int,
    label_shuffle_seed: int | None,
    best_epoch: int,
    best_val_macro_f1: float,
    best_val_loss: float,
    max_epochs: int,
    patience: int,
    sensor_scales: np.ndarray,
    channel_map_confirmation: dict[str, list[str]],
    split_distance_mm: dict[str, list[int]],
    dataset_path: Path,
    dataset_sha256_value: str,
    selected_for_final_test: bool,
    selected_seed: int,
) -> dict[str, object]:
    return {
        "format": "c170-cnn-checkpoint",
        "format_version": 1,
        "model_kind": model_kind,
        "model_name": model_display_name(model_kind),
        "architecture": model_architecture(model_kind),
        "state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "class_order": list(DIRECTIONS),
        "training_hyperparameters": {
            "seed": int(seed),
            "numpy_seed": int(seed),
            "torch_seed": int(seed),
            "label_shuffle_seed": label_shuffle_seed,
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "weight_decay": 0.001,
            "label_smoothing": 0.02,
        },
        "preprocessing": {
            "sample_rate_hz": 250,
            "filter_band_hz": [0.5, 40.0],
            "filter_type": "butterworth_bandpass_sosfiltfilt_order_3",
            "dc_removal": "subtract_per_trial_per_channel_temporal_mean",
            "sensor_scales_source": "training_split_median_absolute_centered_signal",
            "training_derived_sensor_scales": [float(value) for value in sensor_scales],
            "model_input_transform": (
                "harmonic_c4_invariants" if model_kind == "harmonic" else "identity"
            ),
        },
        "experiment_name": experiment_name,
        "selection_basis": {
            "criterion": "highest_validation_macro_f1_then_lowest_validation_loss_then_lowest_seed",
            "validation_macro_f1": float(best_val_macro_f1),
            "validation_loss": float(best_val_loss),
            "best_epoch_zero_based": int(best_epoch),
            "selected_for_final_test": bool(selected_for_final_test),
            "selected_seed_for_condition_and_model": int(selected_seed),
        },
        "channel_map_confirmation": channel_map_confirmation,
        "physical_order_confirmation_source": "User confirmed: B057 and earlier use columns 0,2,4,6; B058 onward use 0,1,2,3; both orders correspond to S0,S1,S2,S3",
        "split_distance_mm": split_distance_mm,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256_value,
    }


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as checkpoint_file:
        torch.save(payload, checkpoint_file)


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "c170-cnn-checkpoint" or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if payload.get("class_order") != list(DIRECTIONS):
        raise ValueError(f"Checkpoint class order mismatch: {path}")
    model_kind = str(payload["model_kind"])
    if payload.get("architecture") != model_architecture(model_kind):
        raise ValueError(f"Checkpoint architecture record mismatch: {path}")
    model = build_model(model_kind)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def prepare_checkpoint_inputs(
    windows: np.ndarray,
    preprocessing: dict[str, object],
) -> np.ndarray:
    if int(preprocessing["sample_rate_hz"]) != 250:
        raise ValueError("Checkpoint sample_rate_hz does not match the C170 data")
    if preprocessing["filter_band_hz"] != [0.5, 40.0]:
        raise ValueError("Checkpoint filter_band_hz does not match the current preprocessing")
    filtered = filter_windows(windows, sample_rate=int(preprocessing["sample_rate_hz"]))
    normalized, _ = calibrate_windows(
        filtered,
        scale=np.asarray(preprocessing["training_derived_sensor_scales"], dtype=np.float32),
    )
    model_input_transform = str(preprocessing.get("model_input_transform", "identity"))
    if model_input_transform == "harmonic_c4_invariants":
        return harmonic_invariants(normalized)
    if model_input_transform != "identity":
        raise ValueError(f"Unsupported checkpoint model_input_transform: {model_input_transform}")
    return normalized.astype(np.float32)


def run_validation_phase(
    dataset_path: Path,
    output_dir: Path,
    assignments: dict[str, Sequence[int]],
    confirm_physical_order: bool,
    seeds: Sequence[int],
    max_epochs: int,
    patience: int,
    model_kinds: Sequence[str] | None = None,
) -> dict[str, object]:
    dataset_hash = dataset_sha256(dataset_path)
    dataset = load_dataset(dataset_path)
    if dataset_sha256(dataset_path) != dataset_hash:
        raise RuntimeError("dataset.npz changed while being loaded")
    splits = split_by_distance(dataset["distance_mm"], assignments)
    assert_recording_groups_intact(dataset)
    assert_no_distance_crossing(dataset["distance_mm"], splits)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    selected_model_kinds = tuple(DEFAULT_MODEL_KINDS if model_kinds is None else model_kinds)
    if not selected_model_kinds or len(set(selected_model_kinds)) != len(selected_model_kinds):
        raise ValueError("model_kinds must be nonempty and unique")
    unknown_model_kinds = sorted(set(selected_model_kinds) - set(MODEL_KINDS))
    if unknown_model_kinds:
        raise ValueError(f"Unknown model_kinds: {unknown_model_kinds}")
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive integers")
    labels = dataset["labels"]
    for split_name in ("train", "val"):
        if not np.array_equal(np.unique(labels[splits[split_name]]), np.arange(len(DIRECTIONS))):
            raise ValueError(f"{split_name} split does not contain all four direction classes")

    selected_indices = np.concatenate([splits[name] for name in SPLIT_NAMES])
    channel_map_confirmation = validate_channel_maps(
        dataset["channel_map"][selected_indices],
        confirm_physical_order=confirm_physical_order,
    )
    normalized, sensor_scales = prepare_model_inputs(dataset["X_raw"], splits)
    output_dir = create_output_directory(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    trained_runs: list[dict[str, object]] = []
    for experiment_name in ("true_training_labels", "shuffled_training_labels"):
        for model_kind in selected_model_kinds:
            for seed_value in seeds:
                seed = int(seed_value)
                label_shuffle_seed = None
                train_labels = labels[splits["train"]].copy()
                if experiment_name == "shuffled_training_labels":
                    label_shuffle_seed = seed + 10_000
                    shuffled_all_labels = shuffle_training_labels_by_recording(
                        labels,
                        dataset["distance_mm"],
                        dataset["block"],
                        splits["train"],
                        seed=label_shuffle_seed,
                    )
                    train_labels = shuffled_all_labels[splits["train"]]

                trained = train_temporal_model(
                    prepare_candidate_inputs(normalized["train"], model_kind),
                    train_labels,
                    prepare_candidate_inputs(normalized["val"], model_kind),
                    labels[splits["val"]],
                    model_kind=model_kind,
                    seed=seed,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                run = {
                    "experiment": experiment_name,
                    "model_kind": model_kind,
                    "seed": seed,
                    "label_shuffle_seed": label_shuffle_seed,
                    "best_epoch": int(trained.best_epoch),
                    "best_val_macro_f1": float(trained.best_val_macro_f1),
                    "best_val_loss": float(trained.best_val_loss),
                    "model": trained.model,
                }
                trained_runs.append(run)
                for split_name, indices, predictions in (
                    ("train", splits["train"], trained.train_predictions),
                    ("val", splits["val"], trained.val_predictions),
                ):
                    metric_rows.append(
                        _metric_row(
                            dataset,
                            indices,
                            predictions,
                            experiment_name,
                            model_kind,
                            seed,
                            split_name,
                            int(trained.best_epoch),
                            label_shuffle_seed,
                        )
                    )
                    prediction_rows.extend(
                        _prediction_rows(
                            dataset,
                            indices,
                            predictions,
                            experiment_name,
                            model_kind,
                            seed,
                            split_name,
                        )
                    )

    selected_runs: dict[tuple[str, str], dict[str, object]] = {}
    for experiment_name in ("true_training_labels", "shuffled_training_labels"):
        for model_kind in selected_model_kinds:
            candidates = [
                run
                for run in trained_runs
                if run["experiment"] == experiment_name and run["model_kind"] == model_kind
            ]
            selected_runs[(experiment_name, model_kind)] = min(
                candidates,
                key=lambda run: (
                    -float(run["best_val_macro_f1"]),
                    float(run["best_val_loss"]),
                    int(run["seed"]),
                ),
            )

    split_distance_mm = {
        split_name: sorted(np.unique(dataset["distance_mm"][indices]).astype(int).tolist())
        for split_name, indices in splits.items()
    }
    selected_checkpoints: dict[str, dict[str, object]] = {}
    checkpoint_records: list[dict[str, object]] = []
    for run in trained_runs:
        experiment_name = str(run["experiment"])
        model_kind = str(run["model_kind"])
        selected_run = selected_runs[(experiment_name, model_kind)]
        selected_seed = int(selected_run["seed"])
        is_selected = int(run["seed"]) == selected_seed
        checkpoint_name = (
            f"{experiment_name}__{model_kind}__seed-{int(run['seed'])}.pt"
        )
        checkpoint_path = checkpoint_dir / checkpoint_name
        payload = create_checkpoint_payload(
            model=run["model"],
            model_kind=model_kind,
            experiment_name=experiment_name,
            seed=int(run["seed"]),
            label_shuffle_seed=run["label_shuffle_seed"],
            best_epoch=int(run["best_epoch"]),
            best_val_macro_f1=float(run["best_val_macro_f1"]),
            best_val_loss=float(run["best_val_loss"]),
            max_epochs=max_epochs,
            patience=patience,
            sensor_scales=sensor_scales,
            channel_map_confirmation=channel_map_confirmation,
            split_distance_mm=split_distance_mm,
            dataset_path=dataset_path,
            dataset_sha256_value=dataset_hash,
            selected_for_final_test=is_selected,
            selected_seed=selected_seed,
        )
        save_checkpoint(checkpoint_path, payload)
        checkpoint_record = {
            "experiment": experiment_name,
            "model_kind": model_kind,
            "seed": int(run["seed"]),
            "path": str(checkpoint_path.relative_to(output_dir)),
            "selected_for_final_test": is_selected,
            "validation_macro_f1": float(run["best_val_macro_f1"]),
            "validation_loss": float(run["best_val_loss"]),
            "best_epoch_zero_based": int(run["best_epoch"]),
        }
        checkpoint_records.append(checkpoint_record)
        if is_selected:
            selected_checkpoints[f"{experiment_name}:{model_kind}"] = checkpoint_record

    summaries = summarize_metrics(metric_rows)
    primary_candidates = [
        row
        for row in summaries
        if row["experiment"] == "true_training_labels" and row["split"] == "val"
    ]
    selected_primary_model = max(
        primary_candidates,
        key=lambda row: float(row["overall"]["macro_f1_mean"]),
    )
    unassigned_distances = sorted(
        set(np.unique(dataset["distance_mm"]).astype(int).tolist())
        - set(np.concatenate([dataset["distance_mm"][indices] for indices in splits.values()]).astype(int).tolist())
    )
    config = {
        "phase": "validation-only",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_hash,
        "sample_count_after_duplicate_exclusion": int(len(dataset["X_raw"])),
        "duplicate_rows_excluded": int(dataset["duplicate_rows_excluded"]),
        "channel_map_confirmation": channel_map_confirmation,
        "physical_order_confirmation_source": "User confirmed: B057 and earlier use columns 0,2,4,6; B058 onward use 0,1,2,3; both orders correspond to S0,S1,S2,S3",
        "confirm_physical_order_flag": bool(confirm_physical_order),
        "split_distance_mm": split_distance_mm,
        "unassigned_distance_mm": unassigned_distances,
        "split_sample_counts": {
            split_name: int(len(indices)) for split_name, indices in splits.items()
        },
        "direction_order": list(DIRECTIONS),
        "model_kinds": list(selected_model_kinds),
        "training_label_conditions": ["true_training_labels", "shuffled_training_labels"],
        "label_shuffle_unit": "Permute direction labels among the four complete recording blocks within each training position, preserving class-block counts",
        "seeds": [int(seed) for seed in seeds],
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "sample_rate_hz": 250,
        "filter_band_hz": [0.5, 40.0],
        "training_derived_sensor_scales": [float(scale) for scale in sensor_scales],
        "candidate_selection": {
            "criterion": "Highest validation macro F1, then lowest validation loss, then lowest seed",
            "selected_checkpoints": selected_checkpoints,
            "all_checkpoints": checkpoint_records,
        },
        "selected_primary_model_by_validation_macro_f1": selected_primary_model["model_kind"],
    }
    metrics = {"runs": metric_rows, "summaries": summaries}
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "validation_metrics.json", metrics)
    _write_predictions(output_dir / "validation_predictions.csv", prediction_rows)
    return {"config": config, "metrics": metrics, "prediction_count": len(prediction_rows)}


def run_final_test_phase(
    output_dir: Path,
    dataset_path: Path | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    started_path = output_dir / "final_test_started.json"
    if started_path.exists():
        raise FileExistsError(f"final-test has already started for this validation-only output: {output_dir}")
    final_artifact_paths = (
        output_dir / "final_test_metrics.json",
        output_dir / "final_test_predictions.csv",
        output_dir / "final_test_completed.json",
    )
    existing_final_artifacts = [str(path) for path in final_artifact_paths if path.exists()]
    if existing_final_artifacts:
        raise FileExistsError(f"final-test artifacts already exist: {existing_final_artifacts}")
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("phase") != "validation-only":
        raise ValueError("final-test requires a configuration produced by validation-only")
    configured_dataset = Path(config["dataset_path"]).resolve()
    selected_checkpoints = config["candidate_selection"]["selected_checkpoints"]
    model_kinds = tuple(config.get("model_kinds", DEFAULT_MODEL_KINDS))
    if not model_kinds or len(set(model_kinds)) != len(model_kinds):
        raise ValueError("Invalid model_kinds in the validation-only configuration")
    if set(model_kinds) - set(MODEL_KINDS):
        raise ValueError("The validation-only configuration contains an unsupported model_kind")
    expected_keys = {
        f"{condition}:{model_kind}"
        for condition in ("true_training_labels", "shuffled_training_labels")
        for model_kind in model_kinds
    }
    if set(selected_checkpoints) != expected_keys:
        raise ValueError("The validation-only configuration lacks a selected checkpoint for each label condition and model")
    if dataset_path is not None and Path(dataset_path).resolve() != configured_dataset:
        raise ValueError("final-test dataset path differs from the validation-only configuration")
    expected_dataset_sha256 = str(config["dataset_sha256"])
    current_dataset_sha256 = dataset_sha256(configured_dataset)
    if current_dataset_sha256 != expected_dataset_sha256:
        raise ValueError("dataset.npz SHA-256 differs from the validation-only configuration")
    marker = {
        "phase": "final-test-started",
        "dataset_path": str(configured_dataset),
        "dataset_sha256": expected_dataset_sha256,
        "test_distance_mm": config["split_distance_mm"]["test"],
    }
    _write_json_exclusive(started_path, marker)

    loaded_models: list[tuple[dict[str, object], nn.Module, dict[str, object]]] = []
    for key in sorted(expected_keys):
        checkpoint_record = selected_checkpoints[key]
        if not checkpoint_record["selected_for_final_test"]:
            raise ValueError(f"Checkpoint was not selected by validation: {key}")
        checkpoint_path = output_dir / str(checkpoint_record["path"])
        model, payload = load_checkpoint(checkpoint_path)
        if payload["experiment_name"] != checkpoint_record["experiment"]:
            raise ValueError(f"Checkpoint label condition differs from the validation configuration: {checkpoint_path}")
        if payload["model_kind"] != checkpoint_record["model_kind"]:
            raise ValueError(f"Checkpoint model kind differs from the validation configuration: {checkpoint_path}")
        if payload["training_hyperparameters"]["seed"] != checkpoint_record["seed"]:
            raise ValueError(f"Checkpoint seed differs from the validation configuration: {checkpoint_path}")
        if payload["dataset_path"] != str(configured_dataset):
            raise ValueError(f"Checkpoint dataset path differs from the validation configuration: {checkpoint_path}")
        if payload["dataset_sha256"] != expected_dataset_sha256:
            raise ValueError(f"Checkpoint dataset SHA-256 differs from the validation configuration: {checkpoint_path}")
        if payload["split_distance_mm"] != config["split_distance_mm"]:
            raise ValueError(f"Checkpoint position splits differ from the validation configuration: {checkpoint_path}")
        if payload["channel_map_confirmation"] != config["channel_map_confirmation"]:
            raise ValueError(f"Checkpoint channel_map confirmation differs from the validation configuration: {checkpoint_path}")
        if payload["selection_basis"]["selected_seed_for_condition_and_model"] != checkpoint_record["seed"]:
            raise ValueError(f"Checkpoint selection record differs from the validation configuration: {checkpoint_path}")
        if not payload["selection_basis"]["selected_for_final_test"]:
            raise ValueError(f"Checkpoint lacks a validation-selection marker: {checkpoint_path}")
        loaded_models.append((checkpoint_record, model, payload))

    preprocessing = loaded_models[0][2]["preprocessing"]
    for _, _, payload in loaded_models[1:]:
        common = {
            key: value for key, value in payload["preprocessing"].items()
            if key != "model_input_transform"
        }
        reference_common = {
            key: value for key, value in preprocessing.items()
            if key != "model_input_transform"
        }
        if common != reference_common:
            raise ValueError("Selected checkpoints have inconsistent preprocessing parameters")
    dataset = load_dataset(configured_dataset)
    if dataset_sha256(configured_dataset) != expected_dataset_sha256:
        raise ValueError("dataset.npz changed during final-test loading")
    splits = split_by_distance(dataset["distance_mm"], config["split_distance_mm"])
    assert_recording_groups_intact(dataset)
    assert_no_distance_crossing(dataset["distance_mm"], splits)
    test_indices = splits["test"]
    if not np.array_equal(np.unique(dataset["labels"][test_indices]), np.arange(len(DIRECTIONS))):
        raise ValueError("Final-test positions do not contain all four direction classes")
    test_channel_map_confirmation = validate_channel_maps(
        dataset["channel_map"][test_indices],
        confirm_physical_order=bool(config["confirm_physical_order_flag"]),
    )
    if not np.allclose(
        np.asarray(preprocessing["training_derived_sensor_scales"], dtype=np.float32),
        np.asarray(config["training_derived_sensor_scales"], dtype=np.float32),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Checkpoint training scale differs from the validation configuration")
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for checkpoint_record, model, payload in loaded_models:
        test_inputs = prepare_checkpoint_inputs(
            dataset["X_raw"][test_indices], payload["preprocessing"]
        )
        predictions = predict_temporal_model(model, test_inputs)
        metric_rows.append(
            _metric_row(
                dataset,
                test_indices,
                predictions,
                str(checkpoint_record["experiment"]),
                str(checkpoint_record["model_kind"]),
                int(checkpoint_record["seed"]),
                "test",
                int(payload["selection_basis"]["best_epoch_zero_based"]),
                payload["training_hyperparameters"]["label_shuffle_seed"],
            )
        )
        prediction_rows.extend(
            _prediction_rows(
                dataset,
                test_indices,
                predictions,
                str(checkpoint_record["experiment"]),
                str(checkpoint_record["model_kind"]),
                int(checkpoint_record["seed"]),
                "test",
            )
        )

    metrics = {
        "phase": "final-test",
        "runs": metric_rows,
        "summaries": summarize_metrics(metric_rows),
        "test_channel_map_confirmation": test_channel_map_confirmation,
    }
    _write_json_exclusive(output_dir / "final_test_metrics.json", metrics)
    _write_predictions(output_dir / "final_test_predictions.csv", prediction_rows)
    _write_json_exclusive(
        output_dir / "final_test_completed.json",
        {
            "phase": "final-test-completed",
            "run_count": len(metric_rows),
            "prediction_count": len(prediction_rows),
            "test_distance_mm": config["split_distance_mm"]["test"],
        },
    )
    return {"metrics": metrics, "prediction_count": len(prediction_rows)}


def main() -> None:
    args = parse_args()
    validate_phase_args(args)
    if args.phase == "validation-only":
        assignments = {
            "train": args.train_distance_mm,
            "val": args.val_distance_mm,
            "test": args.test_distance_mm,
        }
        result = run_validation_phase(
            args.dataset,
            args.output_dir,
            assignments,
            args.confirm_physical_order,
            args.seeds,
            args.max_epochs,
            args.patience,
            args.model_kinds,
        )
    else:
        result = run_final_test_phase(args.output_dir, args.dataset)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "output_dir": str(args.output_dir),
                "prediction_count": result["prediction_count"],
                "selected_primary_model_by_validation_macro_f1": result.get("config", {}).get(
                    "selected_primary_model_by_validation_macro_f1"
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
