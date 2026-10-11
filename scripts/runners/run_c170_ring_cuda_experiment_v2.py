"""Run one validation-only CUDA C170 cross-position ring experiment (v2)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from scripts.runners import run_c170_experiment as c170
from src.c170_ring_representation import (
    RingRepresentationClassifier,
    cross_position_supervised_contrastive_loss,
)


TRAIN_DISTANCE_MM = (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)
VAL_DISTANCE_MM = (5, 40, 90, 130)
REGISTERED_TEST_DISTANCE_MM = (15, 25, 110, 150)
EXCLUDED_DISTANCE_MM = (70,)
SEEDS = (17, 29, 43, 61, 89)
ANGLE_PERIOD_MM = 170.0
N_ACTIONS = 4
UPDATES_PER_EPOCH = 20
SUPCON_TEMPERATURE = 0.1
SUPCON_WEIGHT = 0.1
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
LABEL_SHUFFLE_SEED_OFFSET = 10_000
DEFAULT_DATASET = PROJECT_DIR / "data" / "processed_c170_20260926" / "dataset.npz"

VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "ring_main": {
        "model_variant": "main",
        "use_supcon": True,
        "shuffle_labels": False,
    },
    "ring_ce": {
        "model_variant": "classification_only",
        "use_supcon": False,
        "shuffle_labels": False,
    },
    "slot_supcon": {
        "model_variant": "slot_readout",
        "use_supcon": True,
        "shuffle_labels": False,
    },
    "ring_shuffle": {
        "model_variant": "main",
        "use_supcon": True,
        "shuffle_labels": True,
    },
}

CURVE_FIELDS = (
    "epoch",
    "optimizer_updates",
    "epoch_seconds",
    "elapsed_seconds",
    "train_weighted_ce_online",
    "train_supcon_batch_mean_online",
    "train_objective_online",
    "val_cross_entropy",
    "val_macro_f1",
    "val_recall_up",
    "val_recall_down",
    "val_recall_left",
    "val_recall_right",
    "val_unique_predicted_classes",
    "val_f1_by_position_json",
    "val_position_channel_map_json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one fixed C170 ring v2 seed using train/validation only."
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _position_weights(train_positions: Sequence[int]) -> dict[int, float]:
    positions = sorted({int(value) for value in train_positions})
    if len(positions) < 2:
        raise ValueError("at least two train positions are required for circular weighting")
    lengths: dict[int, float] = {}
    for offset, position in enumerate(positions):
        previous = positions[offset - 1]
        following = positions[(offset + 1) % len(positions)]
        lengths[position] = 0.5 * (
            (position - previous) % ANGLE_PERIOD_MM
            + (following - position) % ANGLE_PERIOD_MM
        )
    total = sum(lengths.values())
    if not math.isclose(total, ANGLE_PERIOD_MM, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError(f"circular position weights sum to {total}, expected {ANGLE_PERIOD_MM}")
    return {position: value / total for position, value in lengths.items()}


def _training_groups(
    train_indices: np.ndarray,
    original_labels: np.ndarray,
    distances: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for index in np.asarray(train_indices, dtype=np.int64).tolist():
        key = (int(distances[index]), int(original_labels[index]))
        grouped.setdefault(key, []).append(int(index))
    if {position for position, _label in grouped} != set(TRAIN_DISTANCE_MM):
        raise ValueError("sampler groups do not cover the fixed train positions")
    for position in TRAIN_DISTANCE_MM:
        if {label for current, label in grouped if current == position} != set(range(N_ACTIONS)):
            raise ValueError(f"position {position} mm must contain all four original labels")
    sizes = sorted(len(values) for values in grouped.values())
    if len(grouped) != 44 or sizes.count(19) != 1 or sizes.count(20) != 43:
        raise ValueError(f"expected 44 original-label groups sized 19/20, got {sizes}")
    return {key: np.asarray(values, dtype=np.int64) for key, values in sorted(grouped.items())}


def _training_sample_weights(
    groups: dict[tuple[int, int], np.ndarray],
) -> tuple[dict[int, float], dict[int, float]]:
    position_weights = _position_weights(TRAIN_DISTANCE_MM)
    sample_weights: dict[int, float] = {}
    for (position, _original_label), indices in groups.items():
        weight = position_weights[position] / (N_ACTIONS * len(indices))
        for index in indices.tolist():
            sample_weights[int(index)] = weight
    if not math.isclose(sum(sample_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("training sample weights must sum to one")
    return position_weights, sample_weights


def _batches_for_epoch(
    groups: dict[tuple[int, int], np.ndarray],
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    """Sample original physical groups without replacement, independent of model RNG."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    keys = sorted(groups)
    order = rng.permutation(len(keys))
    shuffled = {
        keys[int(key_index)]: rng.permutation(groups[keys[int(key_index)]])
        for key_index in order
    }
    batches: list[np.ndarray] = []
    for step in range(max(len(group) for group in shuffled.values())):
        batch = np.asarray(
            [group[step] for group in shuffled.values() if step < len(group)],
            dtype=np.int64,
        )
        rng.shuffle(batch)
        batches.append(batch)
    flattened = np.concatenate(batches)
    expected = np.concatenate(list(groups.values()))
    if [len(batch) for batch in batches] != [44] * 19 + [43]:
        raise RuntimeError("sampler must produce 19 batches of 44 and a final batch of 43")
    if len(flattened) != 879 or len(np.unique(flattened)) != 879:
        raise RuntimeError("sampler must use all 879 train windows exactly once")
    if set(flattened.tolist()) != set(expected.tolist()):
        raise RuntimeError("sampler indices differ from the training split")
    return batches


def _make_training_targets(
    original_labels: np.ndarray,
    distances: np.ndarray,
    blocks: np.ndarray,
    train_indices: np.ndarray,
    variant: str,
    seed: int,
) -> np.ndarray:
    targets = np.asarray(original_labels, dtype=np.int64).copy()
    if variant == "ring_shuffle":
        targets = c170.shuffle_training_labels_by_recording(
            targets,
            np.asarray(distances, dtype=np.int64),
            np.asarray(blocks, dtype=np.int64),
            np.asarray(train_indices, dtype=np.int64),
            seed=int(seed) + LABEL_SHUFFLE_SEED_OFFSET,
        )
    elif variant not in VARIANT_CONFIG:
        raise ValueError(f"unsupported v2 variant: {variant}")

    training = np.asarray(train_indices, dtype=np.int64)
    if variant == "ring_shuffle":
        for block in np.unique(blocks[training]):
            selected = training[np.asarray(blocks[training]) == block]
            if len(np.unique(targets[selected])) != 1:
                raise RuntimeError(f"shuffled training block {int(block)} has multiple targets")
        for distance in TRAIN_DISTANCE_MM:
            position_targets = targets[
                training[np.asarray(distances)[training] == distance]
            ]
            if set(np.unique(position_targets).tolist()) != set(range(N_ACTIONS)):
                raise RuntimeError(f"block shuffle lost a direction target at {distance} mm")
    return targets


def _make_shuffle_mapping(
    dataset: dict[str, Any],
    train_indices: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block in sorted(np.unique(dataset["block"][train_indices]).tolist()):
        indices = train_indices[dataset["block"][train_indices] == block]
        original = np.unique(dataset["labels"][indices])
        assigned = np.unique(targets[indices])
        if len(original) != 1 or len(assigned) != 1:
            raise ValueError(f"training recording block {int(block)} must have a single original and target label")
        rows.append(
            {
                "distance_mm": int(dataset["distance_mm"][indices[0]]),
                "block": int(block),
                "original_direction": c170.DIRECTIONS[int(original[0])],
                "training_target_direction": c170.DIRECTIONS[int(assigned[0])],
            }
        )
    return rows


def batch_objective(
    per_sample_ce: torch.Tensor,
    sample_weights: torch.Tensor,
    supcon_loss: torch.Tensor | None,
    *,
    use_supcon: bool,
) -> torch.Tensor:
    if per_sample_ce.ndim != 1 or sample_weights.shape != per_sample_ce.shape:
        raise ValueError("CE values and sample weights must be aligned one-dimensional tensors")
    data_term = UPDATES_PER_EPOCH * torch.sum(per_sample_ce * sample_weights)
    if use_supcon:
        if supcon_loss is None or supcon_loss.ndim != 0:
            raise ValueError("SupCon variants require one scalar batch-mean contrastive loss")
        return data_term + SUPCON_WEIGHT * supcon_loss
    if supcon_loss is not None:
        raise ValueError("the CE-only variant must not compute a SupCon loss")
    return data_term


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not len(y_true):
        raise ValueError("classification metrics require aligned non-empty label vectors")
    if np.any((y_true < 0) | (y_true >= N_ACTIONS)) or np.any((y_pred < 0) | (y_pred >= N_ACTIONS)):
        raise ValueError("classification metrics require class IDs 0..3")
    confusion = np.zeros((N_ACTIONS, N_ACTIONS), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    true_count = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(true_positive, true_count, out=np.zeros(N_ACTIONS), where=true_count > 0)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(N_ACTIONS),
        where=predicted_count > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(N_ACTIONS),
        where=(precision + recall) > 0,
    )
    return {
        "count": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1.mean()),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion.tolist(),
        "prediction_counts": predicted_count.tolist(),
        "prediction_unique_class_count": int(np.count_nonzero(predicted_count)),
    }


def _stratified_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    distances: np.ndarray,
    channel_maps: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": _classification_metrics(labels, predictions)}
    by_position: dict[str, Any] = {}
    by_position_and_map: dict[str, Any] = {}
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    dist = np.asarray(distances, dtype=np.int64)
    maps = np.asarray(channel_maps, dtype=np.int64)
    if dist.shape != y.shape or maps.shape != (len(y), 4):
        raise ValueError("validation strata metadata shapes do not match the labels")
    for distance in sorted(np.unique(dist).tolist()):
        mask = dist == distance
        by_position[str(int(distance))] = _classification_metrics(y[mask], pred[mask])
        by_map: dict[str, Any] = {}
        for mapping in sorted({tuple(int(v) for v in row) for row in maps[mask].tolist()}):
            map_mask = mask & np.all(maps == np.asarray(mapping, dtype=np.int64), axis=1)
            by_map[",".join(str(value) for value in mapping)] = _classification_metrics(
                y[map_mask], pred[map_mask]
            )
        by_position_and_map[str(int(distance))] = by_map
    result["by_position_mm"] = by_position
    result["by_position_mm_and_channel_map"] = by_position_and_map
    return result


def _evaluate(
    model: RingRepresentationClassifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    distances: np.ndarray,
    channel_maps: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, float]:
    model.eval()
    with torch.no_grad():
        logits = model(inputs).float()
        per_sample_ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=0.0)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)
    probabilities_np = probabilities.detach().cpu().numpy().astype(np.float64)
    predictions_np = predictions.detach().cpu().numpy().astype(np.int64)
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    metrics = _stratified_metrics(labels_np, predictions_np, distances, channel_maps)
    return metrics, probabilities_np, predictions_np, float(per_sample_ce.mean().detach().cpu())


def _train_checkpoint_metrics(
    model: RingRepresentationClassifier,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    distances: np.ndarray,
    channel_maps: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    ce_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(inputs), 128):
            logits = model(inputs[start : start + 128]).float()
            chunk_labels = labels[start : start + 128]
            logits_parts.append(logits)
            ce_parts.append(
                F.cross_entropy(logits, chunk_labels, reduction="none", label_smoothing=0.0)
            )
    logits_all = torch.cat(logits_parts, dim=0)
    ce_all = torch.cat(ce_parts, dim=0)
    predictions = logits_all.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    target_np = labels.detach().cpu().numpy().astype(np.int64)
    metrics = _stratified_metrics(target_np, predictions, distances, channel_maps)
    return {
        "evaluation_target": "training targets (block-permuted targets for ring_shuffle)",
        "weighted_action_cross_entropy": float(torch.sum(weights * ce_all).detach().cpu()),
        "unweighted_action_cross_entropy": float(ce_all.mean().detach().cpu()),
        **metrics,
    }


def _prediction_rows(
    dataset: dict[str, Any],
    val_indices: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    variant: str,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, index in enumerate(val_indices.tolist()):
        row: dict[str, object] = {
            "variant": variant,
            "seed": int(seed),
            "sample_index": int(dataset["sample_index"][index]),
            "distance_mm": int(dataset["distance_mm"][index]),
            "block": int(dataset["block"][index]),
            "true_direction": c170.DIRECTIONS[int(dataset["labels"][index])],
            "prediction": c170.DIRECTIONS[int(predictions[offset])],
            "channel_map": ",".join(str(int(value)) for value in dataset["channel_map"][index]),
        }
        for class_id, name in enumerate(c170.DIRECTIONS):
            row[f"probability_{name.lower()}"] = float(probabilities[offset, class_id])
        rows.append(row)
    return rows


def _collect_code_hashes() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "ring_representation": PROJECT_DIR / "src" / "c170_ring_representation.py",
        "dataset_preprocessing_helpers": PROJECT_DIR / 'scripts/runners/run_c170_experiment.py',
        "preprocessing_transitive_signal_processing": PROJECT_DIR / "src" / "rotation_poc.py",
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing computation dependencies: {missing}")
    return {name: _file_sha256(path) for name, path in files.items()}


def _validate_and_prepare(
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

    normalized, sensor_scales = c170.prepare_model_inputs(
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
    original_groups = _training_groups(splits["train"], labels, distances)
    position_weights, sample_weights = _training_sample_weights(original_groups)
    weight_array = np.asarray(
        [sample_weights[int(index)] for index in splits["train"]], dtype=np.float32
    )
    if not math.isclose(float(weight_array.sum()), 1.0, rel_tol=0.0, abs_tol=2e-7):
        raise ValueError("training sample weights must sum to one")

    all_training_targets = _make_training_targets(
        labels,
        distances,
        blocks,
        splits["train"],
        args.variant,
        args.seed,
    )
    if not np.array_equal(all_training_targets[splits["val"]], labels[splits["val"]]):
        raise ValueError("training label shuffling modified validation targets")
    train_targets = torch.as_tensor(
        all_training_targets[splits["train"]], dtype=torch.long, device=device
    )
    val_targets = torch.as_tensor(labels[splits["val"]], dtype=torch.long, device=device)
    train_weights = torch.as_tensor(weight_array, dtype=torch.float32, device=device)

    metadata = {
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
                ",".join(str(int(v)) for v in mapping): int(
                    np.count_nonzero(
                        np.all(dataset["channel_map"][indices] == np.asarray(mapping), axis=1)
                    )
                )
                for mapping in sorted(
                    {
                        tuple(int(v) for v in row)
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
            args.seed + LABEL_SHUFFLE_SEED_OFFSET
            if VARIANT_CONFIG[args.variant]["shuffle_labels"]
            else None
        ),
        "label_shuffle_mapping": (
            _make_shuffle_mapping(dataset, splits["train"], all_training_targets)
            if VARIANT_CONFIG[args.variant]["shuffle_labels"]
            else []
        ),
    }
    preprocessing = {
        "sample_rate_hz": 250,
        "filter_band_hz": [0.5, 40.0],
        "filter_type": "butterworth_bandpass_sosfiltfilt_order_3",
        "dc_removal": "subtract_per_trial_per_channel_temporal_mean",
        "sensor_scale_source": "training_split_median_absolute_centered_signal",
        "sensor_scales": [float(value) for value in sensor_scales],
        "channel_order": ["S0", "S1", "S2", "S3"],
        "channel_map_as_model_input": False,
        "normalization_fit_splits": ["train"],
        "processed_splits": ["train", "validation"],
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
        "all_training_targets": all_training_targets,
    }
    return dataset, splits, metadata, {"preprocessing": preprocessing, "tensors": tensors}


def _manifest_base(
    config: dict[str, Any],
    config_sha256: str,
    started_utc: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "format": "c170-ring-cuda-v2-run",
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
    started_utc = _utc_now()
    dataset_path = args.dataset.resolve()
    dataset_sha256 = _file_sha256(dataset_path)
    config = {
        "variant": args.variant,
        "model_variant": VARIANT_CONFIG[args.variant]["model_variant"],
        "use_supcon": VARIANT_CONFIG[args.variant]["use_supcon"],
        "shuffle_labels": VARIANT_CONFIG[args.variant]["shuffle_labels"],
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
        "position_weight_period_mm": ANGLE_PERIOD_MM,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "cross_entropy_label_smoothing": 0.0,
        "batch_ce_objective": "20*sum(sample_weight_i*per_sample_ce_i)",
        "supcon_temperature": SUPCON_TEMPERATURE,
        "supcon_weight": SUPCON_WEIGHT if VARIANT_CONFIG[args.variant]["use_supcon"] else 0.0,
        "supcon_batch_reduction": "existing_helper_scalar_mean_over_anchors",
        "checkpoint_selection": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
        "physical_order_confirmed_by_operator": True,
    }
    config_sha256 = _canonical_json_sha256(config)
    output_dir = args.output_dir.resolve()
    c170.create_output_directory(output_dir)
    manifest = _manifest_base(config, config_sha256, started_utc, args, device)
    manifest_path = output_dir / "run_manifest.json"
    _atomic_write_json(manifest_path, manifest)

    try:
        torch.set_num_threads(args.cpu_threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats(device)

        preprocessing_started = time.monotonic()
        dataset, splits, split_metadata, prepared = _validate_and_prepare(args, device)
        tensors = prepared["tensors"]
        preprocessing_seconds = time.monotonic() - preprocessing_started
        manifest["split"] = split_metadata
        manifest["preprocessing"] = prepared["preprocessing"]
        manifest["timing"]["preprocessing_seconds"] = preprocessing_seconds
        _atomic_write_json(manifest_path, manifest)

        model = RingRepresentationClassifier(
            variant=VARIANT_CONFIG[args.variant]["model_variant"], n_classes=N_ACTIONS
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        train_indices = tensors["train_indices"]
        val_indices = tensors["validation_indices"]
        original_groups = tensors["original_groups"]
        global_to_local = {int(index): offset for offset, index in enumerate(train_indices)}
        train_distance_tensor = torch.as_tensor(
            tensors["train_distances"], dtype=torch.long, device=device
        )
        train_block_tensor = torch.as_tensor(tensors["train_blocks"], dtype=torch.long, device=device)
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

        for epoch in range(args.max_epochs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_started = time.monotonic()
            model.train()
            batches = _batches_for_epoch(original_groups, args.seed, epoch)
            weighted_ce_epoch = torch.zeros((), dtype=torch.float32, device=device)
            supcon_epoch = torch.zeros((), dtype=torch.float32, device=device)

            for batch_global in batches:
                local_np = np.asarray(
                    [global_to_local[int(index)] for index in batch_global], dtype=np.int64
                )
                local_indices = torch.as_tensor(local_np, dtype=torch.long, device=device)
                samples = tensors["train_inputs"][local_indices]
                targets = tensors["train_targets"][local_indices]
                weights = tensors["train_weights"][local_indices]
                logits, projection = model.forward_with_projection(samples)
                per_sample_ce = F.cross_entropy(
                    logits.float(), targets, reduction="none", label_smoothing=0.0
                )
                if VARIANT_CONFIG[args.variant]["use_supcon"]:
                    contrastive = cross_position_supervised_contrastive_loss(
                        projection,
                        targets,
                        train_distance_tensor[local_indices],
                        train_block_tensor[local_indices],
                        temperature=SUPCON_TEMPERATURE,
                    )
                else:
                    contrastive = None
                loss = batch_objective(
                    per_sample_ce,
                    weights,
                    contrastive,
                    use_supcon=bool(VARIANT_CONFIG[args.variant]["use_supcon"]),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                weighted_ce_epoch = weighted_ce_epoch + torch.sum(weights * per_sample_ce).detach()
                if contrastive is not None:
                    supcon_epoch = supcon_epoch + contrastive.detach()

            val_metrics, val_probabilities, val_predictions, val_cross_entropy = _evaluate(
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
                checkpoint_payload = {
                    "format": "c170-ring-cuda-v2-checkpoint",
                    "format_version": 1,
                    "variant": args.variant,
                    "model_variant": VARIANT_CONFIG[args.variant]["model_variant"],
                    "seed": int(args.seed),
                    "best_epoch_zero_based": int(epoch),
                    "state_dict": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    },
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
                _atomic_torch_save(best_checkpoint_path, checkpoint_payload)
            else:
                bad_epochs += 1

            weighted_ce_value = float(weighted_ce_epoch.detach().cpu())
            supcon_mean_value = (
                float((supcon_epoch / UPDATES_PER_EPOCH).detach().cpu())
                if VARIANT_CONFIG[args.variant]["use_supcon"]
                else None
            )
            objective_value = weighted_ce_value + (
                SUPCON_WEIGHT * supcon_mean_value if supcon_mean_value is not None else 0.0
            )
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
                    "train_supcon_batch_mean_online": supcon_mean_value,
                    "train_objective_online": objective_value,
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
            _atomic_write_csv(output_dir / "learning_curve.csv", CURVE_FIELDS, curve_rows)
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
                "best_checkpoint": best_checkpoint_path.name,
                "learning_curve": "learning_curve.csv",
            }
            _atomic_write_json(manifest_path, manifest)
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

        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        train_checkpoint_metrics = _train_checkpoint_metrics(
            model,
            tensors["train_inputs"],
            tensors["train_targets"],
            tensors["train_weights"],
            tensors["train_distances"],
            tensors["train_channel_maps"],
        )
        # Re-evaluate the selected checkpoint on validation so stored rows exactly match it.
        final_val_metrics, final_val_probabilities, final_val_predictions, final_val_ce = _evaluate(
            model,
            tensors["validation_inputs"],
            tensors["validation_targets"],
            tensors["validation_distances"],
            tensors["validation_channel_maps"],
        )
        if not np.allclose(final_val_probabilities, best_probabilities, rtol=0.0, atol=1e-6):
            raise RuntimeError("reloaded selected checkpoint changed validation probabilities")
        if not np.array_equal(final_val_predictions, best_predictions):
            raise RuntimeError("reloaded selected checkpoint changed validation predictions")
        if not math.isclose(final_val_ce, best_cross_entropy, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError("reloaded selected checkpoint changed validation cross-entropy")

        prediction_rows = _prediction_rows(
            dataset,
            val_indices,
            final_val_predictions,
            final_val_probabilities,
            args.variant,
            args.seed,
        )
        _atomic_write_csv(
            output_dir / "validation_predictions.csv",
            tuple(prediction_rows[0].keys()),
            prediction_rows,
        )
        checkpoint_sha256 = _file_sha256(best_checkpoint_path)
        curve_sha256 = _file_sha256(output_dir / "learning_curve.csv")
        predictions_sha256 = _file_sha256(output_dir / "validation_predictions.csv")
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        )
        manifest["status"] = "completed"
        manifest["timing"].update(
            {
                "training_seconds": time.monotonic() - training_started,
                "total_elapsed_seconds": time.monotonic() - started,
                "peak_cuda_memory_allocated_bytes": peak_memory,
                "completed_utc": _utc_now(),
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
            "best_checkpoint": {"path": best_checkpoint_path.name, "sha256": checkpoint_sha256},
            "learning_curve": {"path": "learning_curve.csv", "sha256": curve_sha256},
            "validation_predictions": {
                "path": "validation_predictions.csv",
                "sha256": predictions_sha256,
            },
        }
        manifest["test_metrics_computed"] = False
        _atomic_write_json(manifest_path, manifest)
        manifest_sha256 = _file_sha256(manifest_path)
        _atomic_write_json(
            output_dir / "RUN_COMPLETE.json",
            {
                "format": "c170-ring-cuda-v2-complete",
                "format_version": 1,
                "status": "completed",
                "config_sha256": config_sha256,
                "manifest_sha256": manifest_sha256,
                "artifact_sha256": {
                    "best_checkpoint": checkpoint_sha256,
                    "learning_curve": curve_sha256,
                    "validation_predictions": predictions_sha256,
                },
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
            _atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
