"""Create a fail-closed, validation-only summary of the fixed C170 ring v2 suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
FORMAL_VARIANTS = ("ring_main", "ring_ce", "slot_supcon", "ring_shuffle")
FORMAL_SEEDS = (17, 29, 43, 61, 89)
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
TRAIN_POSITIONS = (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)
VAL_POSITIONS = (5, 40, 90, 130)
TEST_POSITIONS = (15, 25, 110, 150)
EXCLUDED_POSITIONS = (70,)
SOURCE_KEYS = (
    "runner",
    "ring_representation",
    "dataset_preprocessing_helpers",
    "preprocessing_transitive_signal_processing",
)
SOURCE_PATHS = {
    "runner": Path('scripts/runners/run_c170_ring_cuda_experiment_v2.py'),
    "ring_representation": Path("src/c170_ring_representation.py"),
    "dataset_preprocessing_helpers": Path('scripts/runners/run_c170_experiment.py'),
    "preprocessing_transitive_signal_processing": Path("src/rotation_poc.py"),
}


class AuditError(RuntimeError):
    """Raised when a run bundle is missing, inconsistent, or not the frozen suite."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise AuditError(f"expected a JSON object: {path}")
    return result


def _canonical_config_sha(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"n": 0, "mean": None, "std_population": None}
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "std_population": statistics.pstdev(numeric),
    }


def _classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, Any]:
    if not labels or len(labels) != len(predictions):
        raise AuditError("classification labels and predictions must be non-empty and aligned")
    confusion = [[0 for _ in DIRECTIONS] for _ in DIRECTIONS]
    for truth, prediction in zip(labels, predictions):
        if truth not in range(4) or prediction not in range(4):
            raise AuditError("classification rows contain an unknown action ID")
        confusion[int(truth)][int(prediction)] += 1
    recalls: list[float] = []
    f1s: list[float] = []
    for action in range(4):
        tp = confusion[action][action]
        support = sum(confusion[action])
        predicted = sum(confusion[row][action] for row in range(4))
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    counts = [sum(confusion[row][action] for row in range(4)) for action in range(4)]
    return {
        "count": len(labels),
        "accuracy": sum(a == b for a, b in zip(labels, predictions)) / len(labels),
        "macro_f1": statistics.fmean(f1s),
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "confusion_matrix": confusion,
        "prediction_counts": counts,
        "prediction_unique_class_count": sum(value > 0 for value in counts),
    }


def _validate_exact_matrix(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(variant, seed) for variant in FORMAL_VARIANTS for seed in FORMAL_SEEDS}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    extra: list[tuple[str, int]] = []
    for run in runs:
        variant = str(run.get("variant"))
        try:
            seed = int(run["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"run has invalid seed metadata: {run}") from exc
        key = (variant, seed)
        if key not in expected:
            extra.append(key)
        elif key in found:
            duplicates.append(key)
        else:
            found[key] = run
    missing = sorted(expected - set(found))
    if missing or duplicates or extra:
        raise AuditError(
            f"formal matrix must contain exactly 20 unique variant/seed pairs; "
            f"missing={missing}, duplicates={duplicates}, unexpected={extra}"
        )
    return sorted(found.values(), key=lambda item: (FORMAL_VARIANTS.index(item["variant"]), item["seed"]))


def _candidate_markers(roots: Sequence[Path]) -> list[Path]:
    markers: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise AuditError(f"run root is missing or not a directory: {root}")
        direct = root / "RUN_COMPLETE.json"
        if direct.is_file():
            markers.add(direct)
        markers.update(path.resolve() for path in root.rglob("RUN_COMPLETE.json"))
    return sorted(markers)


def _read_verified_snapshot(roots: Sequence[Path]) -> tuple[Path, dict[str, str]]:
    for root in roots:
        snapshot_dir = root.expanduser().resolve() / "source_snapshot"
        path = snapshot_dir / "snapshot_manifest.json"
        if not path.is_file():
            continue
        snapshot = _read_json(path)
        source_files = snapshot.get("source_files")
        if not isinstance(source_files, dict):
            raise AuditError(f"source_files missing from {path}")
        hashes: dict[str, str] = {}
        for key in SOURCE_KEYS:
            entry = source_files.get(key)
            if not isinstance(entry, dict) or not entry.get("path") or not entry.get("sha256"):
                raise AuditError(f"source snapshot is missing {key}")
            saved = snapshot_dir / str(entry["path"])
            if not saved.is_file() or _sha256(saved) != entry["sha256"]:
                raise AuditError(f"snapshot source hash mismatch for {key}: {saved}")
            hashes[key] = str(entry["sha256"])
        return snapshot_dir, hashes
    raise AuditError("a verified source_snapshot/snapshot_manifest.json is required")


def _verify_run(marker_path: Path, source_hashes: dict[str, str]) -> dict[str, Any]:
    run_dir = marker_path.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise AuditError(f"completion marker has no manifest: {run_dir}")
    marker = _read_json(marker_path)
    manifest = _read_json(manifest_path)
    if marker.get("format") != "c170-ring-cuda-v2-complete" or marker.get("status") != "completed":
        raise AuditError(f"invalid completion marker: {marker_path}")
    if manifest.get("format") != "c170-ring-cuda-v2-run" or manifest.get("status") != "completed":
        raise AuditError(f"run manifest is not completed: {manifest_path}")
    if marker.get("manifest_sha256") != _sha256(manifest_path):
        raise AuditError(f"manifest hash mismatch: {manifest_path}")
    if manifest.get("test_metrics_computed") is not False or marker.get("test_metrics_computed") is not False:
        raise AuditError(f"run reports test metrics were computed: {run_dir}")
    test_access = manifest.get("test_access", {})
    if any(test_access.get(field) is not False for field in ("used_as_model_input", "preprocessed", "used_to_estimate_statistics", "predicted_or_scored")):
        raise AuditError(f"test boundary violation recorded in {run_dir}")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise AuditError(f"missing config in {run_dir}")
    config_hash = _canonical_config_sha(config)
    if config_hash != manifest.get("config_sha256") or config_hash != marker.get("config_sha256"):
        raise AuditError(f"canonical config hash mismatch: {run_dir}")
    variant = str(config.get("variant"))
    seed = int(config.get("seed", -1))
    if variant not in FORMAL_VARIANTS or seed not in FORMAL_SEEDS:
        raise AuditError(f"unexpected variant or seed: {variant}/{seed}")
    model_variant, use_supcon, shuffle = {
        "ring_main": ("main", True, False),
        "ring_ce": ("classification_only", False, False),
        "slot_supcon": ("slot_readout", True, False),
        "ring_shuffle": ("main", True, True),
    }[variant]
    expected_config = {
        "model_variant": model_variant,
        "use_supcon": use_supcon,
        "shuffle_labels": shuffle,
        "max_epochs": 160,
        "patience": 25,
        "device": "cuda:0",
        "train_distance_mm": list(TRAIN_POSITIONS),
        "validation_distance_mm": list(VAL_POSITIONS),
        "registered_test_distance_mm": list(TEST_POSITIONS),
        "excluded_distance_mm": list(EXCLUDED_POSITIONS),
        "expected_split_counts": {"train": 879, "validation": 320, "registered_test": 320},
        "batch_sizes_per_epoch": [44] * 19 + [43],
        "updates_per_epoch": 20,
        "sampler_group_key": "original_distance_mm_and_direction_label",
        "sampler_rng": "numpy_seedsequence_seed_epoch_independent_of_model_rng",
        "sample_weight_rule": "position_voronoi_weight/(4*original_group_size)",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "cross_entropy_label_smoothing": 0.0,
        "supcon_temperature": 0.1,
        "supcon_weight": 0.1 if use_supcon else 0.0,
        "checkpoint_selection": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise AuditError(f"frozen config mismatch for {key} in {run_dir}: {config.get(key)!r}")
    split = manifest.get("split", {})
    if split.get("split_counts") != {"train": 879, "val": 320, "test": 320}:
        raise AuditError(f"split metadata counts mismatch: {run_dir}")
    preprocessing = manifest.get("preprocessing", {})
    if preprocessing.get("processed_splits") != ["train", "validation"] or preprocessing.get("normalization_fit_splits") != ["train"]:
        raise AuditError(f"preprocessing split provenance mismatch: {run_dir}")
    if manifest.get("provenance", {}).get("code_sha256") != source_hashes:
        raise AuditError(f"source code hashes differ from source snapshot: {run_dir}")
    if marker.get("selected_epoch_one_based") != manifest.get("selection", {}).get("best_epoch_one_based"):
        raise AuditError(f"completion marker selected epoch mismatch: {run_dir}")

    artifacts = manifest.get("artifacts")
    expected_artifact_keys = {"best_checkpoint", "learning_curve", "validation_predictions"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifact_keys:
        raise AuditError(f"unexpected/missing artifacts in {run_dir}")
    marker_hashes = marker.get("artifact_sha256")
    if not isinstance(marker_hashes, dict) or set(marker_hashes) != expected_artifact_keys:
        raise AuditError(f"completion marker artifact set mismatch: {run_dir}")
    artifact_hashes: dict[str, str] = {}
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("sha256"):
            raise AuditError(f"invalid artifact record {name}: {run_dir}")
        path = run_dir / str(artifact["path"])
        digest = _sha256(path) if path.is_file() else None
        if digest != artifact["sha256"] or marker_hashes.get(name) != digest:
            raise AuditError(f"artifact hash mismatch: {path}")
        artifact_hashes[name] = digest
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": run_dir,
        "manifest": manifest,
        "marker": marker,
        "config": config,
        "artifact_sha256": artifact_hashes,
    }


def _load_predictions(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = run["run_dir"]
    path = run_dir / run["manifest"]["artifacts"]["validation_predictions"]["path"]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 320:
        raise AuditError(f"expected 320 validation predictions, got {len(rows)} in {path}")
    samples: set[int] = set()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    distances: list[int] = []
    maps: list[tuple[int, ...]] = []
    for row in rows:
        if row.get("variant") != run["variant"] or int(row.get("seed", -1)) != run["seed"]:
            raise AuditError(f"prediction row identity mismatch: {path}")
        sample = int(row["sample_index"])
        if sample in samples:
            raise AuditError(f"duplicate validation sample {sample}: {path}")
        samples.add(sample)
        distance = int(row["distance_mm"])
        if distance not in VAL_POSITIONS:
            raise AuditError(f"unexpected validation position {distance}: {path}")
        truth_name = row["true_direction"]
        prediction_name = row["prediction"]
        if truth_name not in DIRECTIONS or prediction_name not in DIRECTIONS:
            raise AuditError(f"unknown class label in {path}")
        labels.append(DIRECTIONS.index(truth_name))
        predictions.append(DIRECTIONS.index(prediction_name))
        probability = [float(row[f"probability_{name.lower()}"]) for name in DIRECTIONS]
        if any(not math.isfinite(value) or value < 0.0 for value in probability):
            raise AuditError(f"invalid validation probability in {path}")
        if not math.isclose(sum(probability), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise AuditError(f"validation probabilities do not sum to one in {path}")
        if DIRECTIONS.index(max(DIRECTIONS, key=lambda name: probability[DIRECTIONS.index(name)])) != predictions[-1]:
            raise AuditError(f"prediction does not match probability argmax in {path}")
        mapping = tuple(int(value) for value in row["channel_map"].split(","))
        if len(mapping) != 4:
            raise AuditError(f"invalid channel-map metadata in {path}")
        probabilities.append(probability)
        distances.append(distance)
        maps.append(mapping)
    for position in VAL_POSITIONS:
        position_indices = [index for index, value in enumerate(distances) if value == position]
        if len(position_indices) != 80:
            raise AuditError(f"expected 80 validation samples at {position} mm in {path}")
        position_labels = [labels[index] for index in position_indices]
        if any(position_labels.count(action) != 20 for action in range(4)):
            raise AuditError(f"expected 20 samples of each action at {position} mm in {path}")
        expected_map = (0, 2, 4, 6) if position in (5, 40) else (0, 1, 2, 3)
        if {maps[index] for index in position_indices} != {expected_map}:
            raise AuditError(f"channel-map metadata changed at {position} mm in {path}")

    overall = _classification_metrics(labels, predictions)
    by_position: dict[str, Any] = {}
    by_position_and_map: dict[str, Any] = {}
    for position in VAL_POSITIONS:
        offsets = [index for index, value in enumerate(distances) if value == position]
        by_position[str(position)] = _classification_metrics(
            [labels[index] for index in offsets], [predictions[index] for index in offsets]
        )
        map_metrics: dict[str, Any] = {}
        map_keys = sorted({maps[index] for index in offsets})
        for mapping in map_keys:
            selected = [index for index in offsets if maps[index] == mapping]
            map_metrics[",".join(str(value) for value in mapping)] = _classification_metrics(
                [labels[index] for index in selected], [predictions[index] for index in selected]
            )
        by_position_and_map[str(position)] = map_metrics
    cross_entropy = statistics.fmean(
        -math.log(max(probabilities[index][labels[index]], 1e-300))
        for index in range(len(labels))
    )
    measured = {
        "overall": overall,
        "by_position_mm": by_position,
        "by_position_mm_and_channel_map": by_position_and_map,
    }
    selection = run["manifest"]["selection"]["validation_metrics"]
    if not math.isclose(overall["macro_f1"], selection["overall"]["macro_f1"], rel_tol=0.0, abs_tol=1e-10):
        raise AuditError(f"manifest validation F1 disagrees with predictions: {path}")
    if overall["confusion_matrix"] != selection["overall"]["confusion_matrix"]:
        raise AuditError(f"manifest confusion matrix disagrees with predictions: {path}")
    for actual, recorded in zip(overall["per_class_recall"], selection["overall"]["per_class_recall"]):
        if not math.isclose(actual, recorded, rel_tol=0.0, abs_tol=1e-10):
            raise AuditError(f"manifest per-class recall disagrees with predictions: {path}")
    if not math.isclose(cross_entropy, run["manifest"]["selection"]["validation_cross_entropy"], rel_tol=0.0, abs_tol=1e-5):
        raise AuditError(f"manifest validation CE disagrees with stored probabilities: {path}")
    for position, item in by_position.items():
        recorded = selection["by_position_mm"][position]
        if item["confusion_matrix"] != recorded["confusion_matrix"]:
            raise AuditError(f"manifest confusion matrix disagrees at {position} mm: {path}")
        if not math.isclose(item["macro_f1"], recorded["macro_f1"], rel_tol=0.0, abs_tol=1e-10):
            raise AuditError(f"manifest position F1 disagrees at {position} mm: {path}")
    return {"metrics": measured, "cross_entropy": cross_entropy}


def _load_curve(run: dict[str, Any]) -> list[dict[str, Any]]:
    path = run["run_dir"] / run["manifest"]["artifacts"]["learning_curve"]["path"]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise AuditError(f"empty learning curve: {path}")
    rows: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(raw_rows, start=1):
        try:
            parsed = {
                "epoch": int(row["epoch"]),
                "optimizer_updates": int(row["optimizer_updates"]),
                "epoch_seconds": float(row["epoch_seconds"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "train_weighted_ce_online": float(row["train_weighted_ce_online"]),
                "train_supcon_batch_mean_online": (
                    None if row["train_supcon_batch_mean_online"] == "" else float(row["train_supcon_batch_mean_online"])
                ),
                "val_cross_entropy": float(row["val_cross_entropy"]),
                "val_macro_f1": float(row["val_macro_f1"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"invalid learning-curve row in {path}: {exc}") from exc
        if parsed["epoch"] != expected_epoch or parsed["optimizer_updates"] != 20:
            raise AuditError(f"noncontiguous epoch/update count in {path}")
        if any(not math.isfinite(float(value)) for key, value in parsed.items() if key != "train_supcon_batch_mean_online" and value is not None):
            raise AuditError(f"nonfinite learning-curve value in {path}")
        if parsed["val_macro_f1"] < 0.0 or parsed["val_macro_f1"] > 1.0:
            raise AuditError(f"invalid validation F1 in {path}")
        rows.append(parsed)
    selection = run["manifest"]["selection"]
    if len(rows) != selection.get("actual_epochs") or len(rows) > 160:
        raise AuditError(f"learning-curve length disagrees with manifest: {path}")
    if len(rows) * 20 != selection.get("optimizer_updates"):
        raise AuditError(f"learning-curve update total disagrees with manifest: {path}")
    selected = int(selection["best_epoch_one_based"])
    best_row = max(rows, key=lambda row: (row["val_macro_f1"], -row["val_cross_entropy"], -row["epoch"]))
    if selected != best_row["epoch"]:
        raise AuditError(f"selected epoch does not match validation curve: {path}")
    if not math.isclose(best_row["val_macro_f1"], selection["validation_macro_f1"], rel_tol=0.0, abs_tol=1e-12):
        raise AuditError(f"selected macro-F1 disagrees with curve: {path}")
    return rows


def _analyze_run(run: dict[str, Any]) -> dict[str, Any]:
    predictions = _load_predictions(run)
    curve = _load_curve(run)
    manifest = run["manifest"]
    train_eval = manifest.get("selected_checkpoint_train_evaluation", {})
    train_overall = train_eval.get("overall", {})
    if train_overall.get("count") != 879:
        raise AuditError(f"selected-checkpoint train evaluation must cover 879 targets: {run['run_dir']}")
    train_f1 = float(train_overall["macro_f1"])
    train_ce = float(train_eval["weighted_action_cross_entropy"])
    if not math.isfinite(train_f1) or not math.isfinite(train_ce):
        raise AuditError(f"nonfinite selected train metrics: {run['run_dir']}")
    metrics = predictions["metrics"]
    return {
        "variant": run["variant"],
        "seed": run["seed"],
        "run_dir": str(run["run_dir"]),
        "config_sha256": manifest["config_sha256"],
        "artifact_sha256": run["artifact_sha256"],
        "validation_macro_f1": metrics["overall"]["macro_f1"],
        "validation_cross_entropy": predictions["cross_entropy"],
        "validation_per_class_recall": dict(zip(DIRECTIONS, metrics["overall"]["per_class_recall"])),
        "validation_confusion_matrix": metrics["overall"]["confusion_matrix"],
        "validation_prediction_counts_by_class": dict(
            zip(DIRECTIONS, metrics["overall"]["prediction_counts"])
        ),
        "validation_by_position": metrics["by_position_mm"],
        "validation_by_position_and_channel_map": metrics["by_position_mm_and_channel_map"],
        "prediction_unique_class_count": metrics["overall"]["prediction_unique_class_count"],
        "selected_checkpoint_train_target_macro_f1": train_f1,
        "selected_checkpoint_train_target_weighted_ce": train_ce,
        "selected_checkpoint_train_target_unweighted_ce": float(train_eval["unweighted_action_cross_entropy"]),
        "train_target_minus_validation_macro_f1": train_f1 - metrics["overall"]["macro_f1"],
        "train_target_minus_validation_weighted_ce": train_ce - predictions["cross_entropy"],
        "selected_epoch_one_based": int(manifest["selection"]["best_epoch_one_based"]),
        "actual_epochs": int(manifest["selection"]["actual_epochs"]),
        "optimizer_updates": int(manifest["selection"]["optimizer_updates"]),
        "epoch_curve": curve,
        "training_target_description": train_eval.get("evaluation_target"),
        "test_metrics_computed": False,
    }


def _sum_matrices(matrices: Sequence[list[list[int]]]) -> list[list[int]]:
    total = [[0 for _ in DIRECTIONS] for _ in DIRECTIONS]
    for matrix in matrices:
        for row in range(4):
            for col in range(4):
                total[row][col] += int(matrix[row][col])
    return total


def _aggregate(rows: list[dict[str, Any]], snapshot_hashes: dict[str, str], roots: Sequence[Path]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in FORMAL_VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if len(selected) != len(FORMAL_SEEDS):
            raise AuditError(f"expected five seeds for {variant}")
        per_class_recall = {
            direction: _stats(row["validation_per_class_recall"][direction] for row in selected)
            for direction in DIRECTIONS
        }
        by_position: dict[str, Any] = {}
        for position in map(str, VAL_POSITIONS):
            by_position[position] = {
                "macro_f1": _stats(row["validation_by_position"][position]["macro_f1"] for row in selected),
                "per_class_recall": {
                    direction: _stats(
                        row["validation_by_position"][position]["per_class_recall"][idx]
                        for row in selected
                    )
                    for idx, direction in enumerate(DIRECTIONS)
                },
                "summed_confusion_matrix": _sum_matrices(
                    [row["validation_by_position"][position]["confusion_matrix"] for row in selected]
                ),
            }
        map_strata: dict[str, Any] = {}
        for position in map(str, VAL_POSITIONS):
            map_names = sorted(
                {
                    map_name
                    for row in selected
                    for map_name in row["validation_by_position_and_channel_map"][position]
                }
            )
            map_strata[position] = {}
            for map_name in map_names:
                per_seed = [
                    row["validation_by_position_and_channel_map"][position][map_name]
                    for row in selected
                    if map_name in row["validation_by_position_and_channel_map"][position]
                ]
                map_strata[position][map_name] = {
                    "n_seed_position_cells": len(per_seed),
                    "macro_f1": _stats(item["macro_f1"] for item in per_seed),
                }
        max_epochs = max(len(row["epoch_curve"]) for row in selected)
        curves: list[dict[str, Any]] = []
        for epoch in range(1, max_epochs + 1):
            epoch_rows = [row["epoch_curve"][epoch - 1] for row in selected if len(row["epoch_curve"]) >= epoch]
            curves.append(
                {
                    "epoch": epoch,
                    "n_seed_runs": len(epoch_rows),
                    "val_macro_f1": _stats(item["val_macro_f1"] for item in epoch_rows),
                    "val_cross_entropy": _stats(item["val_cross_entropy"] for item in epoch_rows),
                    "train_weighted_ce_online": _stats(item["train_weighted_ce_online"] for item in epoch_rows),
                    "train_supcon_batch_mean_online": _stats(
                        item["train_supcon_batch_mean_online"] for item in epoch_rows
                    ),
                    "epoch_seconds": _stats(item["epoch_seconds"] for item in epoch_rows),
                }
            )
        by_variant[variant] = {
            "validation_macro_f1": _stats(row["validation_macro_f1"] for row in selected),
            "validation_cross_entropy": _stats(row["validation_cross_entropy"] for row in selected),
            "validation_per_class_recall": per_class_recall,
            "validation_summed_confusion_matrix": _sum_matrices(
                [row["validation_confusion_matrix"] for row in selected]
            ),
            "prediction_unique_class_count": _stats(row["prediction_unique_class_count"] for row in selected),
            "validation_prediction_counts_by_class": {
                direction: _stats(row["validation_prediction_counts_by_class"][direction] for row in selected)
                for direction in DIRECTIONS
            },
            "validation_prediction_share_by_class": {
                direction: _stats(row["validation_prediction_counts_by_class"][direction] / 320.0 for row in selected)
                for direction in DIRECTIONS
            },
            "selected_checkpoint_train_target_macro_f1": _stats(
                row["selected_checkpoint_train_target_macro_f1"] for row in selected
            ),
            "selected_checkpoint_train_target_weighted_ce": _stats(
                row["selected_checkpoint_train_target_weighted_ce"] for row in selected
            ),
            "train_target_minus_validation_macro_f1": _stats(
                row["train_target_minus_validation_macro_f1"] for row in selected
            ),
            "train_target_minus_validation_weighted_ce": _stats(
                row["train_target_minus_validation_weighted_ce"] for row in selected
            ),
            "selected_epoch_one_based_by_seed": {
                str(row["seed"]): row["selected_epoch_one_based"] for row in selected
            },
            "actual_epochs_by_seed": {str(row["seed"]): row["actual_epochs"] for row in selected},
            "optimizer_updates_by_seed": {str(row["seed"]): row["optimizer_updates"] for row in selected},
            "validation_by_position": by_position,
            "validation_by_position_and_channel_map": map_strata,
            "learning_curve_mean_by_epoch": curves,
        }
    paired: dict[str, Any] = {}
    main_by_seed = {row["seed"]: row for row in rows if row["variant"] == "ring_main"}
    for comparator in ("ring_ce", "slot_supcon", "ring_shuffle"):
        control_by_seed = {row["seed"]: row for row in rows if row["variant"] == comparator}
        differences = {
            str(seed): main_by_seed[seed]["validation_macro_f1"] - control_by_seed[seed]["validation_macro_f1"]
            for seed in FORMAL_SEEDS
        }
        paired[f"ring_main_minus_{comparator}"] = {
            "per_seed": differences,
            "mean": _stats(differences.values())["mean"],
            "std_population": _stats(differences.values())["std_population"],
        }
    main = by_variant["ring_main"]
    gate = {
        "macro_f1_threshold": 0.80,
        "per_class_mean_recall_threshold": 0.65,
        "macro_f1_pass": main["validation_macro_f1"]["mean"] >= 0.80,
        "per_class_recall_pass": {
            direction: main["validation_per_class_recall"][direction]["mean"] >= 0.65
            for direction in DIRECTIONS
        },
    }
    gate["passed"] = bool(gate["macro_f1_pass"] and all(gate["per_class_recall_pass"].values()))
    return {
        "format": "c170-ring-cuda-v2-validation-summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "validation_only": True,
        "test_metrics_computed": False,
        "standard_deviation": "population standard deviation (ddof=0)",
        "expected_run_count": 20,
        "verified_run_count": len(rows),
        "source_snapshot_code_sha256": snapshot_hashes,
        "input_roots": [str(path.resolve()) for path in roots],
        "gate": gate,
        "variants": by_variant,
        "paired_seed_macro_f1_differences": paired,
        "per_seed": rows,
        "interpretation_boundary": (
            "The loader decodes the complete NPZ and finite-checks X_raw, including registered-test waveforms. "
            "Only train/validation indices enter preprocessing and model computations; test contributes split metadata only. "
            "Position and channel-map strata are descriptive; this single-session split does not identify causal effects. "
            "The C4 ring tensor is discrete and does not establish continuous physical rotation invariance. "
            "The registered test split is not predicted or scored; a failed validation gate does not authorize a test run."
        ),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# C170 Ring CUDA v2 Validation Summary",
        "",
        f"Verified runs: {summary['verified_run_count']}/{summary['expected_run_count']}. Test metrics computed: **false**.",
        "",
        "## Variant results",
        "",
        "| Variant | Macro-F1 mean ± SD | UP recall | DOWN recall | LEFT recall | RIGHT recall | selected train-target F1 | train weighted CE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in FORMAL_VARIANTS:
        values = summary["variants"][variant]
        f1 = values["validation_macro_f1"]
        recalls = values["validation_per_class_recall"]
        train_f1 = values["selected_checkpoint_train_target_macro_f1"]
        train_ce = values["selected_checkpoint_train_target_weighted_ce"]
        lines.append(
            f"| {variant} | {f1['mean']:.4f} ± {f1['std_population']:.4f} | "
            + " | ".join(f"{recalls[name]['mean']:.4f}" for name in DIRECTIONS)
            + f" | {train_f1['mean']:.4f} ± {train_f1['std_population']:.4f} | "
            + f"{train_ce['mean']:.4f} ± {train_ce['std_population']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"ring_main macro-F1 gate: {'PASS' if summary['gate']['macro_f1_pass'] else 'FAIL'} "
            f"(mean {summary['variants']['ring_main']['validation_macro_f1']['mean']:.4f} vs 0.80).",
        ]
    )
    for direction, passed in summary["gate"]["per_class_recall_pass"].items():
        mean = summary["variants"]["ring_main"]["validation_per_class_recall"][direction]["mean"]
        lines.append(f"- {direction} mean recall {mean:.4f} vs 0.65: {'PASS' if passed else 'FAIL'}.")
    lines.extend(
        [
            "",
            f"Overall ring_main gate: **{'PASS' if summary['gate']['passed'] else 'FAIL'}**.",
            "",
            "## Validation macro-F1 by position",
            "",
            "Mean ± population SD across five seeds.",
            "",
            "| Position (mm) | ring_main | ring_ce | slot_supcon | ring_shuffle |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for position in map(str, VAL_POSITIONS):
        cells = [
            summary["variants"][variant]["validation_by_position"][position]["macro_f1"]
            for variant in FORMAL_VARIANTS
        ]
        lines.append(
            f"| {position} | "
            + " | ".join(f"{item['mean']:.4f} ± {item['std_population']:.4f}" for item in cells)
            + " |"
        )
    lines.extend(["", "## Paired seed differences", "", "Difference is ring_main minus comparator validation macro-F1.", ""])
    for name, item in summary["paired_seed_macro_f1_differences"].items():
        lines.append(f"- {name}: mean {item['mean']:+.4f}, population SD {item['std_population']:.4f}.")
    lines.extend(
        [
            "",
            "Population standard deviations (ddof=0) are used throughout. Train/validation metric gaps are in the per-seed CSV and JSON.",
            "",
            "## Shuffle validation prediction distribution",
            "",
        ]
    )
    for direction in DIRECTIONS:
        value = summary["variants"]["ring_shuffle"]["validation_prediction_counts_by_class"][direction]
        lines.append(f"- predicted {direction}: {value['mean']:.1f} ± {value['std_population']:.1f} / 320 across seeds.")
    lines.extend(["", summary["interpretation_boundary"], ""])
    return "\n".join(lines)


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    fields = (
        "variant",
        "seed",
        "validation_macro_f1",
        "validation_cross_entropy",
        "validation_recall_up",
        "validation_recall_down",
        "validation_recall_left",
        "validation_recall_right",
        "selected_checkpoint_train_target_macro_f1",
        "selected_checkpoint_train_target_weighted_ce",
        "train_target_minus_validation_macro_f1",
        "train_target_minus_validation_weighted_ce",
        "selected_epoch_one_based",
        "actual_epochs",
        "optimizer_updates",
        "prediction_unique_class_count",
        "run_dir",
    )
    with (output_dir / "per_seed_validation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_seed"]:
            writer.writerow(
                {
                    **{key: row[key] for key in fields if key in row},
                    **{
                        "validation_recall_" + direction.lower(): row["validation_per_class_recall"][direction]
                        for direction in DIRECTIONS
                    },
                }
            )
    curve_fields = ("variant", "epoch", "n_seed_runs", "val_macro_f1_mean", "val_macro_f1_std", "val_ce_mean", "train_weighted_ce_mean", "train_supcon_mean", "epoch_seconds_mean")
    with (output_dir / "learning_curve_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_fields)
        writer.writeheader()
        for variant in FORMAL_VARIANTS:
            for row in summary["variants"][variant]["learning_curve_mean_by_epoch"]:
                writer.writerow(
                    {
                        "variant": variant,
                        "epoch": row["epoch"],
                        "n_seed_runs": row["n_seed_runs"],
                        "val_macro_f1_mean": row["val_macro_f1"]["mean"],
                        "val_macro_f1_std": row["val_macro_f1"]["std_population"],
                        "val_ce_mean": row["val_cross_entropy"]["mean"],
                        "train_weighted_ce_mean": row["train_weighted_ce_online"]["mean"],
                        "train_supcon_mean": row["train_supcon_batch_mean_online"]["mean"],
                        "epoch_seconds_mean": row["epoch_seconds"]["mean"],
                    }
                )
    paired_path = output_dir / "paired_seed_differences.csv"
    with paired_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("comparison", "seed", "ring_main_minus_control"))
        writer.writeheader()
        for comparison, item in summary["paired_seed_macro_f1_differences"].items():
            for seed, difference in item["per_seed"].items():
                writer.writerow({"comparison": comparison, "seed": seed, "ring_main_minus_control": difference})


def _snapshot_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "python_version": sys.version,
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
    }
    try:
        import numpy as np
        import scipy
        import torch

        info.update(
            {
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "torch_version": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "cuda_devices": [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                    }
                    for index in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as exc:
        info["torch_environment_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        info["nvidia_smi"] = result.stdout.strip().splitlines()
    except Exception as exc:
        info["nvidia_smi_unavailable"] = f"{type(exc).__name__}: {exc}"
    try:
        import psutil

        info["memory_total_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        info["memory_total_bytes"] = None
    return info


def _create_source_snapshot(output_root: Path) -> Path:
    root = output_root.expanduser().resolve()
    if not root.is_dir():
        raise AuditError(f"formal output root must already exist: {root}")
    snapshot_dir = root / "source_snapshot"
    if snapshot_dir.exists():
        raise AuditError(f"refusing to overwrite source snapshot: {snapshot_dir}")
    snapshot_dir.mkdir()
    (snapshot_dir / "code").mkdir()
    (snapshot_dir / "docs").mkdir()
    sources: dict[str, dict[str, str]] = {}
    for key, relative in SOURCE_PATHS.items():
        source = PROJECT_DIR / relative
        if not source.is_file():
            raise AuditError(f"required snapshot source is missing: {source}")
        saved_rel = Path("code") / f"{key}_{source.name}"
        saved = snapshot_dir / saved_rel
        shutil.copy2(source, saved)
        digest = _sha256(source)
        if _sha256(saved) != digest:
            raise AuditError(f"source copy hash mismatch: {source}")
        sources[key] = {"path": saved_rel.as_posix(), "origin": str(source), "sha256": digest}
    documents: dict[str, dict[str, str]] = {}
    document_paths = {
        "v2_plan": Path("docs/plans/2026-09-28-c170-ring-cross-position-cuda.md"),
        "prior_design_context": Path("docs/designs/2026-09-28-latent-angle-temporal-classifier.md"),
    }
    for key, relative in document_paths.items():
        source = PROJECT_DIR / relative
        if not source.is_file():
            raise AuditError(f"required snapshot document is missing: {source}")
        saved_rel = Path("docs") / source.name
        saved = snapshot_dir / saved_rel
        shutil.copy2(source, saved)
        documents[key] = {"path": saved_rel.as_posix(), "origin": str(source), "sha256": _sha256(saved)}
    launcher = PROJECT_DIR / 'scripts/launchers/launch_c170_ring_cuda_v2.py'
    orchestration: dict[str, Any] | None = None
    if launcher.is_file():
        destination = snapshot_dir / "code" / launcher.name
        shutil.copy2(launcher, destination)
        orchestration = {
            "path": destination.relative_to(snapshot_dir).as_posix(),
            "origin": str(launcher),
            "sha256": _sha256(destination),
        }
    environment = _snapshot_environment()
    environment_path = snapshot_dir / "environment.json"
    environment_path.write_text(json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format": "c170-ring-cuda-v2-source-snapshot",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_files": sources,
        "documents": documents,
        "orchestration_launcher": orchestration,
        "environment": {"path": "environment.json", "sha256": _sha256(environment_path)},
    }
    (snapshot_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return snapshot_dir


def _run_snapshot(root: Path) -> int:
    snapshot = _create_source_snapshot(root)
    manifest = _read_json(snapshot / "snapshot_manifest.json")
    print(f"SOURCE_SNAPSHOT_COMPLETE path={snapshot}")
    for key in SOURCE_KEYS:
        print(f"{key}={manifest['source_files'][key]['sha256']}")
    print(f"environment_sha256={manifest['environment']['sha256']}")
    return 0


def _run_analyze(roots: Sequence[Path], output_dir: Path) -> int:
    snapshot_dir, source_hashes = _read_verified_snapshot(roots)
    marker_paths = _candidate_markers(roots)
    if not marker_paths:
        raise AuditError("no completed run markers found")
    verified = [_verify_run(path, source_hashes) for path in marker_paths]
    exact = _validate_exact_matrix(verified)
    dataset_hashes = {run["config"].get("dataset_sha256") for run in exact}
    if len(dataset_hashes) != 1 or None in dataset_hashes:
        raise AuditError("formal runs do not share one verified dataset SHA-256")
    analyzed = [_analyze_run(run) for run in exact]
    summary = _aggregate(analyzed, source_hashes, roots)
    summary["source_snapshot_path"] = str(snapshot_dir)
    _write_report(output_dir, summary)
    print(f"VALIDATION_SUMMARY_COMPLETE path={output_dir.resolve()}")
    print(f"verified_runs={summary['verified_run_count']}/{summary['expected_run_count']}")
    print(f"ring_main_gate={'PASS' if summary['gate']['passed'] else 'FAIL'}")
    print(f"test_metrics_computed={summary['test_metrics_computed']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot", help="copy source and runtime provenance into a new formal run root")
    snapshot.add_argument("--output-root", type=Path, required=True)
    analyze = subparsers.add_parser("analyze", help="verify and summarize a complete 20-run validation suite")
    analyze.add_argument("--root", type=Path, action="append", required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "snapshot":
            return _run_snapshot(args.output_root)
        return _run_analyze(args.root, args.output_dir)
    except (AuditError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
