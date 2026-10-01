"""Fail-closed, validation-only summary for the fixed C170 harmonic-phase v4 suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

import analyze_c170_ring_cuda_v2_results as v2
import analyze_c170_shared_scale_v3_results as v3


FORMAL_VARIANTS = ("harmonic_gram", "harmonic_power", "harmonic_phase", "harmonic_shuffle")
FORMAL_SEEDS = (17, 29, 43, 61, 89)
VAL_POSITIONS = (5, 40, 90, 130)
DIRECTIONS = v2.DIRECTIONS
EXPECTED_SCALE = 11123.037109375
ACTIVE_FEATURES = {
    "harmonic_gram": list(range(10)),
    "harmonic_power": [0, 1],
    "harmonic_phase": list(range(12)),
    "harmonic_shuffle": list(range(10)),
}
FEATURE_NAMES = [
    "asinh_s", "log1p_p", "lag1_re", "lag1_im", "lag4_re", "lag4_im",
    "lag16_re", "lag16_im", "lag64_re", "lag64_im", "asinh_a", "asinh_b",
]
AuditError = v2.AuditError


def _validate_exact_matrix(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(variant, seed) for variant in FORMAL_VARIANTS for seed in FORMAL_SEEDS}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        pair = (str(run.get("variant")), int(run.get("seed", -1)))
        if pair not in expected or pair in found:
            raise AuditError(f"unexpected or duplicate v4 variant/seed: {pair}")
        found[pair] = run
    missing = sorted(expected - set(found))
    if missing:
        raise AuditError(f"missing v4 variant/seed pairs: {missing}")
    return sorted(found.values(), key=lambda row: (FORMAL_VARIANTS.index(row["variant"]), row["seed"]))


def _verify_snapshot(root: Path) -> tuple[Path, dict[str, str], dict[str, Any]]:
    snapshot_dir = root.resolve() / "source_snapshot"
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    snapshot = v2._read_json(manifest_path)
    if "v4" not in str(snapshot.get("format", "")).lower():
        raise AuditError(f"unexpected v4 source snapshot format: {manifest_path}")
    code_hashes = snapshot.get("provenance_code_sha256")
    copied_hashes = snapshot.get("snapshot_file_sha256")
    if not isinstance(code_hashes, dict) or not isinstance(copied_hashes, dict) or not code_hashes:
        raise AuditError(f"source snapshot is missing code/file hashes: {manifest_path}")
    snapshot_docs = {
        "docs/designs/2026-09-29-c170-harmonic-phase-v4.md": snapshot_dir / "docs" / "design.md",
        "docs/plans/2026-09-29-c170-harmonic-phase-v4.md": snapshot_dir / "docs" / "plan.md",
    }
    for relative, expected_hash in copied_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise AuditError(f"invalid source snapshot entry: {manifest_path}")
        # The v4 source snapshot stores design/plan copies under docs/design.md
        # and docs/plan.md; all code and test copies live under code/.
        source = snapshot_docs.get(relative, snapshot_dir / "code" / relative)
        if not source.is_file() or v2._sha256(source) != expected_hash:
            raise AuditError(f"source snapshot file hash mismatch: {source}")
    for key, digest in code_hashes.items():
        if not isinstance(digest, str) or len(digest) != 64:
            raise AuditError(f"invalid source code hash for {key}: {manifest_path}")
    return snapshot_dir, {str(key): str(value) for key, value in code_hashes.items()}, snapshot


def _check_safe_weights(path: Path) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise AuditError(f"empty numeric state dict: {path}")
            for key in archive.files:
                value = archive[key]
                if value.dtype.hasobject or value.dtype.kind not in "biuf":
                    raise AuditError(f"unsafe/non-numeric state tensor {key}: {path}")
                if value.dtype.kind == "f" and not np.isfinite(value).all():
                    raise AuditError(f"non-finite state tensor {key}: {path}")
    except (OSError, ValueError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError(f"cannot safely inspect NPZ state dict {path}: {exc}") from exc


def _validation_signature(path: Path) -> str:
    """Hash split identity/labels only, omitting predictions and probabilities."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    signature = sorted(
        (
            int(row["sample_index"]),
            int(row["distance_mm"]),
            str(row["true_direction"]),
            tuple(int(value) for value in row["channel_map"].split(",")),
        )
        for row in rows
    )
    return hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_run(marker_path: Path, code_hashes: dict[str, str], dataset_sha256: str) -> dict[str, Any]:
    run_dir = marker_path.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise AuditError(f"missing v4 run manifest: {run_dir}")
    marker = v2._read_json(marker_path)
    manifest = v2._read_json(manifest_path)
    if marker.get("format") != "c170-ring-cuda-v4-complete" or marker.get("status") != "completed":
        raise AuditError(f"invalid v4 completion marker: {marker_path}")
    if manifest.get("format") != "c170-ring-cuda-v4-run" or manifest.get("status") != "completed":
        raise AuditError(f"v4 manifest is not completed: {manifest_path}")
    if marker.get("manifest_sha256") != v2._sha256(manifest_path):
        raise AuditError(f"manifest SHA-256 mismatch: {manifest_path}")
    if marker.get("test_metrics_computed") is not False or manifest.get("test_metrics_computed") is not False:
        raise AuditError(f"test metrics are not explicitly disabled: {run_dir}")
    access = manifest.get("test_access")
    if not isinstance(access, dict) or access.get("loaded_as_part_of_complete_npz") is not True:
        raise AuditError(f"full-NPZ load provenance is missing: {run_dir}")
    if any(access.get(key) is not False for key in (
        "used_as_model_input", "preprocessed", "used_to_estimate_statistics", "predicted_or_scored"
    )):
        raise AuditError(f"test boundary violation recorded: {run_dir}")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise AuditError(f"missing run config: {run_dir}")
    config_sha = v2._canonical_config_sha(config)
    if config_sha != manifest.get("config_sha256") or config_sha != marker.get("config_sha256"):
        raise AuditError(f"canonical config SHA-256 mismatch: {run_dir}")
    variant = str(config.get("variant"))
    try:
        seed = int(config["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"invalid v4 variant/seed: {run_dir}") from exc
    if variant not in FORMAL_VARIANTS or seed not in FORMAL_SEEDS:
        raise AuditError(f"unexpected v4 variant/seed {variant}/{seed}")

    expected_config = {
        "model_variant": "harmonic_phase_classifier_v4",
        "active_feature_indices": ACTIVE_FEATURES[variant],
        "feature_names": FEATURE_NAMES,
        "shuffle_labels": variant == "harmonic_shuffle",
        "max_epochs": 160,
        "patience": 25,
        "cpu_threads": 2,
        "device": "cuda:0",
        "dataset_sha256": dataset_sha256,
        "train_distance_mm": [0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160],
        "validation_distance_mm": list(VAL_POSITIONS),
        "registered_test_distance_mm": [15, 25, 110, 150],
        "excluded_distance_mm": [70],
        "expected_split_counts": {"train": 879, "validation": 320, "registered_test": 320},
        "batch_sizes_per_epoch": [44] * 19 + [43],
        "updates_per_epoch": 20,
        "sampler_group_key": "original_distance_mm_and_direction_label",
        "sampler_rng": "numpy_seedsequence_seed_epoch_independent_of_model_rng",
        "sample_weight_rule": "position_voronoi_weight/(4*original_group_size)",
        "position_weight_period_mm": 170.0,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "cross_entropy_label_smoothing": 0.0,
        "batch_ce_objective": "20*sum(sample_weight_i*per_sample_ce_i)",
        "supcon_weight": 0.0,
        "preprocessing": "v3_filter_per_trial_demean_one_shared_train_median_scale",
        "preprocessing_scale_required": EXPECTED_SCALE,
        "preprocessing_scale_fit_split": "train",
        "feature_lags": [1, 4, 16, 64],
        "lag_boundary": "zero_prefix_no_circular_wrap",
        "safe_state_dict_format": "numpy_npz_numeric_arrays",
        "safe_state_dict_reload": "numpy_load_allow_pickle_false_then_strict_state_dict",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
        "physical_order_confirmed_by_operator": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise AuditError(f"frozen config mismatch for {key} in {run_dir}: {config.get(key)!r}")
    if manifest.get("provenance", {}).get("code_sha256") != code_hashes:
        raise AuditError(f"run source hashes differ from source snapshot: {run_dir}")
    if not isinstance(config.get("dataset_path"), str) or not config["dataset_path"]:
        raise AuditError(f"dataset path missing from config: {run_dir}")

    split = manifest.get("split", {})
    if split.get("split_counts") != {"train": 879, "val": 320, "test": 320}:
        raise AuditError(f"split counts changed in {run_dir}")
    expected_shuffle_seed = seed + 10_000 if variant == "harmonic_shuffle" else None
    if split.get("label_shuffle_seed") != expected_shuffle_seed:
        raise AuditError(f"label shuffle seed mismatch: {run_dir}")
    mapping = split.get("label_shuffle_mapping")
    if not isinstance(mapping, list) or (variant == "harmonic_shuffle" and not mapping) or (
        variant != "harmonic_shuffle" and mapping
    ):
        raise AuditError(f"label shuffle mapping mismatch: {run_dir}")

    prep = manifest.get("preprocessing", {})
    if prep.get("normalization_fit_splits") != ["train"] or prep.get("processed_splits") != ["train", "validation"]:
        raise AuditError(f"preprocessing split boundary mismatch: {run_dir}")
    if prep.get("test_preprocessed") is not False or prep.get("channel_map_as_model_input") is not False:
        raise AuditError(f"unexpected test or channel-map model preprocessing: {run_dir}")
    if prep.get("sensor_scale_axes") != ["train_samples", "time", "channels"]:
        raise AuditError(f"shared scale axes changed: {run_dir}")
    if float(prep.get("shared_sensor_scale", float("nan"))) != EXPECTED_SCALE:
        raise AuditError(f"recorded shared scale changed: {run_dir}")
    if marker.get("selected_epoch_one_based") != manifest.get("selection", {}).get("best_epoch_one_based"):
        raise AuditError(f"selected epoch mismatch: {run_dir}")

    names = {
        "best_checkpoint": "best_checkpoint.pt",
        "safe_state_dict": "best_weights.npz",
        "learning_curve": "learning_curve.csv",
        "validation_predictions": "validation_predictions.csv",
    }
    artifacts = manifest.get("artifacts")
    marker_hashes = marker.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != set(names):
        raise AuditError(f"artifact set changed: {run_dir}")
    if not isinstance(marker_hashes, dict) or set(marker_hashes) != set(names):
        raise AuditError(f"completion artifact hash set changed: {run_dir}")
    hashes: dict[str, str] = {}
    for name, expected_path in names.items():
        record = artifacts[name]
        path = run_dir / expected_path
        if not isinstance(record, dict) or record.get("path") != expected_path or not path.is_file():
            raise AuditError(f"missing or misnamed artifact {name}: {run_dir}")
        digest = v2._sha256(path)
        if record.get("sha256") != digest or marker_hashes.get(name) != digest:
            raise AuditError(f"artifact hash mismatch: {path}")
        hashes[name] = digest
        if name == "safe_state_dict":
            _check_safe_weights(path)
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": run_dir,
        "manifest": manifest,
        "marker": marker,
        "config": config,
        "artifact_sha256": hashes,
    }


def _read_validation_predictions(run: dict[str, Any]) -> dict[str, Any]:
    path = run["run_dir"] / run["manifest"]["artifacts"]["validation_predictions"]["path"]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 320:
        raise AuditError(f"expected 320 validation prediction rows: {path}")
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    distances: list[int] = []
    samples: set[int] = set()
    signature: list[tuple[int, int, str, tuple[int, ...]]] = []
    for row in rows:
        if row.get("variant") != run["variant"] or int(row.get("seed", -1)) != run["seed"]:
            raise AuditError(f"prediction identity mismatch: {path}")
        sample = int(row["sample_index"])
        if sample in samples:
            raise AuditError(f"duplicate validation sample index {sample}: {path}")
        samples.add(sample)
        distance = int(row["distance_mm"])
        truth_name = row["true_direction"]
        pred_name = row["prediction"]
        if distance not in VAL_POSITIONS or truth_name not in DIRECTIONS or pred_name not in DIRECTIONS:
            raise AuditError(f"invalid validation label/position: {path}")
        truth = DIRECTIONS.index(truth_name)
        prediction = DIRECTIONS.index(pred_name)
        probability = [float(row[f"probability_{name.lower()}"]) for name in DIRECTIONS]
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probability):
            raise AuditError(f"invalid validation probability: {path}")
        if not math.isclose(sum(probability), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise AuditError(f"probabilities do not sum to one: {path}")
        if max(range(4), key=lambda index: probability[index]) != prediction:
            raise AuditError(f"prediction does not match probability argmax: {path}")
        mapping = tuple(int(value) for value in row["channel_map"].split(","))
        if len(mapping) != 4:
            raise AuditError(f"invalid channel-map metadata: {path}")
        labels.append(truth)
        predictions.append(prediction)
        probabilities.append(probability)
        distances.append(distance)
        signature.append((sample, distance, truth_name, mapping))

    by_position: dict[str, Any] = {}
    for position in VAL_POSITIONS:
        indices = [index for index, item in enumerate(distances) if item == position]
        if len(indices) != 80:
            raise AuditError(f"expected 80 validation rows at {position} mm: {path}")
        if any(sum(labels[index] == action for index in indices) != 20 for action in range(4)):
            raise AuditError(f"validation action balance changed at {position} mm: {path}")
        expected_map = (0, 2, 4, 6) if position in (5, 40) else (0, 1, 2, 3)
        if {tuple(int(value) for value in row["channel_map"].split(",")) for row in rows
                if int(row["distance_mm"]) == position} != {expected_map}:
            raise AuditError(f"channel-map metadata changed at {position} mm: {path}")
        by_position[str(position)] = v2._classification_metrics(
            [labels[index] for index in indices], [predictions[index] for index in indices]
        )
    overall = v2._classification_metrics(labels, predictions)
    cross_entropy = sum(-math.log(max(probabilities[i][labels[i]], 1e-300)) for i in range(320)) / 320.0

    selection = run["manifest"].get("selection", {})
    recorded = selection.get("validation_metrics", {}).get("overall", {})
    if recorded.get("confusion_matrix") != overall["confusion_matrix"]:
        raise AuditError(f"manifest validation confusion disagrees with CSV: {path}")
    if not math.isclose(float(recorded.get("macro_f1", float("nan"))), overall["macro_f1"], rel_tol=0.0, abs_tol=1e-10):
        raise AuditError(f"manifest validation F1 disagrees with CSV: {path}")
    for measured, recorded_recall in zip(overall["per_class_recall"], recorded.get("per_class_recall", [])):
        if not math.isclose(float(recorded_recall), measured, rel_tol=0.0, abs_tol=1e-10):
            raise AuditError(f"manifest validation recall disagrees with CSV: {path}")
    if not math.isclose(float(selection.get("validation_cross_entropy", float("nan"))), cross_entropy, rel_tol=0.0, abs_tol=1e-5):
        raise AuditError(f"manifest validation CE disagrees with CSV: {path}")
    for position, measured in by_position.items():
        recorded_position = selection["validation_metrics"]["by_position_mm"][position]
        if recorded_position.get("confusion_matrix") != measured["confusion_matrix"]:
            raise AuditError(f"manifest position confusion disagrees with CSV at {position} mm: {path}")
        if not math.isclose(float(recorded_position.get("macro_f1", float("nan"))), measured["macro_f1"], rel_tol=0.0, abs_tol=1e-10):
            raise AuditError(f"manifest position F1 disagrees with CSV at {position} mm: {path}")
    train_eval = run["manifest"].get("selected_checkpoint_train_evaluation", {})
    train_overall = train_eval.get("overall", {})
    if train_overall.get("count") != 879:
        raise AuditError(f"expected 879 selected-checkpoint training targets: {path}")
    train_f1 = float(train_overall["macro_f1"])
    train_ce = float(train_eval["weighted_action_cross_entropy"])
    if not math.isfinite(train_f1) or not math.isfinite(train_ce):
        raise AuditError(f"non-finite selected-checkpoint training metrics: {path}")
    return {
        "variant": run["variant"],
        "seed": run["seed"],
        "run_dir": str(run["run_dir"]),
        "config_sha256": run["manifest"]["config_sha256"],
        "artifact_sha256": run["artifact_sha256"],
        "validation_macro_f1": overall["macro_f1"],
        "validation_cross_entropy": cross_entropy,
        "validation_per_class_recall": dict(zip(DIRECTIONS, overall["per_class_recall"])),
        "validation_confusion_matrix": overall["confusion_matrix"],
        "validation_prediction_counts_by_class": dict(zip(DIRECTIONS, overall["prediction_counts"])),
        "prediction_unique_class_count": overall["prediction_unique_class_count"],
        "validation_by_position": by_position,
        "validation_signature_sha256": _validation_signature(path),
        "selected_checkpoint_train_target_macro_f1": train_f1,
        "selected_checkpoint_train_target_weighted_ce": train_ce,
        "train_target_minus_validation_macro_f1": train_f1 - overall["macro_f1"],
        "train_target_minus_validation_weighted_ce": train_ce - cross_entropy,
        "selected_epoch_one_based": int(run["manifest"]["selection"]["best_epoch_one_based"]),
        "actual_epochs": int(run["manifest"]["selection"]["actual_epochs"]),
        "optimizer_updates": int(run["manifest"]["selection"]["optimizer_updates"]),
        "validation_signature": sorted(signature),
    }


def _load_v4(root: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    snapshot_dir, code_hashes, snapshot = _verify_snapshot(root)
    marker_paths = sorted((root.resolve()).rglob("RUN_COMPLETE.json"))
    if len(marker_paths) != 20:
        raise AuditError(f"expected exactly 20 v4 completion markers, found {len(marker_paths)}")
    dataset_hashes: set[str] = set()
    runs = []
    for marker_path in marker_paths:
        run = _verify_run(marker_path, code_hashes, str(snapshot.get("dataset", {}).get("sha256", "")))
        dataset_hashes.add(str(run["config"]["dataset_sha256"]))
        runs.append(run)
    exact = _validate_exact_matrix(runs)
    if len(dataset_hashes) != 1 or "" in dataset_hashes:
        raise AuditError("v4 runs do not share one dataset SHA-256")
    if snapshot.get("dataset", {}).get("sha256") not in dataset_hashes:
        raise AuditError("v4 dataset hash differs from source snapshot")
    summaries = [_read_validation_predictions(run) for run in exact]
    baseline_signature: str | None = None
    for item in summaries:
        signature = item["validation_signature_sha256"]
        if baseline_signature is None:
            baseline_signature = signature
        elif baseline_signature != signature:
            raise AuditError("validation sample identities/labels differ across v4 runs")
    return summaries, code_hashes, snapshot


def _load_v3_baseline(root: Path, expected_dataset_hash: str) -> list[dict[str, Any]]:
    _snapshot_dir, source_hashes, _snapshot = v3._read_v3_snapshot(root)
    rows = []
    for seed in FORMAL_SEEDS:
        marker = root.resolve() / f"shared_scale_ce__seed-{seed}" / "RUN_COMPLETE.json"
        if not marker.is_file():
            raise AuditError(f"missing v3 shared_scale_ce seed {seed}: {marker}")
        checked = v3._verify_v3_run(marker, source_hashes)
        if checked["variant"] != "shared_scale_ce" or checked["seed"] != seed:
            raise AuditError(f"wrong v3 comparator identity: {marker}")
        if checked["config"].get("dataset_sha256") != expected_dataset_hash:
            raise AuditError("v3 comparator dataset SHA-256 differs from v4")
        row = v3._analyze_run(checked, True)
        prediction_path = checked["run_dir"] / checked["manifest"]["artifacts"]["validation_predictions"]["path"]
        row["validation_signature_sha256"] = _validation_signature(prediction_path)
        rows.append(row)
    return rows


def _stats(values: Sequence[float]) -> dict[str, Any]:
    return v2._stats(values)


def _aggregate(rows: list[dict[str, Any]], baseline: list[dict[str, Any]], roots: tuple[Path, Path], code_hashes: dict[str, str]) -> dict[str, Any]:
    by_variant_seed = {variant: {row["seed"]: row for row in rows if row["variant"] == variant} for variant in FORMAL_VARIANTS}
    baseline_by_seed = {row["seed"]: row for row in baseline}
    if set(baseline_by_seed) != set(FORMAL_SEEDS):
        raise AuditError("v3 comparator does not contain one run for every registered seed")
    variants: dict[str, Any] = {}
    for variant in FORMAL_VARIANTS:
        selected = [by_variant_seed[variant][seed] for seed in FORMAL_SEEDS]
        by_position = {}
        for position in map(str, VAL_POSITIONS):
            cells = [row["validation_by_position"][position] for row in selected]
            by_position[position] = {
                "macro_f1": _stats([cell["macro_f1"] for cell in cells]),
                "per_class_recall": {
                    direction: _stats([cell["per_class_recall"][index] for cell in cells])
                    for index, direction in enumerate(DIRECTIONS)
                },
            }
        variants[variant] = {
            "validation_macro_f1": _stats([row["validation_macro_f1"] for row in selected]),
            "validation_cross_entropy": _stats([row["validation_cross_entropy"] for row in selected]),
            "validation_per_class_recall": {
                direction: _stats([row["validation_per_class_recall"][direction] for row in selected])
                for direction in DIRECTIONS
            },
            "validation_prediction_unique_class_count": _stats([row["prediction_unique_class_count"] for row in selected]),
            "selected_checkpoint_train_target_macro_f1": _stats([row["selected_checkpoint_train_target_macro_f1"] for row in selected]),
            "selected_checkpoint_train_target_weighted_ce": _stats([row["selected_checkpoint_train_target_weighted_ce"] for row in selected]),
            "train_target_minus_validation_macro_f1": _stats([row["train_target_minus_validation_macro_f1"] for row in selected]),
            "train_target_minus_validation_weighted_ce": _stats([row["train_target_minus_validation_weighted_ce"] for row in selected]),
            "selected_epoch_one_based_by_seed": {str(row["seed"]): row["selected_epoch_one_based"] for row in selected},
            "actual_epochs_by_seed": {str(row["seed"]): row["actual_epochs"] for row in selected},
            "validation_by_position": by_position,
        }
    comparisons = {
        "harmonic_gram_minus_harmonic_power": {
            seed: by_variant_seed["harmonic_gram"][seed]["validation_macro_f1"] - by_variant_seed["harmonic_power"][seed]["validation_macro_f1"]
            for seed in FORMAL_SEEDS
        },
        "harmonic_gram_minus_harmonic_phase": {
            seed: by_variant_seed["harmonic_gram"][seed]["validation_macro_f1"] - by_variant_seed["harmonic_phase"][seed]["validation_macro_f1"]
            for seed in FORMAL_SEEDS
        },
        "harmonic_gram_minus_harmonic_shuffle": {
            seed: by_variant_seed["harmonic_gram"][seed]["validation_macro_f1"] - by_variant_seed["harmonic_shuffle"][seed]["validation_macro_f1"]
            for seed in FORMAL_SEEDS
        },
        "harmonic_gram_minus_v3_shared_scale_ce": {
            seed: by_variant_seed["harmonic_gram"][seed]["validation_macro_f1"] - baseline_by_seed[seed]["validation_macro_f1"]
            for seed in FORMAL_SEEDS
        },
    }
    paired = {
        name: {
            "per_seed": {str(seed): value for seed, value in values.items()},
            **_stats(list(values.values())),
        }
        for name, values in comparisons.items()
    }
    return {
        "format": "c170-harmonic-phase-v4-validation-summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "validation_only": True,
        "test_metrics_computed": False,
        "expected_run_count": 20,
        "verified_run_count": len(rows),
        "standard_deviation": "population standard deviation (ddof=0)",
        "v4_root": str(roots[0].resolve()),
        "v3_comparator_root": str(roots[1].resolve()),
        "v4_source_snapshot_code_sha256": code_hashes,
        "variants": variants,
        "paired_seed_macro_f1_differences": paired,
        "per_seed": rows,
        "v3_shared_scale_ce_per_seed": baseline,
        "interpretation_boundary": (
            "Metrics are recomputed only from the 320-row validation prediction CSVs and selected-checkpoint training metrics in the manifests. "
            "The complete compressed NPZ may have been decoded and finite-checked by the runner, but test waveforms were not preprocessed, "
            "used to fit statistics, supplied to a model, predicted, or scored. The harmonic_gram/power features are invariant to a common "
            "complex phase in the discrete first-harmonic representation; this does not establish continuous physical SO(2) invariance. "
            "The harmonic_shuffle training score is against permuted training targets. A failed validation gate does not authorize test evaluation."
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# C170 Harmonic-Phase v4 Validation Summary",
        "",
        f"Verified v4 runs: {summary['verified_run_count']}/{summary['expected_run_count']}. Test metrics computed: false.",
        "Mean ± population SD across five seeds; all scores below are validation-only.",
        "",
        "| Variant | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Validation CE | Selected train-target F1 | Train-target F1 gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in FORMAL_VARIANTS:
        item = summary["variants"][variant]
        cells = [
            f"{item['validation_macro_f1']['mean']:.4f} ± {item['validation_macro_f1']['std_population']:.4f}",
            *(f"{item['validation_per_class_recall'][d]['mean']:.4f} ± {item['validation_per_class_recall'][d]['std_population']:.4f}" for d in DIRECTIONS),
            f"{item['validation_cross_entropy']['mean']:.4f} ± {item['validation_cross_entropy']['std_population']:.4f}",
            f"{item['selected_checkpoint_train_target_macro_f1']['mean']:.4f} ± {item['selected_checkpoint_train_target_macro_f1']['std_population']:.4f}",
            f"{item['train_target_minus_validation_macro_f1']['mean']:+.4f} ± {item['train_target_minus_validation_macro_f1']['std_population']:.4f}",
        ]
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Validation Macro-F1 by Position", "", "| Position (mm) | harmonic_gram | harmonic_power | harmonic_phase | harmonic_shuffle |", "|---:|---:|---:|---:|---:|"])
    for position in map(str, VAL_POSITIONS):
        cells = [summary["variants"][variant]["validation_by_position"][position]["macro_f1"] for variant in FORMAL_VARIANTS]
        lines.append(f"| {position} | " + " | ".join(f"{v['mean']:.4f} ± {v['std_population']:.4f}" for v in cells) + " |")
    lines.extend(["", "## Paired Seed Differences", "", "Difference is harmonic_gram minus comparator validation macro-F1.", ""])
    for name, comparison in summary["paired_seed_macro_f1_differences"].items():
        values = ", ".join(f"{seed}:{value:+.4f}" for seed, value in comparison["per_seed"].items())
        lines.append(f"- {name}: {comparison['mean']:+.4f} ± {comparison['std_population']:.4f}; {values}.")
    lines.extend([
        "",
        "The harmonic_shuffle train-target F1 is measured against its permuted training labels. The full NPZ may be decoded and finite-checked; the registered test split was not preprocessed, used to estimate statistics, predicted, or scored.",
        "",
        summary["interpretation_boundary"],
        "",
    ])
    return "\n".join(lines)


def _write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "validation_summary.md").write_text(_markdown(summary), encoding="utf-8")
    fields = [
        "variant", "seed", "validation_macro_f1", "validation_cross_entropy",
        "selected_checkpoint_train_target_macro_f1", "selected_checkpoint_train_target_weighted_ce",
        "train_target_minus_validation_macro_f1", "train_target_minus_validation_weighted_ce",
        "selected_epoch_one_based", "actual_epochs", "optimizer_updates", "prediction_unique_class_count", "run_dir",
    ]
    fields.extend(f"validation_recall_{d.lower()}" for d in DIRECTIONS)
    fields.extend(f"validation_f1_at_{p}mm" for p in VAL_POSITIONS)
    with (output_dir / "per_seed_validation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_seed"]:
            out = {key: row[key] for key in fields if key in row}
            out.update({f"validation_recall_{d.lower()}": row["validation_per_class_recall"][d] for d in DIRECTIONS})
            out.update({f"validation_f1_at_{p}mm": row["validation_by_position"][str(p)]["macro_f1"] for p in VAL_POSITIONS})
            writer.writerow(out)
    with (output_dir / "paired_seed_differences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("comparison", "seed", "macro_f1_difference"))
        writer.writeheader()
        for name, comparison in summary["paired_seed_macro_f1_differences"].items():
            for seed, value in comparison["per_seed"].items():
                writer.writerow({"comparison": name, "seed": seed, "macro_f1_difference": value})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--v3-root", type=Path, default=Path(__file__).resolve().parent / "outputs" / "c170_ring_v3_formal_cuda_20260929")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        v4_rows, source_hashes, snapshot = _load_v4(args.v4_root)
        dataset_hash = str(snapshot.get("dataset", {}).get("sha256", ""))
        v3_rows = _load_v3_baseline(args.v3_root, dataset_hash)
        # Require identical validation sample identity/labels/position metadata across versions.
        v4_signatures = {row["validation_signature_sha256"] for row in v4_rows}
        v3_signatures = {row["validation_signature_sha256"] for row in v3_rows}
        if len(v3_signatures) != 1 or v4_signatures != v3_signatures:
            raise AuditError("v3 and v4 validation identities/labels do not match")
        summary = _aggregate(v4_rows, v3_rows, (args.v4_root, args.v3_root), source_hashes)
        _write_outputs(args.output_dir, summary)
        print(f"VALIDATION_SUMMARY_COMPLETE path={args.output_dir.resolve()}")
        print("verified_runs=20/20; test_metrics_computed=False")
        for variant in FORMAL_VARIANTS:
            item = summary["variants"][variant]["validation_macro_f1"]
            print(f"{variant} macro_f1={item['mean']:.6f} ± {item['std_population']:.6f}")
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
