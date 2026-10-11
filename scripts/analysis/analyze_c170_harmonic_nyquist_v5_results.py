"""One-off fail-closed aggregation of the fixed C170 v5 validation runs only."""

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

from scripts.analysis import analyze_c170_ring_cuda_v2_results as v2
from scripts.launchers import launch_c170_harmonic_nyquist_v5 as driver
VARIANTS = ("nyquist_coupled", "nyquist_uncoupled", "gram_control", "nyquist_shuffle")
SEEDS = (17, 29, 43, 61, 89)
POSITIONS = (5, 40, 90, 130)
DIRECTIONS = v2.DIRECTIONS
AuditError = v2.AuditError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read valid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def _exact_runs(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        try:
            key = (str(record["variant"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"invalid run mapping entry: {record!r}") from exc
        if key not in expected or key in found:
            raise AuditError(f"unexpected or duplicate v5 run identity: {key}")
        found[key] = record
    missing = sorted(expected - set(found))
    if missing:
        raise AuditError(f"v5 matrix is incomplete; missing {missing}")
    return [found[(variant, seed)] for variant in VARIANTS for seed in SEEDS]


def _resolve_runs(root: Path, mapping_path: Path | None) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    if mapping_path is None:
        markers = sorted(root.rglob("RUN_COMPLETE.json"))
        if len(markers) != 20:
            raise AuditError(f"expected exactly 20 completed v5 runs, found {len(markers)}")
        records = []
        for marker in markers:
            run_dir = marker.parent
            manifest = _read_json(run_dir / "run_manifest.json")
            config = manifest.get("config", {})
            records.append({
                "variant": config.get("variant"),
                "seed": config.get("seed"),
                "run_dir": str(run_dir),
            })
        return _exact_runs(records)

    mapping_path = mapping_path.expanduser().resolve()
    mapping = _read_json(mapping_path)
    entries = mapping.get("entries")
    if not isinstance(entries, list):
        raise AuditError(f"mapping entries must be a list: {mapping_path}")
    records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("run_dir"), str):
            raise AuditError(f"invalid mapped run entry in {mapping_path}")
        if entry.get("partial_checkpoint_reused") is True:
            raise AuditError(f"mapped run declares partial checkpoint reuse: {entry}")
        candidate = Path(entry["run_dir"])
        if not candidate.is_absolute():
            recovery_root = mapping.get("recovery_root")
            base = Path(recovery_root) if isinstance(recovery_root, str) else mapping_path.parent
            candidate = base / candidate
        records.append({**entry, "run_dir": str(candidate.expanduser().resolve())})
    return _exact_runs(records)


def _verify_snapshot_metadata(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Verify source snapshot copies without opening the dataset container."""
    snapshot_dir = root.expanduser().resolve() / "source_snapshot"
    snapshot = _read_json(snapshot_dir / "snapshot_manifest.json")
    if "v5" not in str(snapshot.get("format", "")).lower():
        raise AuditError("unexpected v5 source snapshot format")
    code_hashes = snapshot.get("provenance_code_sha256")
    if not isinstance(code_hashes, dict) or not code_hashes:
        raise AuditError("v5 snapshot has no source code hashes")
    expected_code_hashes = {name: _sha256(path) for name, path in driver.SOURCE_PATHS.items()}
    if code_hashes != expected_code_hashes:
        raise AuditError("v5 snapshot code hashes differ from frozen source files")
    copied_hashes = snapshot.get("snapshot_file_sha256")
    if not isinstance(copied_hashes, dict) or not set(driver.SNAPSHOT_DOC_COPIES).issubset(copied_hashes):
        raise AuditError("v5 source snapshot is missing copied-file hashes or design/plan")
    for relative, expected_hash in copied_hashes.items():
        physical = (snapshot_dir / driver.SNAPSHOT_DOC_COPIES[relative]
                    if relative in driver.SNAPSHOT_DOC_COPIES else snapshot_dir / "code" / relative)
        if not physical.is_file() or _sha256(physical) != expected_hash:
            raise AuditError(f"source snapshot file hash mismatch: {physical}")
    dataset = snapshot.get("dataset")
    dataset_hash = dataset.get("sha256") if isinstance(dataset, dict) else None
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise AuditError("v5 snapshot is missing a dataset SHA-256")
    return {str(k): str(v) for k, v in code_hashes.items()}, snapshot


def _verify_completed_run(run_dir: Path, variant: str, seed: int, dataset_hash: str,
                          code_hashes: dict[str, str]) -> dict[str, Any]:
    """Verify a completed bundle using config metadata and outputs, never dataset bytes."""
    manifest_path = run_dir / "run_manifest.json"
    marker_path = run_dir / "RUN_COMPLETE.json"
    manifest = _read_json(manifest_path)
    marker = _read_json(marker_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise AuditError(f"run config missing: {run_dir}")
    config_hash = hashlib.sha256(driver._canonical_json(config).encode("utf-8")).hexdigest()
    if manifest.get("config_sha256") != config_hash or marker.get("config_sha256") != config_hash:
        raise AuditError(f"config SHA-256 mismatch: {run_dir}")
    expected = {
        "variant": variant,
        "model_variant": "harmonic_nyquist_classifier_v5",
        "active_feature_indices": driver.ACTIVE_FEATURES[variant],
        "feature_names": driver.FEATURE_NAMES,
        "feature_definition": driver.FEATURE_DEFINITION,
        "shuffle_labels": variant == "nyquist_shuffle",
        "seed": seed,
        "max_epochs": driver.FORMAL_EPOCHS,
        "patience": driver.FORMAL_PATIENCE,
        "cpu_threads": driver.CPU_THREADS,
        "device": "cuda:0",
        "dataset_path": str(driver.DATASET_PATH.resolve()),
        "dataset_sha256": dataset_hash,
        "train_distance_mm": [0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160],
        "validation_distance_mm": list(POSITIONS),
        "registered_test_distance_mm": [15, 25, 110, 150],
        "excluded_distance_mm": [70],
        "expected_split_counts": {"train": 879, "validation": 320, "registered_test": 320},
        "batch_sizes_per_epoch": [44] * 19 + [43],
        "updates_per_epoch": driver.UPDATES_PER_EPOCH,
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
        "preprocessing_scale_required": driver.EXPECTED_SHARED_SCALE,
        "preprocessing_scale_fit_split": "train",
        "feature_lags": [1, 4, 16, 64],
        "lag_boundary": "zero_prefix_no_circular_wrap",
        "variant_feature_masks": driver.ACTIVE_FEATURES,
        "architecture": {
            "stem": "conv1d_17_to_24_k9_stride2_pad4_biasfalse_gn4_silu",
            "residual_dilations": [1, 2, 4],
            "residual_block": "depthwise24_k7_gn4_silu_pointwise24_gn4_residual_silu",
            "temporal_pooling": "adaptive_avg_lengths_1_2_4_plus_population_std_unbiased_false",
            "dropout": 0.2,
            "classifier": "linear_192_to_4",
            "trainable_parameters": 7012,
            "initialization": "same_torch_manual_seed_and_parameter_shapes_for_all_variants",
        },
        "checkpoint_selection": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
        "safe_state_dict_format": "numpy_npz_numeric_arrays",
        "safe_state_dict_reload": "numpy_load_allow_pickle_false_then_strict_state_dict",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
        "physical_order_confirmed_by_operator": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise AuditError(f"frozen protocol config mismatch for {key}: {run_dir}")
    if manifest.get("format") != "c170-harmonic-nyquist-v5-run" or manifest.get("status") != "completed":
        raise AuditError(f"runner manifest is not a completed v5 run: {run_dir}")
    if marker.get("format") != "c170-harmonic-nyquist-v5-complete" or marker.get("status") != "completed":
        raise AuditError(f"completion marker is invalid: {run_dir}")
    if manifest.get("test_metrics_computed") is not False or marker.get("test_metrics_computed") is not False:
        raise AuditError(f"test metrics are not explicitly disabled: {run_dir}")
    access = manifest.get("test_access")
    if not isinstance(access, dict) or access.get("loaded_as_part_of_complete_npz") is not True or any(
        access.get(key) is not False for key in ("used_as_model_input", "preprocessed", "used_to_estimate_statistics", "predicted_or_scored")
    ):
        raise AuditError(f"test-access boundary declaration is invalid: {run_dir}")
    if marker.get("manifest_sha256") != _sha256(manifest_path):
        raise AuditError(f"manifest SHA-256 mismatch: {run_dir}")
    provenance = manifest.get("provenance", {}).get("code_sha256")
    if provenance != code_hashes:
        raise AuditError(f"run source hashes differ from source snapshot: {run_dir}")
    split = manifest.get("split", {})
    if split.get("split_counts") != {"train": 879, "val": 320, "test": 320} or split.get("v5_variant") != variant:
        raise AuditError(f"split metadata differs from v5 protocol: {run_dir}")
    shuffled = variant == "nyquist_shuffle"
    if split.get("label_shuffle_seed") != (seed + 10_000 if shuffled else None):
        raise AuditError(f"label shuffle seed mismatch: {run_dir}")
    label_map = split.get("label_shuffle_mapping")
    if not isinstance(label_map, list) or (shuffled and not label_map) or (not shuffled and label_map):
        raise AuditError(f"label shuffle mapping mismatch: {run_dir}")
    prep = manifest.get("preprocessing", {})
    if (prep.get("normalization_fit_splits") != ["train"] or prep.get("processed_splits") != ["train", "validation"]
            or prep.get("test_preprocessed") is not False or prep.get("channel_map_as_model_input") is not False
            or float(prep.get("shared_sensor_scale", float("nan"))) != driver.EXPECTED_SHARED_SCALE):
        raise AuditError(f"preprocessing boundary or scale mismatch: {run_dir}")
    if marker.get("selected_epoch_one_based") != manifest.get("selection", {}).get("best_epoch_one_based"):
        raise AuditError(f"selected epoch mismatch: {run_dir}")

    artifact_names = {"best_checkpoint": "best_checkpoint.pt", "safe_state_dict": "best_weights.npz",
                      "learning_curve": "learning_curve.csv", "validation_predictions": "validation_predictions.csv"}
    artifacts = manifest.get("artifacts")
    marker_hashes = marker.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != set(artifact_names) or not isinstance(marker_hashes, dict) or set(marker_hashes) != set(artifact_names):
        raise AuditError(f"artifact list changed: {run_dir}")
    artifact_hashes: dict[str, str] = {}
    for name, filename in artifact_names.items():
        path = run_dir / filename
        digest = _sha256(path) if path.is_file() else ""
        record = artifacts.get(name, {})
        if not digest or record.get("path") != filename or record.get("sha256") != digest or marker_hashes.get(name) != digest:
            raise AuditError(f"artifact hash mismatch: {run_dir / filename}")
        artifact_hashes[name] = digest
        if name == "safe_state_dict":
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if not archive.files:
                        raise AuditError(f"safe NPZ is empty: {path}")
                    for tensor_name in archive.files:
                        array = archive[tensor_name]
                        if array.dtype.hasobject or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                            raise AuditError(f"invalid safe tensor {tensor_name}: {path}")
            except (OSError, ValueError) as exc:
                raise AuditError(f"cannot safely inspect NPZ weights: {path}: {exc}") from exc
    with (run_dir / "learning_curve.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        curve = list(csv.DictReader(stream))
    selection = manifest["selection"]
    if not curve or len(curve) != selection.get("actual_epochs"):
        raise AuditError(f"learning curve length mismatch: {run_dir}")
    for epoch, row in enumerate(curve, 1):
        if int(row["epoch"]) != epoch or int(row["optimizer_updates"]) != driver.UPDATES_PER_EPOCH:
            raise AuditError(f"learning-curve epoch/update sequence mismatch: {run_dir}")
    return {"config_sha256": config_hash, "artifact_sha256": artifact_hashes}


def _verify_runs(root: Path, mapping_path: Path | None) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    root = root.expanduser().resolve()
    code_hashes, snapshot = _verify_snapshot_metadata(root)

    records = _resolve_runs(root, mapping_path)
    verified: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    expected_dataset_hash = snapshot.get("dataset", {}).get("sha256")
    for record in records:
        run_dir = Path(record["run_dir"]).resolve()
        marker_path = run_dir / "RUN_COMPLETE.json"
        manifest_path = run_dir / "run_manifest.json"
        if not marker_path.is_file() or not manifest_path.is_file():
            raise AuditError(f"mapped run is incomplete: {run_dir}")
        manifest = _read_json(manifest_path)
        config = manifest.get("config")
        if not isinstance(config, dict):
            raise AuditError(f"run config missing: {run_dir}")
        variant = str(config.get("variant"))
        seed = int(config.get("seed", -1))
        if (variant, seed) != (str(record["variant"]), int(record["seed"])):
            raise AuditError(f"mapping identity differs from run config: {run_dir}")
        if config.get("dataset_sha256") != expected_dataset_hash:
            raise AuditError(f"dataset hash differs from formal source snapshot: {run_dir}")
        checked = _verify_completed_run(run_dir, variant, seed, str(expected_dataset_hash), code_hashes)
        mapped_hash = record.get("config_sha256")
        if mapped_hash is not None and mapped_hash != checked["config_sha256"]:
            raise AuditError(f"mapping config SHA-256 mismatch: {run_dir}")
        if manifest.get("provenance", {}).get("code_sha256") != code_hashes:
            raise AuditError(f"run source hashes differ from the formal source snapshot: {run_dir}")
        identity = (variant, seed)
        if identity in identities:
            raise AuditError(f"duplicate verified run: {identity}")
        identities.add(identity)
        verified.append({
            "variant": variant,
            "seed": seed,
            "run_dir": run_dir,
            "manifest": manifest,
            "checked": checked,
        })
    return _exact_runs(verified), {str(k): str(v) for k, v in code_hashes.items()}, snapshot


def _read_validation(run: dict[str, Any]) -> dict[str, Any]:
    manifest = run["manifest"]
    prediction_path = run["run_dir"] / manifest["artifacts"]["validation_predictions"]["path"]
    with prediction_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 320:
        raise AuditError(f"expected 320 validation rows: {prediction_path}")

    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    distances: list[int] = []
    signature: list[tuple[int, int, str, tuple[int, ...]]] = []
    seen: set[int] = set()
    for row in rows:
        if row.get("variant") != run["variant"] or int(row.get("seed", -1)) != run["seed"]:
            raise AuditError(f"prediction identity mismatch: {prediction_path}")
        sample = int(row["sample_index"])
        if sample in seen:
            raise AuditError(f"duplicate validation sample {sample}: {prediction_path}")
        seen.add(sample)
        distance = int(row["distance_mm"])
        truth_name = row["true_direction"]
        pred_name = row["prediction"]
        if distance not in POSITIONS or truth_name not in DIRECTIONS or pred_name not in DIRECTIONS:
            raise AuditError(f"invalid validation class or position: {prediction_path}")
        truth = DIRECTIONS.index(truth_name)
        pred = DIRECTIONS.index(pred_name)
        probs = [float(row[f"probability_{name.lower()}"]) for name in DIRECTIONS]
        if any(not math.isfinite(x) or x < 0.0 or x > 1.0 for x in probs):
            raise AuditError(f"invalid validation probability: {prediction_path}")
        if not math.isclose(sum(probs), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise AuditError(f"validation probabilities do not sum to one: {prediction_path}")
        if int(np.argmax(probs)) != pred:
            raise AuditError(f"prediction does not match validation probability argmax: {prediction_path}")
        channel_map = tuple(int(v) for v in row["channel_map"].split(","))
        if len(channel_map) != 4:
            raise AuditError(f"invalid channel-map metadata: {prediction_path}")
        labels.append(truth)
        predictions.append(pred)
        probabilities.append(probs)
        distances.append(distance)
        signature.append((sample, distance, truth_name, channel_map))

    overall = v2._classification_metrics(labels, predictions)
    by_position: dict[str, dict[str, Any]] = {}
    for position in POSITIONS:
        indices = [i for i, value in enumerate(distances) if value == position]
        if len(indices) != 80 or any(sum(labels[i] == action for i in indices) != 20 for action in range(4)):
            raise AuditError(f"validation position/class balance changed at {position}: {prediction_path}")
        expected_map = (0, 2, 4, 6) if position in (5, 40) else (0, 1, 2, 3)
        if {signature[i][3] for i in indices} != {expected_map}:
            raise AuditError(f"validation channel-map metadata changed at {position}: {prediction_path}")
        by_position[str(position)] = v2._classification_metrics(
            [labels[i] for i in indices], [predictions[i] for i in indices]
        )
    ce = sum(-math.log(max(probabilities[i][labels[i]], 1e-300)) for i in range(320)) / 320.0

    selection = manifest.get("selection", {})
    recorded = selection.get("validation_metrics", {}).get("overall", {})
    if recorded.get("confusion_matrix") != overall["confusion_matrix"]:
        raise AuditError(f"manifest validation confusion disagrees with CSV: {prediction_path}")
    if not math.isclose(float(recorded.get("macro_f1", float("nan"))), overall["macro_f1"], rel_tol=0, abs_tol=1e-10):
        raise AuditError(f"manifest validation F1 disagrees with CSV: {prediction_path}")
    recorded_recalls = recorded.get("per_class_recall", [])
    if len(recorded_recalls) != len(DIRECTIONS) or any(
        not math.isclose(float(a), b, rel_tol=0, abs_tol=1e-10)
        for a, b in zip(recorded_recalls, overall["per_class_recall"])
    ):
        raise AuditError(f"manifest validation recall disagrees with CSV: {prediction_path}")
    if not math.isclose(float(selection.get("validation_cross_entropy", float("nan"))), ce, rel_tol=0, abs_tol=1e-5):
        raise AuditError(f"manifest validation CE disagrees with CSV: {prediction_path}")
    for position, measured in by_position.items():
        recorded_position = selection["validation_metrics"]["by_position_mm"].get(position, {})
        if recorded_position.get("confusion_matrix") != measured["confusion_matrix"]:
            raise AuditError(f"manifest position confusion disagrees at {position}: {prediction_path}")
        if not math.isclose(float(recorded_position.get("macro_f1", float("nan"))), measured["macro_f1"], rel_tol=0, abs_tol=1e-10):
            raise AuditError(f"manifest position F1 disagrees at {position}: {prediction_path}")

    train_overall = manifest.get("selected_checkpoint_train_evaluation", {}).get("overall", {})
    if train_overall.get("count") != 879:
        raise AuditError(f"expected 879 selected-checkpoint training targets: {prediction_path}")
    train_f1 = float(train_overall["macro_f1"])
    if not math.isfinite(train_f1):
        raise AuditError(f"non-finite selected-checkpoint train-target F1: {prediction_path}")
    identity = sorted(signature)
    sig_hash = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "variant": run["variant"], "seed": run["seed"], "run_dir": str(run["run_dir"]),
        "config_sha256": manifest["config_sha256"],
        "validation_macro_f1": overall["macro_f1"],
        "validation_cross_entropy": ce,
        "validation_per_class_recall": dict(zip(DIRECTIONS, overall["per_class_recall"])),
        "validation_prediction_unique_class_count": overall["prediction_unique_class_count"],
        "validation_by_position": by_position,
        "validation_signature_sha256": sig_hash,
        "selected_checkpoint_train_target_macro_f1": train_f1,
        "train_target_minus_validation_macro_f1": train_f1 - overall["macro_f1"],
        "selected_epoch_one_based": int(selection["best_epoch_one_based"]),
        "actual_epochs": int(selection["actual_epochs"]),
        "optimizer_updates": int(selection["optimizer_updates"]),
        "validation_signature": identity,
    }


def _stats(values: Sequence[float]) -> dict[str, Any]:
    return v2._stats(values)


def _read_v4_baseline(path: Path, signature: str) -> list[dict[str, Any]]:
    summary = _read_json(path)
    if (summary.get("format") != "c170-harmonic-phase-v4-validation-summary"
            or summary.get("validation_only") is not True
            or summary.get("test_metrics_computed") is not False
            or summary.get("verified_run_count") != 20):
        raise AuditError(f"v4 baseline summary is not a completed validation-only 20-run summary: {path}")
    rows = [row for row in summary.get("per_seed", []) if row.get("variant") == "harmonic_gram"]
    if len(rows) != 5 or {int(row["seed"]) for row in rows} != set(SEEDS):
        raise AuditError(f"v4 harmonic_gram baseline does not contain five seeds: {path}")
    if any(row.get("validation_signature_sha256") != signature for row in rows):
        raise AuditError("v4 and v5 validation sample identities/labels differ")
    return [
        {
            "variant": "harmonic_gram",
            "seed": int(row["seed"]),
            "validation_macro_f1": float(row["validation_macro_f1"]),
            "validation_signature_sha256": row["validation_signature_sha256"],
        }
        for row in rows
    ]


def _aggregate(rows: list[dict[str, Any]], root: Path, code_hashes: dict[str, str], snapshot: dict[str, Any], mapping: Path | None,
               v4_baseline: list[dict[str, Any]] | None) -> dict[str, Any]:
    signatures = {row["validation_signature_sha256"] for row in rows}
    if len(signatures) != 1:
        raise AuditError("validation identities/labels differ across v5 runs")
    by_variant_seed = {variant: {row["seed"]: row for row in rows if row["variant"] == variant} for variant in VARIANTS}
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [by_variant_seed[variant][seed] for seed in SEEDS]
        positions = {}
        for position in map(str, POSITIONS):
            cells = [row["validation_by_position"][position] for row in selected]
            positions[position] = {
                "macro_f1": _stats([cell["macro_f1"] for cell in cells]),
                "per_class_recall": {
                    direction: _stats([cell["per_class_recall"][idx] for cell in cells])
                    for idx, direction in enumerate(DIRECTIONS)
                },
            }
        variants[variant] = {
            "validation_macro_f1": _stats([row["validation_macro_f1"] for row in selected]),
            "validation_cross_entropy": _stats([row["validation_cross_entropy"] for row in selected]),
            "validation_per_class_recall": {
                direction: _stats([row["validation_per_class_recall"][direction] for row in selected])
                for direction in DIRECTIONS
            },
            "validation_prediction_unique_class_count": _stats([row["validation_prediction_unique_class_count"] for row in selected]),
            "selected_checkpoint_train_target_macro_f1": _stats([row["selected_checkpoint_train_target_macro_f1"] for row in selected]),
            "train_target_minus_validation_macro_f1": _stats([row["train_target_minus_validation_macro_f1"] for row in selected]),
            "selected_epoch_one_based_by_seed": {str(row["seed"]): row["selected_epoch_one_based"] for row in selected},
            "validation_by_position": positions,
        }
    comparisons: dict[str, dict[int, float]] = {
        "nyquist_coupled_minus_nyquist_uncoupled": {
            seed: by_variant_seed["nyquist_coupled"][seed]["validation_macro_f1"] - by_variant_seed["nyquist_uncoupled"][seed]["validation_macro_f1"] for seed in SEEDS
        },
        "nyquist_coupled_minus_gram_control": {
            seed: by_variant_seed["nyquist_coupled"][seed]["validation_macro_f1"] - by_variant_seed["gram_control"][seed]["validation_macro_f1"] for seed in SEEDS
        },
        "nyquist_coupled_minus_nyquist_shuffle": {
            seed: by_variant_seed["nyquist_coupled"][seed]["validation_macro_f1"] - by_variant_seed["nyquist_shuffle"][seed]["validation_macro_f1"] for seed in SEEDS
        },
    }
    if v4_baseline is not None:
        v4_by_seed = {int(row["seed"]): row for row in v4_baseline}
        for variant in ("nyquist_coupled", "gram_control"):
            comparisons[f"{variant}_minus_v4_harmonic_gram"] = {
                seed: by_variant_seed[variant][seed]["validation_macro_f1"] - float(v4_by_seed[seed]["validation_macro_f1"])
                for seed in SEEDS
            }
    paired = {
        name: {"per_seed": {str(seed): value for seed, value in differences.items()}, **_stats(list(differences.values()))}
        for name, differences in comparisons.items()
    }
    return {
        "format": "c170-harmonic-nyquist-v5-validation-summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "validation_only": True,
        "test_metrics_computed": False,
        "test_access": {"loaded_as_part_of_complete_npz": True, "preprocessed": False, "used_to_estimate_statistics": False, "used_as_model_input": False, "predicted_or_scored": False},
        "expected_run_count": 20,
        "verified_run_count": len(rows),
        "standard_deviation": "population standard deviation (ddof=0)",
        "v5_formal_root": str(root.resolve()),
        "run_mapping": str(mapping.resolve()) if mapping else None,
        "source_snapshot_code_sha256": code_hashes,
        "dataset_sha256": snapshot.get("dataset", {}).get("sha256"),
        "variants": variants,
        "paired_seed_macro_f1_differences": paired,
        "per_seed": [{key: value for key, value in row.items() if key != "validation_signature"} for row in rows],
        "v4_harmonic_gram_baseline_per_seed": v4_baseline,
        "interpretation_boundary": (
            "All recomputed scores use only the 320-row validation prediction CSVs; selected-checkpoint training F1 is read from each run manifest. "
            "Per-seed differences are paired by the fixed seed. The complete compressed NPZ may be decoded and integrity-checked by the training runner, "
            "but test waveforms were not preprocessed, used to fit statistics, supplied to a model, predicted, or scored. "
            "The coupled Nyquist feature tests a discrete C4-restricted phase tradeoff, not continuous physical SO(2) wrist invariance. "
            "The shuffle-arm training score is against permuted training targets; it is not a clean-label train score."
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# C170 Harmonic-Nyquist v5 Validation Summary", "",
        f"Verified runs: {summary['verified_run_count']}/{summary['expected_run_count']}. Test metrics computed: false.",
        "Mean ± population SD across five seeds; all reported model scores are validation-only.", "",
        "| Variant | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Validation CE | Selected train-target F1 | Train-target F1 gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = summary["variants"][variant]
        fields = [item["validation_macro_f1"], *(item["validation_per_class_recall"][d] for d in DIRECTIONS),
                  item["validation_cross_entropy"], item["selected_checkpoint_train_target_macro_f1"], item["train_target_minus_validation_macro_f1"]]
        cells = [f"{fields[0]['mean']:.4f} ± {fields[0]['std_population']:.4f}"]
        cells += [f"{x['mean']:.4f} ± {x['std_population']:.4f}" for x in fields[1:6]]
        cells += [f"{x['mean']:.4f} ± {x['std_population']:.4f}" for x in fields[6:]]
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Validation Macro-F1 by Position", "", "| Position (mm) | Coupled | Uncoupled | Gram control | Shuffle |", "|---:|---:|---:|---:|---:|"])
    for position in map(str, POSITIONS):
        values = [summary["variants"][variant]["validation_by_position"][position]["macro_f1"] for variant in VARIANTS]
        lines.append(f"| {position} | " + " | ".join(f"{x['mean']:.4f} ± {x['std_population']:.4f}" for x in values) + " |")
    lines.extend(["", "## Paired Seed Differences", "", "Differences are first arm minus comparator; mean ± population SD.", ""])
    for name, item in summary["paired_seed_macro_f1_differences"].items():
        detail = ", ".join(f"{seed}:{value:+.4f}" for seed, value in item["per_seed"].items())
        lines.append(f"- {name}: {item['mean']:+.4f} ± {item['std_population']:.4f}; {detail}.")
    lines.extend(["", summary["interpretation_boundary"], ""])
    return "\n".join(lines)


def _write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "validation_summary.md").write_text(_markdown(summary), encoding="utf-8")
    fields = ["variant", "seed", "validation_macro_f1", "validation_cross_entropy", "selected_checkpoint_train_target_macro_f1",
              "train_target_minus_validation_macro_f1", "selected_epoch_one_based", "actual_epochs", "optimizer_updates", "run_dir"]
    fields += [f"validation_recall_{d.lower()}" for d in DIRECTIONS]
    fields += [f"validation_f1_at_{p}mm" for p in POSITIONS]
    with (output_dir / "per_seed_validation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_seed"]:
            output = {key: row[key] for key in fields if key in row}
            output.update({f"validation_recall_{d.lower()}": row["validation_per_class_recall"][d] for d in DIRECTIONS})
            output.update({f"validation_f1_at_{p}mm": row["validation_by_position"][str(p)]["macro_f1"] for p in POSITIONS})
            writer.writerow(output)
    with (output_dir / "paired_seed_differences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("comparison", "seed", "macro_f1_difference"))
        writer.writeheader()
        for name, item in summary["paired_seed_macro_f1_differences"].items():
            for seed, difference in item["per_seed"].items():
                writer.writerow({"comparison": name, "seed": seed, "macro_f1_difference": difference})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True, help="Original formal root containing source_snapshot")
    parser.add_argument("--run-mapping", type=Path, help="Optional recovery mapping; exactly 20 mapped completed runs are required")
    parser.add_argument("--v4-summary", type=Path, default=Path(__file__).resolve().parent / "outputs" / "c170_harmonic_phase_v4_validation_summary_20260929" / "validation_summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        runs, code_hashes, snapshot = _verify_runs(args.formal_root, args.run_mapping)
        rows = [_read_validation(run) for run in runs]
        signature = rows[0]["validation_signature_sha256"]
        baseline = _read_v4_baseline(args.v4_summary, signature) if args.v4_summary.is_file() else None
        summary = _aggregate(rows, args.formal_root, code_hashes, snapshot, args.run_mapping, baseline)
        _write_outputs(args.output_dir, summary)
        print(f"VALIDATION_SUMMARY_COMPLETE path={args.output_dir.expanduser().resolve()}")
        print("verified_runs=20/20; test_metrics_computed=False")
        for variant in VARIANTS:
            score = summary["variants"][variant]["validation_macro_f1"]
            print(f"{variant} macro_f1={score['mean']:.6f} ± {score['std_population']:.6f}")
        for name, comparison in summary["paired_seed_macro_f1_differences"].items():
            print(f"{name} paired_mean={comparison['mean']:+.6f} ± {comparison['std_population']:.6f}")
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
