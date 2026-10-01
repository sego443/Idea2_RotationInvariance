"""Verify and summarize the frozen C170 latent-angle validation runs.

This standard-library-only tool reads completed run manifests, checkpoints'
checksums, learning curves and validation predictions. It never opens the
dataset or computes a test-set metric, and it fails closed unless all 25
formal variant/seed pairs are present exactly once across the supplied roots.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parent
FORMAL_VARIANTS = ("mlp", "no_angle", "full", "no_rms", "shuffle")
FORMAL_SEEDS = (17, 29, 43, 61, 89)
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
TRAIN_POSITIONS = (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)
VAL_POSITIONS = (5, 40, 90, 130)
TEST_POSITIONS = (15, 25, 110, 150)
EXCLUDED_POSITIONS = (70,)
EXPECTED_CODE_KEYS = (
    "runner",
    "model",
    "dataset_helpers",
    "preprocessing_helpers",
    "encoder_helpers",
)
CURVE_FIELDS = (
    "epoch_seconds",
    "train_action_loss",
    "train_angle_nll_unweighted",
    "train_angle_objective_contribution",
    "train_smoothness_loss",
    "train_total_data_loss",
    "train_total_objective",
    "val_cross_entropy",
    "val_macro_f1",
    "val_recall_up",
    "val_recall_down",
    "val_recall_left",
    "val_recall_right",
    "val_unique_predicted_classes",
    "val_angle_entropy",
    "val_angle_error_up_degrees",
    "val_angle_error_down_degrees",
    "val_angle_error_left_degrees",
    "val_angle_error_right_degrees",
)


class AuditError(RuntimeError):
    """Raised when an experiment bundle is incomplete or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - message aids recovery
        raise AuditError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object in {path}")
    return value


def _canonical_config_sha(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_markers(roots: Iterable[Path]) -> list[Path]:
    markers: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise AuditError(f"root does not exist or is not a directory: {root}")
        direct = root / "RUN_COMPLETE.json"
        if direct.is_file():
            markers.add(direct)
        markers.update(path.resolve() for path in root.glob("*/RUN_COMPLETE.json"))
    return sorted(markers)


def _read_snapshot(roots: Iterable[Path]) -> tuple[Path, dict[str, str]]:
    for root in roots:
        snapshot_dir = root.expanduser().resolve() / "source_snapshot"
        manifest_path = snapshot_dir / "snapshot_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path)
        files = manifest.get("source_files")
        if not isinstance(files, dict):
            raise AuditError(f"source_files missing from {manifest_path}")
        hashes: dict[str, str] = {}
        for key in EXPECTED_CODE_KEYS:
            entry = files.get(key)
            if not isinstance(entry, dict) or not entry.get("sha256") or not entry.get("path"):
                raise AuditError(f"source snapshot lacks required source {key}")
            saved_path = snapshot_dir / str(entry["path"])
            if not saved_path.is_file() or _sha256(saved_path) != entry["sha256"]:
                raise AuditError(f"source snapshot checksum mismatch for {key}: {saved_path}")
            hashes[key] = str(entry["sha256"])
        return snapshot_dir, hashes
    raise AuditError("no verified source_snapshot/snapshot_manifest.json found in supplied roots")


def _verify_completed_run(marker_path: Path, expected_source_hashes: dict[str, str]) -> dict[str, Any]:
    run_dir = marker_path.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise AuditError(f"completion marker has no run manifest: {run_dir}")
    marker = _read_json(marker_path)
    manifest = _read_json(manifest_path)
    if marker.get("format") != "c170-latent-angle-complete" or marker.get("status") != "completed":
        raise AuditError(f"invalid completion marker: {marker_path}")
    if manifest.get("format") != "c170-latent-angle-run" or manifest.get("status") != "completed":
        raise AuditError(f"run manifest is not complete: {manifest_path}")
    if manifest.get("test_metrics_computed") is not False or marker.get("test_metrics_computed") is not False:
        raise AuditError(f"run reports test metrics were computed: {run_dir}")
    if marker.get("manifest_sha256") != _sha256(manifest_path):
        raise AuditError(f"completion marker manifest hash mismatch: {run_dir}")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise AuditError(f"run config is missing: {run_dir}")
    config_sha = _canonical_config_sha(config)
    if config_sha != manifest.get("config_sha256") or config_sha != marker.get("config_sha256"):
        raise AuditError(f"canonical config hash mismatch: {run_dir}")
    variant = str(config.get("variant"))
    try:
        seed = int(config["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"invalid seed in {run_dir}") from exc
    if variant not in FORMAL_VARIANTS or seed not in FORMAL_SEEDS:
        raise AuditError(f"nonformal variant/seed found at {run_dir}: {variant}/{seed}")
    if run_dir.name != f"{variant}__seed-{seed}":
        raise AuditError(f"run directory does not match manifest identity: {run_dir}")

    expected_config = {
        "max_epochs": 160,
        "patience": 25,
        "device": "cuda",
        "updates_per_epoch": 20,
        "num_angles": 64,
        "train_distance_mm": list(TRAIN_POSITIONS),
        "val_distance_mm": list(VAL_POSITIONS),
        "registered_test_distance_mm": list(TEST_POSITIONS),
        "excluded_distance_mm": list(EXCLUDED_POSITIONS),
        "sample_weight_rule": "position_voronoi_weight/(4*original_group_size)",
        "sampler": "original_distance_label_groups_no_replacement_19x44_plus_43",
        "sampler_group_key": "distance_mm_and_original_direction_label",
        "sampler_rng": "numpy_seedsequence_seed_epoch",
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise AuditError(f"formal config mismatch for {name} in {run_dir}")
    if bool(config.get("include_rms")) != (variant != "no_rms"):
        raise AuditError(f"RMS inclusion mismatch in {run_dir}")
    if config.get("label_shuffle_seed") != (seed + 10_000 if variant == "shuffle" else None):
        raise AuditError(f"shuffle seed mismatch in {run_dir}")

    provenance = manifest.get("provenance", {}).get("code_sha256")
    if provenance != expected_source_hashes:
        raise AuditError(f"source provenance mismatch in {run_dir}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise AuditError(f"artifact hashes are missing in {run_dir}")
    required_artifacts = {"checkpoint", "learning_curve", "validation_predictions"}
    if set(artifacts) != required_artifacts:
        raise AuditError(f"unexpected artifact set in {run_dir}: {sorted(artifacts)}")
    artifact_hashes: dict[str, str] = {}
    for name, item in artifacts.items():
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise AuditError(f"incomplete artifact record {name} in {run_dir}")
        artifact_path = run_dir / str(item["path"])
        if not artifact_path.is_file() or _sha256(artifact_path) != item["sha256"]:
            raise AuditError(f"artifact hash mismatch: {artifact_path}")
        artifact_hashes[name] = str(item["sha256"])

    return {
        "run_id": run_dir.name,
        "run_dir": run_dir,
        "manifest": manifest,
        "marker": marker,
        "config": config,
        "manifest_sha256": _sha256(manifest_path),
        "config_sha256": config_sha,
        "marker_sha256": _sha256(marker_path),
        "artifact_sha256": artifact_hashes,
        "variant": variant,
        "seed": seed,
    }


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        raise AuditError(f"nonfinite metric found: {value!r}")
    return parsed


def _stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return {"n": 0, "mean": None, "std_population": None}
    return {
        "n": len(numeric),
        "mean": statistics.fmean(numeric),
        "std_population": statistics.pstdev(numeric),
    }


def _metrics_from_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    matrix = [[0 for _ in DIRECTIONS] for _ in DIRECTIONS]
    by_position: dict[str, list[dict[str, str]]] = {}
    sample_ids: set[str] = set()
    prediction_counts = {direction: 0 for direction in DIRECTIONS}
    for row in rows:
        if row.get("split") != "val":
            raise AuditError("validation prediction artifact contains a non-validation row")
        truth, prediction = row.get("truth"), row.get("prediction")
        if truth not in DIRECTIONS or prediction not in DIRECTIONS:
            raise AuditError(f"unknown direction in validation predictions: {truth!r}/{prediction!r}")
        distance = str(int(row["distance_mm"]))
        if int(distance) not in VAL_POSITIONS:
            raise AuditError(f"unexpected validation distance in predictions: {distance}")
        sample_id = row.get("sample_index", "")
        if sample_id in sample_ids:
            raise AuditError(f"duplicate validation sample id: {sample_id}")
        sample_ids.add(sample_id)
        truth_idx, prediction_idx = DIRECTIONS.index(truth), DIRECTIONS.index(prediction)
        matrix[truth_idx][prediction_idx] += 1
        prediction_counts[prediction] += 1
        by_position.setdefault(distance, []).append(row)
    if len(rows) != 320 or len(sample_ids) != 320:
        raise AuditError(f"expected 320 unique validation predictions, found {len(rows)}")
    expected_position_keys = {str(position) for position in VAL_POSITIONS}
    if set(by_position) != expected_position_keys or any(len(group) != 80 for group in by_position.values()):
        raise AuditError("validation predictions must contain exactly 80 rows at each registered validation position")

    per_class_recall = []
    per_class_f1 = []
    for index in range(len(DIRECTIONS)):
        tp = matrix[index][index]
        actual = sum(matrix[index])
        predicted = sum(matrix[row][index] for row in range(len(DIRECTIONS)))
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_recall.append(recall)
        per_class_f1.append(f1)
    overall = {
        "count": len(rows),
        "accuracy": sum(matrix[i][i] for i in range(len(DIRECTIONS))) / len(rows),
        "macro_f1": statistics.fmean(per_class_f1),
        "per_class_recall": per_class_recall,
        "confusion_matrix": matrix,
        "prediction_counts": [prediction_counts[d] for d in DIRECTIONS],
        "prediction_unique_class_count": sum(v > 0 for v in prediction_counts.values()),
    }
    positions = {
        key: _metrics_from_subset(subset)
        for key, subset in sorted(by_position.items(), key=lambda item: int(item[0]))
    }
    return {"overall": overall, "by_position": positions, "sample_ids": sample_ids}


def _metrics_from_subset(rows: list[dict[str, str]]) -> dict[str, Any]:
    matrix = [[0 for _ in DIRECTIONS] for _ in DIRECTIONS]
    for row in rows:
        matrix[DIRECTIONS.index(row["truth"])][DIRECTIONS.index(row["prediction"])] += 1
    f1s, recalls = [], []
    for index in range(len(DIRECTIONS)):
        tp = matrix[index][index]
        actual = sum(matrix[index])
        predicted = sum(matrix[j][index] for j in range(len(DIRECTIONS)))
        recall = tp / actual if actual else 0.0
        precision = tp / predicted if predicted else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        recalls.append(recall)
    return {
        "count": len(rows),
        "macro_f1": statistics.fmean(f1s),
        "per_class_recall": recalls,
        "confusion_matrix": matrix,
    }


def _load_curve(run: dict[str, Any]) -> list[dict[str, Any]]:
    path = run["run_dir"] / str(run["manifest"]["artifacts"]["learning_curve"]["path"])
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AuditError(f"empty learning curve: {path}")
    parsed: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(rows, start=1):
        if int(row.get("epoch", -1)) != expected_epoch:
            raise AuditError(f"nonconsecutive epochs in {path}")
        if int(row.get("optimizer_updates", -1)) != 20:
            raise AuditError(f"unexpected update count in {path} epoch {expected_epoch}")
        parsed_row: dict[str, Any] = {"epoch": expected_epoch}
        for field in CURVE_FIELDS:
            parsed_row[field] = _number(row.get(field))
        parsed.append(parsed_row)
    selection = run["manifest"].get("selection", {})
    best_epoch = selection.get("best_epoch_zero_based")
    if not isinstance(best_epoch, int) or best_epoch < 0 or best_epoch >= len(parsed):
        raise AuditError(f"invalid selected epoch in {run['run_id']}")
    selected_row = parsed[best_epoch]
    if not math.isclose(
        float(selected_row["val_macro_f1"]),
        float(selection["validation_macro_f1"]),
        rel_tol=0,
        abs_tol=1e-10,
    ):
        raise AuditError(f"selected macro-F1 does not match curve in {run['run_id']}")
    if not math.isclose(
        float(selected_row["val_cross_entropy"]),
        float(selection["validation_cross_entropy"]),
        rel_tol=0,
        abs_tol=1e-6,
    ):
        raise AuditError(f"selected cross-entropy does not match curve in {run['run_id']}")
    best_row = max(parsed, key=lambda row: (float(row["val_macro_f1"]), -float(row["val_cross_entropy"])))
    if int(best_row["epoch"]) - 1 != best_epoch:
        raise AuditError(f"checkpoint selection is not the best validation epoch in {run['run_id']}")
    return parsed


def _analyze_run(run: dict[str, Any]) -> dict[str, Any]:
    manifest = run["manifest"]
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise AuditError(f"selection metrics missing in {run['run_id']}")
    predictions_path = run["run_dir"] / str(manifest["artifacts"]["validation_predictions"]["path"])
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as stream:
        prediction_rows = list(csv.DictReader(stream))
    measured = _metrics_from_rows(prediction_rows)
    recorded = selection.get("validation_overall") or {}
    if measured["overall"]["confusion_matrix"] != recorded.get("confusion_matrix"):
        raise AuditError(f"prediction confusion matrix differs from manifest in {run['run_id']}")
    if not math.isclose(
        measured["overall"]["macro_f1"],
        float(selection["validation_macro_f1"]),
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise AuditError(f"prediction macro-F1 differs from manifest in {run['run_id']}")
    if measured["overall"]["prediction_counts"] != recorded.get("prediction_counts"):
        raise AuditError(f"prediction diversity counts differ from manifest in {run['run_id']}")
    for index, direction in enumerate(DIRECTIONS):
        if not math.isclose(
            measured["overall"]["per_class_recall"][index],
            float(recorded["per_class_recall"][index]),
            rel_tol=0,
            abs_tol=1e-8,
        ):
            raise AuditError(f"prediction recall differs from manifest in {run['run_id']} for {direction}")

    recorded_positions = selection.get("validation_by_distance_mm") or {}
    if set(recorded_positions) != set(measured["by_position"]):
        raise AuditError(f"validation positions differ from manifest in {run['run_id']}")
    for position, metrics in measured["by_position"].items():
        reference = recorded_positions[position]
        if not math.isclose(metrics["macro_f1"], float(reference["macro_f1"]), rel_tol=0, abs_tol=1e-8):
            raise AuditError(f"position macro-F1 differs from manifest in {run['run_id']} at {position} mm")

    angle_diag = manifest.get("diagnostics", {}).get("integration_resolution_64_vs_128")
    if run["variant"] == "full":
        if not isinstance(angle_diag, dict) or angle_diag.get("f1_comparison_used_for_selection") is not False:
            raise AuditError(f"full-model 64/128 stability diagnostic missing or mis-scoped in {run['run_id']}")
    elif angle_diag is not None:
        raise AuditError(f"unexpected 64/128 diagnostic in non-full run {run['run_id']}")

    shuffle_mapping = None
    if run["variant"] == "shuffle":
        shuffle_mapping = manifest.get("metadata", {}).get("label_shuffle_mapping")
        if not isinstance(shuffle_mapping, list) or len(shuffle_mapping) != 44:
            raise AuditError(f"shuffle mapping must contain 44 training blocks in {run['run_id']}")
        seen_blocks: set[int] = set()
        for distance in TRAIN_POSITIONS:
            rows_at_position = [item for item in shuffle_mapping if int(item["distance_mm"]) == distance]
            if len(rows_at_position) != 4:
                raise AuditError(f"shuffle mapping does not contain four blocks at {distance} mm")
            if sorted(item["original_direction"] for item in rows_at_position) != sorted(DIRECTIONS):
                raise AuditError(f"shuffle mapping changed original class balance at {distance} mm")
            if sorted(item["training_target_direction"] for item in rows_at_position) != sorted(DIRECTIONS):
                raise AuditError(f"shuffle mapping did not preserve target balance at {distance} mm")
            for item in rows_at_position:
                block = int(item["block"])
                if block in seen_blocks:
                    raise AuditError(f"duplicate training block in shuffle mapping: {block}")
                seen_blocks.add(block)

    curves = _load_curve(run)
    return {
        "run_id": run["run_id"],
        "variant": run["variant"],
        "seed": run["seed"],
        "run_dir": str(run["run_dir"]),
        "selected_epoch_zero_based": selection["best_epoch_zero_based"],
        "selected_epoch_one_based": int(selection["best_epoch_zero_based"]) + 1,
        "validation_macro_f1": measured["overall"]["macro_f1"],
        "validation_cross_entropy": float(selection["validation_cross_entropy"]),
        "per_class_recall": dict(zip(DIRECTIONS, measured["overall"]["per_class_recall"])),
        "confusion_matrix": measured["overall"]["confusion_matrix"],
        "prediction_counts": dict(zip(DIRECTIONS, measured["overall"]["prediction_counts"])),
        "prediction_unique_class_count": measured["overall"]["prediction_unique_class_count"],
        "by_position": measured["by_position"],
        "by_position_and_channel_map": selection.get("validation_by_distance_mm_and_channel_map"),
        "shuffle_changed_block_count": (
            sum(item["original_direction"] != item["training_target_direction"] for item in shuffle_mapping)
            if shuffle_mapping is not None
            else None
        ),
        "angle_resolution_diagnostic": angle_diag,
        "learning_curve": curves,
        "_validation_sample_ids": sorted(measured["sample_ids"]),
        "hashes": {
            "run_manifest_sha256": run["manifest_sha256"],
            "config_sha256": run["config_sha256"],
            "completion_marker_sha256": run["marker_sha256"],
            "artifacts": run["artifact_sha256"],
        },
    }


def _aggregate(per_seed: list[dict[str, Any]], snapshot_hashes: dict[str, str]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {v: [] for v in FORMAL_VARIANTS}
    for row in per_seed:
        by_variant[row["variant"]].append(row)
    for variant in FORMAL_VARIANTS:
        by_variant[variant].sort(key=lambda row: row["seed"])

    variant_summary: dict[str, Any] = {}
    curve_summary: dict[str, Any] = {}
    curve_rows_for_csv: list[dict[str, Any]] = []
    for variant, rows in by_variant.items():
        if len(rows) != len(FORMAL_SEEDS):
            raise AuditError(f"expected five seeds for {variant}, found {len(rows)}")
        recalls = {
            direction: _stats(row["per_class_recall"][direction] for row in rows)
            for direction in DIRECTIONS
        }
        confusion_sum = [
            [sum(row["confusion_matrix"][i][j] for row in rows) for j in range(4)]
            for i in range(4)
        ]
        position_summary: dict[str, Any] = {}
        for position in map(str, VAL_POSITIONS):
            position_rows = [row["by_position"][position] for row in rows]
            position_summary[position] = {
                "count_per_seed": [item["count"] for item in position_rows],
                "macro_f1": _stats(item["macro_f1"] for item in position_rows),
                "per_class_recall": {
                    direction: _stats(item["per_class_recall"][i] for item in position_rows)
                    for i, direction in enumerate(DIRECTIONS)
                },
            }

        variant_summary[variant] = {
            "seed_count": len(rows),
            "validation_macro_f1": _stats(row["validation_macro_f1"] for row in rows),
            "validation_cross_entropy": _stats(row["validation_cross_entropy"] for row in rows),
            "per_class_recall": recalls,
            "selected_epoch_one_based": [row["selected_epoch_one_based"] for row in rows],
            "prediction_unique_class_count": _stats(row["prediction_unique_class_count"] for row in rows),
            "prediction_counts_per_class": {
                direction: _stats(row["prediction_counts"][direction] for row in rows)
                for direction in DIRECTIONS
            },
            "shuffle_changed_block_count": (
                [row["shuffle_changed_block_count"] for row in rows]
                if variant == "shuffle"
                else None
            ),
            "confusion_matrix_sum_across_seeds": confusion_sum,
            "confusion_matrix_mean_per_seed": [
                [confusion_sum[i][j] / len(rows) for j in range(4)] for i in range(4)
            ],
            "by_position": position_summary,
            "per_seed": rows,
        }

        max_epoch = max(len(row["learning_curve"]) for row in rows)
        curve_summary[variant] = []
        for epoch in range(1, max_epoch + 1):
            epoch_rows = [
                row["learning_curve"][epoch - 1]
                for row in rows
                if len(row["learning_curve"]) >= epoch
            ]
            aggregate = {"epoch": epoch, "seed_count": len(epoch_rows)}
            csv_row: dict[str, Any] = {"variant": variant, "epoch": epoch, "seed_count": len(epoch_rows)}
            for field in CURVE_FIELDS:
                stats = _stats(item[field] for item in epoch_rows)
                aggregate[field] = stats
                csv_row[f"{field}_mean"] = stats["mean"]
                csv_row[f"{field}_std_population"] = stats["std_population"]
            curve_summary[variant].append(aggregate)
            curve_rows_for_csv.append(csv_row)

    paired: dict[str, Any] = {}
    full_by_seed = {row["seed"]: row["validation_macro_f1"] for row in by_variant["full"]}
    for comparator in ("mlp", "no_angle", "no_rms", "shuffle"):
        comparator_by_seed = {row["seed"]: row["validation_macro_f1"] for row in by_variant[comparator]}
        diffs = [full_by_seed[seed] - comparator_by_seed[seed] for seed in FORMAL_SEEDS]
        paired["full_minus_" + comparator] = {
            "per_seed": {str(seed): full_by_seed[seed] - comparator_by_seed[seed] for seed in FORMAL_SEEDS},
            "difference": _stats(diffs),
        }

    full = variant_summary["full"]
    recall_means = {direction: full["per_class_recall"][direction]["mean"] for direction in DIRECTIONS}
    gate_macro = float(full["validation_macro_f1"]["mean"]) >= 0.80
    gate_classes = all(float(recall_means[direction]) >= 0.65 for direction in DIRECTIONS)
    gate = {
        "full_mean_validation_macro_f1": full["validation_macro_f1"]["mean"],
        "required_mean_macro_f1": 0.80,
        "macro_f1_gate_pass": gate_macro,
        "full_mean_per_class_recall": recall_means,
        "required_mean_per_class_recall": 0.65,
        "per_class_recall_gate_pass": gate_classes,
        "gate_pass": gate_macro and gate_classes,
        "final_test_evaluation_started": False,
        "meaning": "gate eligibility only; final-test evaluation still requires a separate explicit main-agent action",
    }

    angle_values = {
        "action_argmax_agreement_fraction": [],
        "max_absolute_action_probability_difference": [],
        "macro_f1_difference_128_minus_64": [],
    }
    angle_per_seed = []
    for row in by_variant["full"]:
        diagnostic = row["angle_resolution_diagnostic"]
        angle_per_seed.append({"seed": row["seed"], **diagnostic})
        for key in angle_values:
            angle_values[key].append(float(diagnostic[key]))

    return {
        "format": "c170-latent-angle-validation-summary",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "integrity": {
            "formal_run_count": sum(len(rows) for rows in by_variant.values()),
            "expected_run_count": 25,
            "all_hashes_verified": True,
            "source_snapshot_hashes": snapshot_hashes,
            "all_runs_test_metrics_computed": False,
            "test_predictions_or_scores_read": False,
        },
        "gate": gate,
        "variants": variant_summary,
        "paired_seed_macro_f1_differences": paired,
        "learning_curve_summary_by_variant_and_epoch": curve_summary,
        "full_model_64_vs_128_stability_diagnostic": {
            "purpose": "numerical stability only; never used for checkpoint/model selection",
            "per_seed": angle_per_seed,
            "summary": {key: _stats(values) for key, values in angle_values.items()},
        },
        "_learning_curve_csv_rows": curve_rows_for_csv,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# C170 latent-angle validation results",
        "",
        "This report uses validation predictions only. It does not read test predictions, scores, or labels.",
        "",
        f"Integrity: {summary['integrity']['formal_run_count']}/25 formal runs verified; all marker, manifest, config, artifact and source hashes passed.",
        "",
        "## Validation results",
        "",
        "| Variant | Macro-F1 mean ± SD | UP recall | DOWN recall | LEFT recall | RIGHT recall | Unique predicted classes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in FORMAL_VARIANTS:
        item = summary["variants"][variant]
        def fmt(stat: dict[str, Any]) -> str:
            return f"{stat['mean']:.3f} ± {stat['std_population']:.3f}"
        recalls = [fmt(item["per_class_recall"][direction]) for direction in DIRECTIONS]
        lines.append(
            f"| {variant} | {fmt(item['validation_macro_f1'])} | "
            + " | ".join(recalls)
            + f" | {fmt(item['prediction_unique_class_count'])} |"
        )
    gate = summary["gate"]
    lines.extend(
        [
            "",
            "## Validation gate",
            "",
            f"Full mean macro-F1 gate: {'PASS' if gate['macro_f1_gate_pass'] else 'FAIL'} "
            f"({gate['full_mean_validation_macro_f1']:.3f} against 0.800).",
            f"All four mean class-recall gates: {'PASS' if gate['per_class_recall_gate_pass'] else 'FAIL'} "
            "(each must be at least 0.650).",
            "Final-test evaluation was not started by this analysis.",
            "",
            "## Paired seed differences",
            "",
            "| Comparison | Mean full-minus-control macro-F1 | SD | Seed differences |",
            "|---|---:|---:|---|",
        ]
    )
    for name, item in summary["paired_seed_macro_f1_differences"].items():
        seed_values = ", ".join(f"{seed}: {value:+.3f}" for seed, value in item["per_seed"].items())
        lines.append(
            f"| {name.replace('_', ' ')} | {item['difference']['mean']:+.3f} | "
            f"{item['difference']['std_population']:.3f} | {seed_values} |"
        )
    lines.extend(
        [
            "",
            "## By-position validation macro-F1",
            "",
            "| Variant | Position (mm) | Mean ± SD |",
            "|---|---:|---:|",
        ]
    )
    for variant in FORMAL_VARIANTS:
        for position, item in summary["variants"][variant]["by_position"].items():
            stat = item["macro_f1"]
            lines.append(f"| {variant} | {position} | {stat['mean']:.3f} ± {stat['std_population']:.3f} |")
    lines.extend(
        [
            "",
            "## Selected epochs",
            "",
            "| Variant | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Seed 89 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in FORMAL_VARIANTS:
        epochs = summary["variants"][variant]["selected_epoch_one_based"]
        lines.append(f"| {variant} | " + " | ".join(str(value) for value in epochs) + " |")
    angle = summary["full_model_64_vs_128_stability_diagnostic"]["summary"]
    lines.extend(
        [
            "",
            "## 64-versus-128 angle-grid stability",
            "",
            "This validation-only diagnostic did not participate in checkpoint or model selection.",
            f"Mean argmax agreement: {angle['action_argmax_agreement_fraction']['mean']:.6f}; "
            f"mean maximum action-probability difference: {angle['max_absolute_action_probability_difference']['mean']:.3g}; "
            f"mean macro-F1 difference (128−64): {angle['macro_f1_difference_128_minus_64']['mean']:+.4f}.",
            "",
            "The full JSON and curve CSV include per-seed confusion matrices, per-class recalls, "
            "prediction diversity, all validation curves, position×channel-map strata, and the shuffle-control distribution.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    if output_dir.exists():
        raise AuditError(f"refusing to overwrite existing analysis output: {output_dir}")
    output_dir.mkdir(parents=True)
    curve_rows = summary.pop("_learning_curve_csv_rows")
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_summary.md").write_text(_markdown(summary), encoding="utf-8")
    fields = ["variant", "epoch", "seed_count"]
    for name in CURVE_FIELDS:
        fields.extend([f"{name}_mean", f"{name}_std_population"])
    with (output_dir / "learning_curve_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(curve_rows)
    seed_fields = [
        "variant", "seed", "run_id", "selected_epoch_one_based", "validation_macro_f1",
        "validation_cross_entropy", *[f"recall_{d.lower()}" for d in DIRECTIONS],
        "prediction_unique_class_count", *[f"predicted_{d.lower()}" for d in DIRECTIONS],
    ]
    seed_rows = []
    for variant in FORMAL_VARIANTS:
        for row in summary["variants"][variant]["per_seed"]:
            seed_rows.append({
                "variant": variant,
                "seed": row["seed"],
                "run_id": row["run_id"],
                "selected_epoch_one_based": row["selected_epoch_one_based"],
                "validation_macro_f1": row["validation_macro_f1"],
                "validation_cross_entropy": row["validation_cross_entropy"],
                **{f"recall_{d.lower()}": row["per_class_recall"][d] for d in DIRECTIONS},
                "prediction_unique_class_count": row["prediction_unique_class_count"],
                **{f"predicted_{d.lower()}": row["prediction_counts"][d] for d in DIRECTIONS},
            })
    with (output_dir / "per_seed_validation.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=seed_fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(seed_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots", type=Path, nargs="+", required=True,
        help="formal result roots, including any recovery roots; each contains variant__seed-* run directories",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="new analysis output directory; existing paths are never overwritten",
    )
    args = parser.parse_args(argv)
    roots = [path.expanduser().resolve() for path in args.roots]
    output_dir = args.output_dir.expanduser().resolve()
    try:
        _, expected_source_hashes = _read_snapshot(roots)
        markers = _candidate_markers(roots)
        verified = [_verify_completed_run(path, expected_source_hashes) for path in markers]
        candidates: dict[tuple[str, int], dict[str, Any]] = {}
        for run in verified:
            key = (run["variant"], run["seed"])
            if key in candidates:
                raise AuditError(f"duplicate completed run for {key}: {candidates[key]['run_dir']} and {run['run_dir']}")
            candidates[key] = run
        expected_pairs = {(variant, seed) for variant in FORMAL_VARIANTS for seed in FORMAL_SEEDS}
        missing = sorted(expected_pairs - set(candidates))
        extra = sorted(set(candidates) - expected_pairs)
        if missing or extra or len(candidates) != 25:
            raise AuditError(
                f"formal suite incomplete or has unexpected runs; found={len(candidates)}/25, "
                f"missing={missing}, extra={extra}; no partial metrics were aggregated"
            )

        dataset_hashes = {run["config"].get("dataset_sha256") for run in verified}
        if len(dataset_hashes) != 1 or None in dataset_hashes:
            raise AuditError("formal runs do not share one dataset SHA256")
        analyzed = [_analyze_run(candidates[(variant, seed)]) for variant in FORMAL_VARIANTS for seed in FORMAL_SEEDS]
        val_id_sets = {tuple(run.pop("_validation_sample_ids")) for run in analyzed}
        if len(val_id_sets) != 1:
            raise AuditError("validation sample identities differ across runs")
        summary = _aggregate(analyzed, expected_source_hashes)
        summary["integrity"]["dataset_sha256"] = next(iter(dataset_hashes))
        summary["integrity"]["source_roots"] = [str(root) for root in roots]
        summary["integrity"]["validation_sample_count_per_run"] = 320
        summary["integrity"]["validation_sample_identity_sets_consistent"] = True
        _write_outputs(output_dir, summary)
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"verified 25 formal runs; wrote validation-only report to {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry point
    raise SystemExit(main())
