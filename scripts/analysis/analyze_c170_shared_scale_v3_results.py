"""Fail-closed validation-only summary for the fixed C170 shared-scale v3 suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis import analyze_c170_ring_cuda_v2_results as v2
FORMAL_VARIANTS = ("shared_scale_ce", "shared_scale_shuffle")
FORMAL_SEEDS = (17, 29, 43, 61, 89)
DIRECTIONS = v2.DIRECTIONS
VAL_POSITIONS = (5, 40, 90, 130)
SOURCE_FILES = {
    "runner": "run_c170_ring_cuda_experiment_v3.py",
    "shared_scale_preprocessing": "src/c170_shared_scale_preprocessing_v3.py",
    "ring_representation": "src/c170_ring_representation.py",
    "frozen_v2_training_helpers": "run_c170_ring_cuda_experiment_v2.py",
    "dataset_preprocessing_helpers": "run_c170_experiment.py",
    "preprocessing_transitive_signal_processing": "src/rotation_poc.py",
}
AuditError = v2.AuditError


def _validate_exact_matrix_v3(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(variant, seed) for variant in FORMAL_VARIANTS for seed in FORMAL_SEEDS}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    extra: list[tuple[str, int]] = []
    for run in runs:
        variant = str(run.get("variant"))
        try:
            seed = int(run["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"invalid run seed metadata: {run}") from exc
        pair = (variant, seed)
        if pair not in expected:
            extra.append(pair)
        elif pair in found:
            duplicates.append(pair)
        else:
            found[pair] = run
    missing = sorted(expected - set(found))
    if missing or duplicates or extra:
        raise AuditError(f"v3 requires exactly 10 variant/seed pairs; missing={missing}, duplicates={duplicates}, extra={extra}")
    return sorted(found.values(), key=lambda item: (FORMAL_VARIANTS.index(item["variant"]), item["seed"]))


def _candidate_markers(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise AuditError(f"run root is missing: {root}")
    return sorted(path.resolve() for path in root.rglob("RUN_COMPLETE.json"))


def _read_v3_snapshot(root: Path) -> tuple[Path, dict[str, str], dict[str, Any]]:
    snapshot_dir = root.expanduser().resolve() / "source_snapshot"
    path = snapshot_dir / "snapshot_manifest.json"
    snapshot = v2._read_json(path)
    if snapshot.get("format") != "c170-ring-v3-source-snapshot":
        raise AuditError(f"unexpected v3 source snapshot: {path}")
    code_hashes = snapshot.get("provenance_code_sha256")
    copied_hashes = snapshot.get("snapshot_file_sha256")
    if not isinstance(code_hashes, dict) or not isinstance(copied_hashes, dict):
        raise AuditError(f"source snapshot is missing code hashes: {path}")
    result: dict[str, str] = {}
    for key, relative in SOURCE_FILES.items():
        # Frozen bundles use root-level runner paths; new snapshots use scripts/.
        current = f"scripts/runners/{relative}" if "/" not in relative else relative
        saved_relative = current if current in copied_hashes else relative
        digest = code_hashes.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or copied_hashes.get(saved_relative) != digest:
            raise AuditError(f"source snapshot provenance mismatch for {key}")
        saved = snapshot_dir / "code" / saved_relative
        if not saved.is_file() or v2._sha256(saved) != digest:
            raise AuditError(f"copied source hash mismatch for {key}: {saved}")
        result[key] = digest
    return snapshot_dir, result, snapshot


def _validate_safe_npz(path: Path) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if not archive.files:
                raise AuditError(f"safe checkpoint NPZ is empty: {path}")
            for key in archive.files:
                array = archive[key]
                if array.dtype.hasobject or array.dtype.kind not in "biuf":
                    raise AuditError(f"safe checkpoint contains nonnumeric tensor {key}: {path}")
                if array.dtype.kind == "f" and not np.isfinite(array).all():
                    raise AuditError(f"safe checkpoint contains nonfinite values in {key}: {path}")
    except (OSError, ValueError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError(f"invalid allow_pickle=False NPZ checkpoint {path}: {exc}") from exc


def _verify_v3_run(marker_path: Path, source_hashes: dict[str, str]) -> dict[str, Any]:
    run_dir = marker_path.parent
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise AuditError(f"missing run manifest: {run_dir}")
    marker = v2._read_json(marker_path)
    manifest = v2._read_json(manifest_path)
    if marker.get("format") != "c170-ring-cuda-v3-complete" or marker.get("status") != "completed":
        raise AuditError(f"invalid completion marker: {marker_path}")
    if manifest.get("format") != "c170-ring-cuda-v3-run" or manifest.get("status") != "completed":
        raise AuditError(f"run manifest is not completed: {manifest_path}")
    if marker.get("manifest_sha256") != v2._sha256(manifest_path):
        raise AuditError(f"manifest SHA-256 mismatch: {manifest_path}")
    if marker.get("test_metrics_computed") is not False or manifest.get("test_metrics_computed") is not False:
        raise AuditError(f"test metrics were computed according to {run_dir}")
    test_access = manifest.get("test_access")
    if not isinstance(test_access, dict) or test_access.get("loaded_as_part_of_complete_npz") is not True:
        raise AuditError(f"test load/finite-check provenance is missing: {run_dir}")
    if any(test_access.get(key) is not False for key in (
        "used_as_model_input", "preprocessed", "used_to_estimate_statistics", "predicted_or_scored"
    )):
        raise AuditError(f"test boundary violation recorded: {run_dir}")

    config = manifest.get("config")
    if not isinstance(config, dict):
        raise AuditError(f"missing config: {run_dir}")
    config_sha = v2._canonical_config_sha(config)
    if config_sha != manifest.get("config_sha256") or config_sha != marker.get("config_sha256"):
        raise AuditError(f"canonical config hash mismatch: {run_dir}")
    variant = str(config.get("variant"))
    try:
        seed = int(config["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditError(f"invalid variant/seed in {run_dir}") from exc
    if variant not in FORMAL_VARIANTS or seed not in FORMAL_SEEDS:
        raise AuditError(f"unexpected variant/seed {variant}/{seed}")
    expected = {
        "model_variant": "classification_only",
        "use_supcon": False,
        "shuffle_labels": variant == "shared_scale_shuffle",
        "max_epochs": 160,
        "patience": 25,
        "cpu_threads": 2,
        "device": "cuda:0",
        "train_distance_mm": [0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160],
        "validation_distance_mm": [5, 40, 90, 130],
        "registered_test_distance_mm": [15, 25, 110, 150],
        "excluded_distance_mm": [70],
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
        "batch_ce_objective": "20*sum(sample_weight_i*per_sample_ce_i)",
        "supcon_weight": 0.0,
        "preprocessing_scale": "max(median(abs(filtered_per_trial_demeaned_train)),float32_epsilon)",
        "preprocessing_scale_axes": ["train_samples", "time", "channels"],
        "preprocessing_scale_scope": "one_scalar_shared_by_S0_S1_S2_S3",
        "preprocessing_scale_fit_split": "train",
        "safe_state_dict_format": "numpy_npz_numeric_arrays",
        "safe_state_dict_reload": "numpy_load_allow_pickle_false_then_strict_state_dict",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise AuditError(f"frozen config mismatch for {key} in {run_dir}: {config.get(key)!r}")
    if not config.get("dataset_sha256") or not config.get("dataset_path"):
        raise AuditError(f"dataset provenance missing: {run_dir}")
    provenance = manifest.get("provenance", {}).get("code_sha256")
    if provenance != source_hashes:
        raise AuditError(f"training source hashes differ from snapshot: {run_dir}")

    split = manifest.get("split", {})
    if split.get("split_counts") != {"train": 879, "val": 320, "test": 320}:
        raise AuditError(f"split count mismatch: {run_dir}")
    expected_shuffle_seed = seed + 10_000 if variant == "shared_scale_shuffle" else None
    if split.get("label_shuffle_seed") != expected_shuffle_seed:
        raise AuditError(f"shuffle seed mismatch: {run_dir}")
    shuffle_map = split.get("label_shuffle_mapping")
    if not isinstance(shuffle_map, list) or (variant.endswith("shuffle") and not shuffle_map) or (variant == "shared_scale_ce" and shuffle_map):
        raise AuditError(f"shuffle mapping mismatch: {run_dir}")

    prep = manifest.get("preprocessing", {})
    if prep.get("normalization_fit_splits") != ["train"] or prep.get("processed_splits") != ["train", "validation"]:
        raise AuditError(f"preprocessing split provenance mismatch: {run_dir}")
    if prep.get("sensor_scale_axes") != ["train_samples", "time", "channels"]:
        raise AuditError(f"shared-scale axes mismatch: {run_dir}")
    if prep.get("channel_map_as_model_input") is not False or prep.get("test_preprocessed") is not False:
        raise AuditError(f"unexpected channel-map/test preprocessing: {run_dir}")
    scale = prep.get("shared_sensor_scale")
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise AuditError(f"invalid train-only shared scale: {run_dir}")
    if marker.get("selected_epoch_one_based") != manifest.get("selection", {}).get("best_epoch_one_based"):
        raise AuditError(f"selected epoch differs between marker and manifest: {run_dir}")

    artifact_paths = {
        "best_checkpoint": "best_checkpoint.pt",
        "safe_state_dict": "best_weights.npz",
        "learning_curve": "learning_curve.csv",
        "validation_predictions": "validation_predictions.csv",
    }
    artifacts = manifest.get("artifacts")
    marker_artifacts = marker.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != set(artifact_paths):
        raise AuditError(f"manifest artifact set mismatch: {run_dir}")
    if not isinstance(marker_artifacts, dict) or set(marker_artifacts) != set(artifact_paths):
        raise AuditError(f"marker artifact set mismatch: {run_dir}")
    hashes: dict[str, str] = {}
    for name, filename in artifact_paths.items():
        record = artifacts[name]
        path = run_dir / filename
        if not isinstance(record, dict) or record.get("path") != filename or not path.is_file():
            raise AuditError(f"missing or incorrectly named artifact {name}: {run_dir}")
        digest = v2._sha256(path)
        if record.get("sha256") != digest or marker_artifacts.get(name) != digest:
            raise AuditError(f"artifact SHA-256 mismatch: {path}")
        hashes[name] = digest
        if name == "safe_state_dict":
            _validate_safe_npz(path)
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": run_dir,
        "manifest": manifest,
        "marker": marker,
        "config": config,
        "artifact_sha256": hashes,
        "shared_sensor_scale": float(scale),
    }


def _stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    return v2._stats(values)


def _sum_matrices(matrices: Sequence[list[list[int]]]) -> list[list[int]]:
    return [[sum(int(matrix[row][col]) for matrix in matrices) for col in range(4)] for row in range(4)]


def _aggregate_v3(
    rows: list[dict[str, Any]],
    v2_ring_ce_rows: list[dict[str, Any]],
    snapshot_hashes: dict[str, str],
    roots: Sequence[Path],
    v2_root: Path,
) -> dict[str, Any]:
    exact = _validate_exact_matrix_v3(rows)
    base = {int(row["seed"]): row for row in v2_ring_ce_rows}
    if len(v2_ring_ce_rows) != 5 or set(base) != set(FORMAL_SEEDS):
        raise AuditError("v2 ring_ce comparator must have one run per registered seed")
    scale_values = {float(row["shared_sensor_scale"]) for row in exact if "shared_sensor_scale" in row}
    if len(scale_values) != 1 or any("shared_sensor_scale" not in row for row in exact):
        raise AuditError("all ten runs must record the same shared train scale")

    variants: dict[str, Any] = {}
    for variant in FORMAL_VARIANTS:
        selected = [row for row in exact if row["variant"] == variant]
        by_position: dict[str, Any] = {}
        for position in map(str, VAL_POSITIONS):
            cells = [row["validation_by_position"][position] for row in selected]
            by_position[position] = {
                "macro_f1": _stats(cell["macro_f1"] for cell in cells),
                "per_class_recall": {
                    direction: _stats(cell["per_class_recall"][index] for cell in cells)
                    for index, direction in enumerate(DIRECTIONS)
                },
                "summed_confusion_matrix": _sum_matrices([cell["confusion_matrix"] for cell in cells]),
            }
        variants[variant] = {
            "validation_macro_f1": _stats(row["validation_macro_f1"] for row in selected),
            "validation_cross_entropy": _stats(row["validation_cross_entropy"] for row in selected),
            "validation_per_class_recall": {
                direction: _stats(row["validation_per_class_recall"][direction] for row in selected)
                for direction in DIRECTIONS
            },
            "validation_summed_confusion_matrix": _sum_matrices(
                [row["validation_confusion_matrix"] for row in selected]
            ),
            "validation_prediction_counts_by_class": {
                direction: _stats(row["validation_prediction_counts_by_class"][direction] for row in selected)
                for direction in DIRECTIONS
            },
            "validation_prediction_share_by_class": {
                direction: _stats(row["validation_prediction_counts_by_class"][direction] / 320.0 for row in selected)
                for direction in DIRECTIONS
            },
            "prediction_unique_class_count": _stats(row["prediction_unique_class_count"] for row in selected),
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
            "selected_epoch_one_based_by_seed": {str(row["seed"]): row["selected_epoch_one_based"] for row in selected},
            "actual_epochs_by_seed": {str(row["seed"]): row["actual_epochs"] for row in selected},
            "validation_by_position": by_position,
        }

    by_variant_seed = {
        variant: {int(row["seed"]): row for row in exact if row["variant"] == variant}
        for variant in FORMAL_VARIANTS
    }
    paired: dict[str, Any] = {}
    for variant in FORMAL_VARIANTS:
        comparisons: dict[str, dict[int, float]] = {
            f"{variant}_minus_v2_ring_ce": {
                seed: by_variant_seed[variant][seed]["validation_macro_f1"] - base[seed]["validation_macro_f1"]
                for seed in FORMAL_SEEDS
            }
        }
        if variant == "shared_scale_ce":
            comparisons["shared_scale_ce_minus_shared_scale_shuffle"] = {
                seed: by_variant_seed["shared_scale_ce"][seed]["validation_macro_f1"]
                - by_variant_seed["shared_scale_shuffle"][seed]["validation_macro_f1"]
                for seed in FORMAL_SEEDS
            }
        for name, differences in comparisons.items():
            paired[name] = {
                "per_seed": {str(seed): value for seed, value in differences.items()},
                "mean": _stats(differences.values())["mean"],
                "std_population": _stats(differences.values())["std_population"],
            }

    candidate = variants["shared_scale_ce"]
    gate = {
        "candidate": "shared_scale_ce",
        "macro_f1_threshold": 0.80,
        "per_class_mean_recall_threshold": 0.65,
        "macro_f1_pass": candidate["validation_macro_f1"]["mean"] >= 0.80,
        "per_class_recall_pass": {
            direction: candidate["validation_per_class_recall"][direction]["mean"] >= 0.65
            for direction in DIRECTIONS
        },
    }
    gate["passed"] = bool(gate["macro_f1_pass"] and all(gate["per_class_recall_pass"].values()))
    return {
        "format": "c170-ring-cuda-v3-validation-summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "validation_only": True,
        "test_metrics_computed": False,
        "standard_deviation": "population standard deviation (ddof=0)",
        "expected_run_count": 10,
        "verified_run_count": len(exact),
        "shared_train_scale": next(iter(scale_values)),
        "source_snapshot_code_sha256": snapshot_hashes,
        "input_roots": [str(path.resolve()) for path in roots],
        "v2_ring_ce_comparator_root": str(v2_root.resolve()),
        "gate": gate,
        "variants": variants,
        "paired_seed_macro_f1_differences": paired,
        "per_seed": exact,
        "v2_ring_ce_per_seed": v2_ring_ce_rows,
        "interpretation_boundary": (
            "The complete compressed NPZ is decoded and the loader finite-checks X_raw, including registered-test waveforms. "
            "Only train and validation indices enter preprocessing and model computations; test is not preprocessed, used to "
            "estimate the shared scale, predicted, or scored. Shuffle-arm train metrics use permuted training labels. This "
            "experiment tests the discrete C4 channel-roll property of preprocessing and does not establish continuous SO(2) "
            "or real physical rotation invariance. A failed validation gate does not authorize test evaluation."
        ),
    }


def _analyze_run(run: dict[str, Any], v3: bool) -> dict[str, Any]:
    result = v2._analyze_run(run)
    if v3:
        result["shared_sensor_scale"] = run["shared_sensor_scale"]
    return result


def _run_analyze(v3_root: Path, v2_root: Path, output_dir: Path) -> int:
    snapshot_dir, source_hashes, snapshot = _read_v3_snapshot(v3_root)
    marker_paths = _candidate_markers(v3_root)
    if len(marker_paths) != 10:
        raise AuditError(f"expected exactly 10 v3 completion markers; found {len(marker_paths)}")
    verified = [_verify_v3_run(path, source_hashes) for path in marker_paths]
    exact = _validate_exact_matrix_v3(verified)
    dataset_hashes = {run["config"].get("dataset_sha256") for run in exact}
    if len(dataset_hashes) != 1 or None in dataset_hashes:
        raise AuditError("v3 runs do not share one dataset SHA-256")
    if snapshot.get("dataset", {}).get("sha256") not in dataset_hashes:
        raise AuditError("v3 dataset differs from source snapshot")
    if len({run["shared_sensor_scale"] for run in exact}) != 1:
        raise AuditError("v3 seeds recorded different train-only shared scales")
    v3_rows = [_analyze_run(run, True) for run in exact]

    v2_root = v2_root.expanduser().resolve()
    _v2_snapshot, v2_hashes = v2._read_verified_snapshot([v2_root])
    v2_rows: list[dict[str, Any]] = []
    v2_datasets: set[str] = set()
    for seed in FORMAL_SEEDS:
        marker = v2_root / f"ring_ce__seed-{seed}" / "RUN_COMPLETE.json"
        if not marker.is_file():
            raise AuditError(f"missing v2 ring_ce seed {seed}: {marker}")
        checked = v2._verify_run(marker, v2_hashes)
        if checked["variant"] != "ring_ce" or checked["seed"] != seed:
            raise AuditError(f"wrong v2 comparator identity: {marker}")
        v2_datasets.add(str(checked["config"]["dataset_sha256"]))
        v2_rows.append(_analyze_run(checked, False))
    if v2_datasets != dataset_hashes:
        raise AuditError("v2 ring_ce comparator uses a different dataset SHA-256")

    summary = _aggregate_v3(v3_rows, v2_rows, source_hashes, [v3_root], v2_root)
    summary["source_snapshot_path"] = str(snapshot_dir)
    _write_report(output_dir, summary)
    print(f"VALIDATION_SUMMARY_COMPLETE path={output_dir.resolve()}")
    print(f"verified_runs={summary['verified_run_count']}/{summary['expected_run_count']}")
    print(f"shared_train_scale={summary['shared_train_scale']:.10g}")
    print(f"shared_scale_ce_gate={'PASS' if summary['gate']['passed'] else 'FAIL'}")
    print("test_metrics_computed=False")
    return 0


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# C170 Shared-Scale v3 Validation Summary",
        "",
        f"Verified v3 runs: {summary['verified_run_count']}/{summary['expected_run_count']}. Test metrics computed: false.",
        f"Train-only shared scale recorded by all runs: {summary['shared_train_scale']:.10g}.",
        "",
        "## Validation metrics",
        "",
        "Mean ± population SD across five seeds.",
        "",
        "| Variant | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Selected train-target F1 | Train-target weighted CE | Unique predicted classes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in FORMAL_VARIANTS:
        item = summary["variants"][variant]
        recalls = item["validation_per_class_recall"]
        metric_cells = [
            f"{item['validation_macro_f1']['mean']:.4f} ± {item['validation_macro_f1']['std_population']:.4f}",
            *(f"{recalls[name]['mean']:.4f} ± {recalls[name]['std_population']:.4f}" for name in DIRECTIONS),
            f"{item['selected_checkpoint_train_target_macro_f1']['mean']:.4f} ± {item['selected_checkpoint_train_target_macro_f1']['std_population']:.4f}",
            f"{item['selected_checkpoint_train_target_weighted_ce']['mean']:.4f} ± {item['selected_checkpoint_train_target_weighted_ce']['std_population']:.4f}",
            f"{item['prediction_unique_class_count']['mean']:.2f} ± {item['prediction_unique_class_count']['std_population']:.2f}",
        ]
        lines.append(f"| {variant} | " + " | ".join(metric_cells) + " |")
    candidate = summary["variants"]["shared_scale_ce"]
    lines.extend(["", "## Validation gate", "", "Candidate: shared_scale_ce. Thresholds: mean macro-F1 ≥ 0.80 and every class mean recall ≥ 0.65.", ""])
    for direction, passed in summary["gate"]["per_class_recall_pass"].items():
        recall = candidate["validation_per_class_recall"][direction]["mean"]
        lines.append(f"- {direction} mean recall {recall:.4f}: {'PASS' if passed else 'FAIL'}.")
    lines.append(
        f"- Macro-F1 mean {candidate['validation_macro_f1']['mean']:.4f}: "
        f"{'PASS' if summary['gate']['macro_f1_pass'] else 'FAIL'}."
    )
    lines.append(f"- Overall gate: {'PASS' if summary['gate']['passed'] else 'FAIL'}.")
    lines.extend(["", "## Validation macro-F1 by position", "", "| Position (mm) | shared_scale_ce | shared_scale_shuffle |", "|---:|---:|---:|"])
    for position in map(str, VAL_POSITIONS):
        values = [summary["variants"][arm]["validation_by_position"][position]["macro_f1"] for arm in FORMAL_VARIANTS]
        lines.append(f"| {position} | " + " | ".join(f"{v['mean']:.4f} ± {v['std_population']:.4f}" for v in values) + " |")
    lines.extend(["", "## Train/validation gaps", "", "Train-target scores for the shuffle arm use permuted training labels.", ""])
    for variant in FORMAL_VARIANTS:
        item = summary["variants"][variant]
        f1 = item["train_target_minus_validation_macro_f1"]
        ce = item["train_target_minus_validation_weighted_ce"]
        lines.append(f"- {variant}: F1 gap {f1['mean']:+.4f} ± {f1['std_population']:.4f}; weighted-CE gap {ce['mean']:+.4f} ± {ce['std_population']:.4f}.")
    lines.extend(["", "## Paired seed differences", "", "Difference is candidate minus comparator validation macro-F1.", ""])
    for name, result in summary["paired_seed_macro_f1_differences"].items():
        lines.append(f"- {name}: {result['mean']:+.4f} ± {result['std_population']:.4f} (population SD).")
    lines.extend(["", "## Shuffle prediction distribution", ""])
    for direction in DIRECTIONS:
        item = summary["variants"]["shared_scale_shuffle"]["validation_prediction_counts_by_class"][direction]
        lines.append(f"- predicted {direction}: {item['mean']:.1f} ± {item['std_population']:.1f} / 320.")
    lines.extend(["", summary["interpretation_boundary"], ""])
    return "\n".join(lines)


def _write_report(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "validation_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    base_fields = (
        "variant", "seed", "validation_macro_f1", "validation_cross_entropy",
        "selected_checkpoint_train_target_macro_f1", "selected_checkpoint_train_target_weighted_ce",
        "train_target_minus_validation_macro_f1", "train_target_minus_validation_weighted_ce",
        "selected_epoch_one_based", "actual_epochs", "optimizer_updates", "prediction_unique_class_count",
        "shared_sensor_scale", "run_dir",
    )
    fields = (
        *base_fields,
        *(f"validation_recall_{direction.lower()}" for direction in DIRECTIONS),
        *(f"validation_f1_at_{position}mm" for position in VAL_POSITIONS),
    )
    with (output_dir / "per_seed_validation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_seed"]:
            item = {key: row[key] for key in base_fields}
            item.update({f"validation_recall_{direction.lower()}": row["validation_per_class_recall"][direction] for direction in DIRECTIONS})
            item.update({f"validation_f1_at_{p}mm": row["validation_by_position"][str(p)]["macro_f1"] for p in VAL_POSITIONS})
            writer.writerow(item)
    with (output_dir / "paired_seed_differences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("comparison", "seed", "macro_f1_difference"))
        writer.writeheader()
        for comparison, item in summary["paired_seed_macro_f1_differences"].items():
            for seed, value in item["per_seed"].items():
                writer.writerow({"comparison": comparison, "seed": seed, "macro_f1_difference": value})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, required=True)
    parser.add_argument("--v2-ring-ce-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run_analyze(args.v3_root, args.v2_ring_ce_root, args.output_dir)
    except (AuditError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
