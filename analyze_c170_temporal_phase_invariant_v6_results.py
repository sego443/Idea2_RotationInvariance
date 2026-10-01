"""One-off validation-only aggregation for the frozen C170 temporal phase v6 suite."""

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

import analyze_c170_harmonic_nyquist_v5_results as v5
import analyze_c170_ring_cuda_v2_results as v2
import launch_c170_temporal_phase_invariant_v6 as driver


VARIANTS = driver.FORMAL_VARIANTS
SEEDS = driver.SEEDS
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
        raise AuditError(f"expected a JSON object: {path}")
    return value


def _snapshot(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Verify code/docs snapshot without opening the NPZ dataset."""
    snapshot_dir = root.resolve() / "source_snapshot"
    snapshot = _read_json(snapshot_dir / "snapshot_manifest.json")
    if "v6" not in str(snapshot.get("format", "")).lower():
        raise AuditError("unexpected v6 source snapshot format")
    code_hashes = snapshot.get("provenance_code_sha256")
    copied_hashes = snapshot.get("snapshot_file_sha256")
    expected_hashes = {key: _sha256(path) for key, path in driver.SOURCE_PATHS.items()}
    if code_hashes != expected_hashes:
        raise AuditError("v6 source snapshot hashes differ from frozen source files")
    if not isinstance(copied_hashes, dict) or not set(driver.SNAPSHOT_DOC_COPIES).issubset(copied_hashes):
        raise AuditError("v6 source snapshot is missing copied-file hashes or design/plan")
    for relative, expected in copied_hashes.items():
        source = (snapshot_dir / driver.SNAPSHOT_DOC_COPIES[relative]
                  if relative in driver.SNAPSHOT_DOC_COPIES else snapshot_dir / "code" / relative)
        if not source.is_file() or _sha256(source) != expected:
            raise AuditError(f"source snapshot file hash mismatch: {source}")
    dataset = snapshot.get("dataset")
    dataset_hash = dataset.get("sha256") if isinstance(dataset, dict) else None
    if not isinstance(dataset_hash, str) or len(dataset_hash) != 64:
        raise AuditError("v6 snapshot is missing the recorded dataset hash")
    return {str(key): str(value) for key, value in code_hashes.items()}, snapshot


def _matrix(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        try:
            key = (str(record["variant"]), int(record["seed"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"invalid v6 run entry: {record!r}") from exc
        if key not in expected or key in found:
            raise AuditError(f"unexpected or duplicate v6 arm/seed: {key}")
        found[key] = record
    missing = sorted(expected - set(found))
    if missing:
        raise AuditError(f"incomplete v6 matrix; missing {missing}")
    return [found[(variant, seed)] for variant in VARIANTS for seed in SEEDS]


def _resolve_runs(root: Path, mapping_path: Path | None) -> list[dict[str, Any]]:
    if mapping_path is None:
        markers = sorted(root.resolve().rglob("RUN_COMPLETE.json"))
        if len(markers) != 20:
            raise AuditError(f"expected exactly 20 v6 completion markers, found {len(markers)}")
        entries = []
        for marker in markers:
            run_dir = marker.parent
            manifest = _read_json(run_dir / "run_manifest.json")
            config = manifest.get("config", {})
            entries.append({"variant": config.get("variant"), "seed": config.get("seed"), "run_dir": str(run_dir)})
        return _matrix(entries)
    mapping_path = mapping_path.resolve()
    mapping = _read_json(mapping_path)
    entries = mapping.get("entries")
    if not isinstance(entries, list):
        raise AuditError(f"mapping must contain an entries list: {mapping_path}")
    resolved = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("run_dir"), str):
            raise AuditError(f"invalid v6 mapping entry: {mapping_path}")
        if entry.get("partial_checkpoint_reused") is True:
            raise AuditError(f"mapped v6 run declares partial checkpoint reuse: {entry}")
        path = Path(entry["run_dir"])
        if not path.is_absolute():
            path = Path(mapping["recovery_root"]) / path if mapping.get("recovery_root") else mapping_path.parent / path
        resolved.append({**entry, "run_dir": str(path.resolve())})
    return _matrix(resolved)


def _verify_run(run_dir: Path, variant: str, seed: int, dataset_hash: str,
                source_hashes: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = run_dir / "run_manifest.json"
    marker_path = run_dir / "RUN_COMPLETE.json"
    manifest = _read_json(manifest_path)
    marker = _read_json(marker_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise AuditError(f"missing run config: {run_dir}")
    config_hash = hashlib.sha256(driver._canonical_json(config).encode("utf-8")).hexdigest()
    if manifest.get("config_sha256") != config_hash or marker.get("config_sha256") != config_hash:
        raise AuditError(f"config hash mismatch: {run_dir}")
    expected = {
        "variant": variant,
        "model_variant": "temporal_phase_invariant_classifier_v6",
        "active_feature_indices": driver.ACTIVE_FEATURES[variant],
        "feature_names": driver.FEATURE_NAMES,
        "feature_definition": driver.FEATURE_DEFINITION,
        "shuffle_labels": variant == "learned_shuffle",
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
        "feature_lags": driver.FEATURE_LAGS,
        "fir_lags": driver.FEATURE_LAGS,
        "lag_boundary": "causal_left_zero_pad_64_no_future",
        "variant_feature_masks": driver.ACTIVE_FEATURES,
        "fir_bank": {"filters": 8, "kernel_size": 65, "learned_parameter_count": 455,
                      "identity_is_fixed_buffer": True, "fixed_gram_freezes_seven_filter_parameters": True},
        "architecture": {
            "stem": "conv1d_23_to_24_k9_stride2_pad4_biasfalse_gn4_silu",
            "residual_dilations": [1, 2, 4],
            "residual_block": "depthwise24_k7_gn4_silu_pointwise24_gn4_residual_silu",
            "temporal_pooling": "adaptive_avg_lengths_1_2_4_plus_population_std_unbiased_false",
            "dropout": 0.2,
            "classifier": "linear_192_to_4",
            "total_parameters": 8763,
            "trainable_parameters": 8308 if variant == "fixed_gram" else 8763,
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
            raise AuditError(f"frozen config mismatch for {key}: {run_dir}")
    if (manifest.get("format") != "c170-temporal-phase-invariant-v6-run" or manifest.get("format_version") != 1
            or manifest.get("status") != "completed"):
        raise AuditError(f"run manifest is not a completed v6 run: {run_dir}")
    if (marker.get("format") != "c170-temporal-phase-invariant-v6-complete" or marker.get("format_version") != 1
            or marker.get("status") != "completed"):
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
    if manifest.get("provenance", {}).get("code_sha256") != source_hashes:
        raise AuditError(f"run source hashes differ from snapshot: {run_dir}")
    split = manifest.get("split", {})
    if split.get("split_counts") != {"train": 879, "val": 320, "test": 320} or split.get("v6_variant") != variant:
        raise AuditError(f"v6 split metadata mismatch: {run_dir}")
    shuffled = variant == "learned_shuffle"
    if split.get("label_shuffle_seed") != (seed + 10_000 if shuffled else None):
        raise AuditError(f"label shuffle seed mismatch: {run_dir}")
    shuffle_map = split.get("label_shuffle_mapping")
    if not isinstance(shuffle_map, list) or (shuffled and not shuffle_map) or (not shuffled and shuffle_map):
        raise AuditError(f"label shuffle mapping mismatch: {run_dir}")
    prep = manifest.get("preprocessing", {})
    if (prep.get("normalization_fit_splits") != ["train"] or prep.get("processed_splits") != ["train", "validation"]
            or prep.get("test_preprocessed") is not False or prep.get("channel_map_as_model_input") is not False
            or float(prep.get("shared_sensor_scale", float("nan"))) != driver.EXPECTED_SHARED_SCALE):
        raise AuditError(f"preprocessing split/scale mismatch: {run_dir}")
    if marker.get("selected_epoch_one_based") != manifest.get("selection", {}).get("best_epoch_one_based"):
        raise AuditError(f"selected epoch differs from completion marker: {run_dir}")

    names = {"best_checkpoint": "best_checkpoint.pt", "safe_state_dict": "best_weights.npz",
             "learning_curve": "learning_curve.csv", "validation_predictions": "validation_predictions.csv"}
    artifacts = manifest.get("artifacts")
    marker_hashes = marker.get("artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != set(names) or not isinstance(marker_hashes, dict) or set(marker_hashes) != set(names):
        raise AuditError(f"artifact list changed: {run_dir}")
    artifact_hashes: dict[str, str] = {}
    for name, filename in names.items():
        path = run_dir / filename
        digest = _sha256(path) if path.is_file() else ""
        record = artifacts.get(name, {})
        if not digest or record.get("path") != filename or record.get("sha256") != digest or marker_hashes.get(name) != digest:
            raise AuditError(f"artifact hash mismatch: {path}")
        artifact_hashes[name] = digest
        if name == "safe_state_dict":
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if not archive.files:
                        raise AuditError(f"safe NPZ is empty: {path}")
                    for tensor_name in archive.files:
                        array = archive[tensor_name]
                        if array.dtype.hasobject or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                            raise AuditError(f"invalid safe array {tensor_name}: {path}")
            except (OSError, ValueError) as exc:
                raise AuditError(f"cannot safely inspect selected NPZ: {path}: {exc}") from exc
    with (run_dir / "learning_curve.csv").open("r", encoding="utf-8-sig", newline="") as stream:
        curve = list(csv.DictReader(stream))
    selection = manifest.get("selection", {})
    if not curve or len(curve) != selection.get("actual_epochs"):
        raise AuditError(f"learning-curve length mismatch: {run_dir}")
    for epoch, row in enumerate(curve, 1):
        if int(row["epoch"]) != epoch or int(row["optimizer_updates"]) != driver.UPDATES_PER_EPOCH:
            raise AuditError(f"learning-curve update count mismatch: {run_dir}")
    return manifest, {"config_sha256": config_hash, "artifact_sha256": artifact_hashes}


def _load_runs(root: Path, mapping_path: Path | None) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    source_hashes, snapshot = _snapshot(root)
    dataset_hash = snapshot.get("dataset", {}).get("sha256")
    records = _resolve_runs(root, mapping_path)
    verified: list[dict[str, Any]] = []
    for record in records:
        run_dir = Path(record["run_dir"]).resolve()
        marker = run_dir / "RUN_COMPLETE.json"
        if not marker.is_file():
            raise AuditError(f"mapped run has no completion marker: {run_dir}")
        variant, seed = str(record["variant"]), int(record["seed"])
        manifest, checked = _verify_run(run_dir, variant, seed, str(dataset_hash), source_hashes)
        if record.get("config_sha256") is not None and record["config_sha256"] != checked["config_sha256"]:
            raise AuditError(f"mapping config hash mismatch: {run_dir}")
        metric_run = {"variant": variant, "seed": seed, "run_dir": run_dir, "manifest": manifest}
        row = v5._read_validation(metric_run)
        row.pop("validation_signature", None)
        row["artifact_sha256"] = checked["artifact_sha256"]
        verified.append(row)
    if len({row["validation_signature_sha256"] for row in verified}) != 1:
        raise AuditError("validation identities/labels differ across v6 runs")
    return _matrix(verified), source_hashes, snapshot


def _baseline(path: Path, variant: str, expected_signature: str, expected_format: str) -> list[dict[str, Any]]:
    summary = _read_json(path)
    if (summary.get("format") != expected_format or summary.get("validation_only") is not True
            or summary.get("test_metrics_computed") is not False or summary.get("verified_run_count") != 20):
        raise AuditError(f"historical baseline is not a completed validation-only 20-run report: {path}")
    rows = [row for row in summary.get("per_seed", []) if row.get("variant") == variant]
    if len(rows) != 5 or {int(row["seed"]) for row in rows} != set(SEEDS):
        raise AuditError(f"baseline arm {variant} is missing a seed: {path}")
    for row in rows:
        if row.get("validation_signature_sha256") != expected_signature:
            raise AuditError(f"validation identity mismatch against baseline: {path}")
    return [{"variant": variant, "seed": int(row["seed"]),
             "validation_macro_f1": float(row["validation_macro_f1"]),
             "validation_signature_sha256": row["validation_signature_sha256"]} for row in rows]


def _aggregate(rows: list[dict[str, Any]], root: Path, mapping_path: Path | None,
               source_hashes: dict[str, str], snapshot: dict[str, Any],
               v4_rows: list[dict[str, Any]] | None, v5_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    by_arm = {variant: {row["seed"]: row for row in rows if row["variant"] == variant} for variant in VARIANTS}
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [by_arm[variant][seed] for seed in SEEDS]
        positions = {}
        for position in map(str, POSITIONS):
            cells = [row["validation_by_position"][position] for row in selected]
            positions[position] = {
                "macro_f1": v2._stats([cell["macro_f1"] for cell in cells]),
                "per_class_recall": {direction: v2._stats([cell["per_class_recall"][i] for cell in cells])
                                      for i, direction in enumerate(DIRECTIONS)},
            }
        trainable = 8308 if variant == "fixed_gram" else 8763
        variants[variant] = {
            "total_parameters": 8763,
            "trainable_parameters": trainable,
            "validation_macro_f1": v2._stats([r["validation_macro_f1"] for r in selected]),
            "validation_cross_entropy": v2._stats([r["validation_cross_entropy"] for r in selected]),
            "validation_per_class_recall": {d: v2._stats([r["validation_per_class_recall"][d] for r in selected]) for d in DIRECTIONS},
            "selected_checkpoint_train_target_macro_f1": v2._stats([r["selected_checkpoint_train_target_macro_f1"] for r in selected]),
            "train_target_minus_validation_macro_f1": v2._stats([r["train_target_minus_validation_macro_f1"] for r in selected]),
            "validation_by_position": positions,
            "selected_epoch_one_based_by_seed": {str(r["seed"]): r["selected_epoch_one_based"] for r in selected},
        }
    differences: dict[str, dict[int, float]] = {
        "learned_gram_minus_fixed_gram": {s: by_arm["learned_gram"][s]["validation_macro_f1"] - by_arm["fixed_gram"][s]["validation_macro_f1"] for s in SEEDS},
        "learned_gram_minus_learned_power": {s: by_arm["learned_gram"][s]["validation_macro_f1"] - by_arm["learned_power"][s]["validation_macro_f1"] for s in SEEDS},
        "learned_gram_minus_learned_shuffle": {s: by_arm["learned_gram"][s]["validation_macro_f1"] - by_arm["learned_shuffle"][s]["validation_macro_f1"] for s in SEEDS},
    }
    for baseline_name, baseline_rows in (("v4_harmonic_gram", v4_rows), ("v5_gram_control", v5_rows)):
        if baseline_rows is not None:
            base_by_seed = {r["seed"]: r for r in baseline_rows}
            differences[f"learned_gram_minus_{baseline_name}"] = {
                s: by_arm["learned_gram"][s]["validation_macro_f1"] - base_by_seed[s]["validation_macro_f1"] for s in SEEDS
            }
            differences[f"fixed_gram_minus_{baseline_name}"] = {
                s: by_arm["fixed_gram"][s]["validation_macro_f1"] - base_by_seed[s]["validation_macro_f1"] for s in SEEDS
            }
    paired = {name: {"per_seed": {str(s): value for s, value in values.items()}, **v2._stats(list(values.values()))}
              for name, values in differences.items()}
    return {
        "format": "c170-temporal-phase-invariant-v6-validation-summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "validation_only": True,
        "test_metrics_computed": False,
        "test_access": {"loaded_as_part_of_complete_npz": True, "preprocessed": False,
                        "used_to_estimate_statistics": False, "used_as_model_input": False, "predicted_or_scored": False},
        "expected_run_count": 20,
        "verified_run_count": len(rows),
        "standard_deviation": "population standard deviation (ddof=0)",
        "formal_root": str(root.resolve()),
        "run_mapping": str(mapping_path.resolve()) if mapping_path else None,
        "source_snapshot_code_sha256": source_hashes,
        "dataset_sha256_from_snapshot_metadata": snapshot.get("dataset", {}).get("sha256"),
        "variants": variants,
        "paired_seed_macro_f1_differences": paired,
        "per_seed": rows,
        "v4_harmonic_gram_baseline_per_seed": v4_rows,
        "v5_gram_control_baseline_per_seed": v5_rows,
        "interpretation_boundary": (
            "Metrics are recomputed only from validation prediction CSVs and selected-checkpoint train metrics in run manifests. "
            "Aggregation verifies hashes and reads the numeric selected-weight NPZ with allow_pickle=False; it does not open the dataset NPZ. "
            "The test split was not preprocessed, used to estimate statistics, passed to a model, predicted, or scored. "
            "The phase invariance is exact for C4 channel rolls and an ideal common first-harmonic phase; it does not establish continuous physical SO(2) wrist invariance. "
            "The shuffle-arm train-target score uses permuted training labels."
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = ["# C170 Temporal Phase-Invariant v6 Validation Summary", "",
             f"Verified runs: {summary['verified_run_count']}/20. Test metrics computed: false.",
             "All means and standard deviations are across five seeds; scores are validation-only.", "",
             "| Arm | Total/trainable params | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Selected train-target F1 | Train−validation F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in VARIANTS:
        item = summary["variants"][arm]
        metrics = [item["validation_macro_f1"], *(item["validation_per_class_recall"][d] for d in DIRECTIONS),
                   item["selected_checkpoint_train_target_macro_f1"], item["train_target_minus_validation_macro_f1"]]
        cells = [f"{metrics[0]['mean']:.4f} ± {metrics[0]['std_population']:.4f}"]
        cells += [f"{m['mean']:.4f} ± {m['std_population']:.4f}" for m in metrics[1:]]
        lines.append(f"| {arm} | {item['total_parameters']}/{item['trainable_parameters']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Validation Macro-F1 by Position", "", "| Position (mm) | learned_gram | fixed_gram | learned_power | learned_shuffle |", "|---:|---:|---:|---:|---:|"])
    for position in map(str, POSITIONS):
        values = [summary["variants"][arm]["validation_by_position"][position]["macro_f1"] for arm in VARIANTS]
        lines.append(f"| {position} | " + " | ".join(f"{v['mean']:.4f} ± {v['std_population']:.4f}" for v in values) + " |")
    lines.extend(["", "## Paired Seed Differences", "", "Differences are first arm minus comparator.", ""])
    for name, comparison in summary["paired_seed_macro_f1_differences"].items():
        detail = ", ".join(f"{seed}:{value:+.4f}" for seed, value in comparison["per_seed"].items())
        lines.append(f"- {name}: {comparison['mean']:+.4f} ± {comparison['std_population']:.4f}; {detail}.")
    lines.extend(["", summary["interpretation_boundary"], ""])
    return "\n".join(lines)


def _write(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "validation_summary.md").write_text(_markdown(summary), encoding="utf-8")
    fields = ["variant", "seed", "validation_macro_f1", "validation_cross_entropy", "selected_checkpoint_train_target_macro_f1",
              "train_target_minus_validation_macro_f1", "selected_epoch_one_based", "actual_epochs", "optimizer_updates", "run_dir"]
    fields += [f"validation_recall_{d.lower()}" for d in DIRECTIONS] + [f"validation_f1_at_{p}mm" for p in POSITIONS]
    with (output_dir / "per_seed_validation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["per_seed"]:
            out = {key: row[key] for key in fields if key in row}
            out.update({f"validation_recall_{d.lower()}": row["validation_per_class_recall"][d] for d in DIRECTIONS})
            out.update({f"validation_f1_at_{p}mm": row["validation_by_position"][str(p)]["macro_f1"] for p in POSITIONS})
            writer.writerow(out)
    with (output_dir / "paired_seed_differences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("comparison", "seed", "macro_f1_difference"))
        writer.writeheader()
        for name, values in summary["paired_seed_macro_f1_differences"].items():
            for seed, difference in values["per_seed"].items():
                writer.writerow({"comparison": name, "seed": seed, "macro_f1_difference": difference})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--run-mapping", type=Path)
    parser.add_argument("--v4-summary", type=Path, default=Path(__file__).resolve().parent / "outputs" / "c170_harmonic_phase_v4_validation_summary_20260929" / "validation_summary.json")
    parser.add_argument("--v5-summary", type=Path, default=Path(__file__).resolve().parent / "outputs" / "c170_harmonic_nyquist_v5_validation_summary_20260929" / "validation_summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows, source_hashes, snapshot = _load_runs(args.formal_root, args.run_mapping)
        signature = rows[0]["validation_signature_sha256"]
        v4_rows = _baseline(args.v4_summary, "harmonic_gram", signature, "c170-harmonic-phase-v4-validation-summary") if args.v4_summary.is_file() else None
        v5_rows = _baseline(args.v5_summary, "gram_control", signature, "c170-harmonic-nyquist-v5-validation-summary") if args.v5_summary.is_file() else None
        summary = _aggregate(rows, args.formal_root, args.run_mapping, source_hashes, snapshot, v4_rows, v5_rows)
        _write(args.output_dir, summary)
        print(f"VALIDATION_SUMMARY_COMPLETE path={args.output_dir.expanduser().resolve()}")
        print("verified_runs=20/20; test_metrics_computed=False")
        for variant in VARIANTS:
            stat = summary["variants"][variant]["validation_macro_f1"]
            print(f"{variant} macro_f1={stat['mean']:.6f} ± {stat['std_population']:.6f}")
        for name, stat in summary["paired_seed_macro_f1_differences"].items():
            print(f"{name} paired_mean={stat['mean']:+.6f} ± {stat['std_population']:.6f}")
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
