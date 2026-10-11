"""Fixed 0.5/0.5 validation-only late fusion of v4 harmonic_gram and v6 learned_power."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

from scripts.analysis import analyze_c170_harmonic_phase_v4_results as v4
from scripts.analysis import analyze_c170_harmonic_nyquist_v5_results as v5
from scripts.analysis import analyze_c170_ring_cuda_v2_results as v2
from scripts.analysis import analyze_c170_temporal_phase_invariant_v6_results as v6
SEEDS = (17, 29, 43, 61, 89)
POSITIONS = (5, 40, 90, 130)
DIRECTIONS = v2.DIRECTIONS
AuditError = v2.AuditError


def _read_rows(path: Path, variant: str, seed: int) -> tuple[list[dict[str, str]], list[tuple[int, int, str, tuple[int, ...]]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 320:
        raise AuditError(f"expected 320 validation rows: {path}")
    identity = []
    samples = set()
    for row in rows:
        if row.get("variant") != variant or int(row.get("seed", -1)) != seed:
            raise AuditError(f"CSV arm/seed identity mismatch: {path}")
        sample = int(row["sample_index"])
        if sample in samples:
            raise AuditError(f"duplicate validation sample index {sample}: {path}")
        samples.add(sample)
        distance = int(row["distance_mm"])
        truth = row["true_direction"]
        channel_map = tuple(int(x) for x in row["channel_map"].split(","))
        if distance not in POSITIONS or truth not in DIRECTIONS or len(channel_map) != 4:
            raise AuditError(f"invalid validation identity metadata: {path}")
        identity.append((sample, distance, truth, channel_map))
    return rows, identity


def _probabilities(row: dict[str, str], path: Path) -> list[float]:
    probs = [float(row[f"probability_{direction.lower()}"]) for direction in DIRECTIONS]
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probs):
        raise AuditError(f"invalid probability in {path}")
    if not math.isclose(sum(probs), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise AuditError(f"probabilities do not sum to one in {path}")
    return probs


def _metrics(labels: list[int], predictions: list[int], distances: list[int]) -> dict[str, Any]:
    overall = v2._classification_metrics(labels, predictions)
    by_position = {}
    for position in POSITIONS:
        indices = [i for i, distance in enumerate(distances) if distance == position]
        if len(indices) != 80:
            raise AuditError(f"expected 80 validation rows at {position} mm")
        by_position[str(position)] = v2._classification_metrics(
            [labels[i] for i in indices], [predictions[i] for i in indices]
        )
    return {
        "macro_f1": overall["macro_f1"],
        "per_class_recall": dict(zip(DIRECTIONS, overall["per_class_recall"])),
        "prediction_unique_class_count": overall["prediction_unique_class_count"],
        "by_position": by_position,
    }


def _verify_branch_runs(v4_root: Path, v6_root: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    _v4_snapshot_dir, v4_hashes, v4_snapshot = v4._verify_snapshot(v4_root)
    v6_hashes, v6_snapshot = v6._snapshot(v6_root)
    v4_dataset_hash = str(v4_snapshot.get("dataset", {}).get("sha256", ""))
    v6_dataset_hash = str(v6_snapshot.get("dataset", {}).get("sha256", ""))
    if not v4_dataset_hash or v4_dataset_hash != v6_dataset_hash:
        raise AuditError("v4/v6 source snapshots do not name the same dataset hash")

    v4_runs: dict[int, dict[str, Any]] = {}
    v6_runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        v4_marker = v4_root.resolve() / f"harmonic_gram__seed-{seed}" / "RUN_COMPLETE.json"
        v4_run = v4._verify_run(v4_marker, v4_hashes, v4_dataset_hash)
        if (v4_run["variant"], v4_run["seed"]) != ("harmonic_gram", seed):
            raise AuditError(f"wrong v4 branch identity: {v4_marker}")
        v4_rows = v4._read_validation_predictions(v4_run)

        v6_dir = v6_root.resolve() / f"learned_power__seed-{seed}"
        v6_manifest, _checked = v6._verify_run(v6_dir, "learned_power", seed, v6_dataset_hash, v6_hashes)
        v6_run = {"variant": "learned_power", "seed": seed, "run_dir": v6_dir, "manifest": v6_manifest}
        v6_metrics = v5._read_validation(v6_run)

        v4_run["fusion_identity"] = v4_rows["validation_signature"]
        v4_run["fusion_prediction_path"] = v4_run["run_dir"] / v4_run["manifest"]["artifacts"]["validation_predictions"]["path"]
        v6_run["fusion_identity"] = v6_metrics["validation_signature"]
        v6_run["fusion_prediction_path"] = v6_dir / v6_manifest["artifacts"]["validation_predictions"]["path"]
        if v4_run["fusion_identity"] != v6_run["fusion_identity"]:
            raise AuditError(f"v4/v6 validation rows differ for seed {seed}")
        v4_runs[seed] = v4_run
        v6_runs[seed] = v6_run
    return v4_runs, v6_runs


def _seed_fusion(seed: int, v4_run: dict[str, Any], v6_run: dict[str, Any]) -> dict[str, Any]:
    path4, path6 = v4_run["fusion_prediction_path"], v6_run["fusion_prediction_path"]
    rows4, identity4 = _read_rows(path4, "harmonic_gram", seed)
    rows6, identity6 = _read_rows(path6, "learned_power", seed)
    if identity4 != identity6:
        raise AuditError(f"sample_index/distance/true-label/channel-map mismatch for seed {seed}")
    labels, distances = [], []
    pred4, pred6, fused = [], [], []
    counts = {"both_wrong": 0, "exactly_one_wrong": 0, "both_correct": 0}
    for row4, row6, (_, distance, true_name, _) in zip(rows4, rows6, identity4):
        true4, true6 = row4["true_direction"], row6["true_direction"]
        if true4 != true6:
            raise AuditError(f"true label mismatch for seed {seed}")
        y = DIRECTIONS.index(true_name)
        p4, p6 = _probabilities(row4, path4), _probabilities(row6, path6)
        own4, own6 = DIRECTIONS.index(row4["prediction"]), DIRECTIONS.index(row6["prediction"])
        if max(range(4), key=lambda i: p4[i]) != own4 or max(range(4), key=lambda i: p6[i]) != own6:
            raise AuditError(f"saved prediction does not match its probability argmax for seed {seed}")
        mean_probability = [(p4[i] + p6[i]) * 0.5 for i in range(4)]
        if not math.isclose(sum(mean_probability), 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise AuditError(f"fused probabilities do not sum to one for seed {seed}")
        fusion_prediction = max(range(4), key=lambda i: mean_probability[i])
        wrong4, wrong6 = own4 != y, own6 != y
        counts["both_wrong" if wrong4 and wrong6 else "exactly_one_wrong" if wrong4 or wrong6 else "both_correct"] += 1
        labels.append(y)
        distances.append(distance)
        pred4.append(own4)
        pred6.append(own6)
        fused.append(fusion_prediction)

    m4 = _metrics(labels, pred4, distances)
    m6 = _metrics(labels, pred6, distances)
    mf = _metrics(labels, fused, distances)
    return {
        "seed": seed,
        "v4_harmonic_gram_macro_f1": m4["macro_f1"],
        "v6_learned_power_macro_f1": m6["macro_f1"],
        "fused_macro_f1": mf["macro_f1"],
        "fused_per_class_recall": mf["per_class_recall"],
        "fused_by_position": mf["by_position"],
        "paired_fused_minus_v4": mf["macro_f1"] - m4["macro_f1"],
        "paired_fused_minus_v6": mf["macro_f1"] - m6["macro_f1"],
        "error_complementarity_counts": counts,
        "error_complementarity_rates": {key: value / 320.0 for key, value in counts.items()},
        "validation_rows": len(labels),
    }


def _aggregate(rows: list[dict[str, Any]], v4_root: Path, v6_root: Path) -> dict[str, Any]:
    by_arm = {
        "v4_harmonic_gram_macro_f1": [row["v4_harmonic_gram_macro_f1"] for row in rows],
        "v6_learned_power_macro_f1": [row["v6_learned_power_macro_f1"] for row in rows],
        "fused_macro_f1": [row["fused_macro_f1"] for row in rows],
        "fused_minus_v4": [row["paired_fused_minus_v4"] for row in rows],
        "fused_minus_v6": [row["paired_fused_minus_v6"] for row in rows],
    }
    categories = {
        key: v2._stats([row["error_complementarity_rates"][key] for row in rows])
        for key in ("both_wrong", "exactly_one_wrong", "both_correct")
    }
    positions = {
        str(position): v2._stats([row["fused_by_position"][str(position)]["macro_f1"] for row in rows])
        for position in POSITIONS
    }
    recalls = {
        direction: v2._stats([row["fused_per_class_recall"][direction] for row in rows])
        for direction in DIRECTIONS
    }
    return {
        "format": "c170-v4-v6-fixed-probability-fusion-validation-diagnostic",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "validation_only": True,
        "test_metrics_computed": False,
        "fusion": {"branches": ["v4 harmonic_gram", "v6 learned_power"], "seeds": list(SEEDS),
                   "pairing": "same seed across both branches", "seed_selection": False,
                   "probability_average": "fixed arithmetic 0.5/0.5 per aligned sample", "prediction": "argmax",
                   "temperature_or_weight_search": False,
                   "symmetry_boundary": "branch probabilities and their arithmetic average preserve exact C4 and ideal common first-harmonic U1 invariance; this does not establish physical continuous SO(2) rotation"},
        "verified_seed_count": len(rows),
        "v4_root": str(v4_root.resolve()),
        "v6_root": str(v6_root.resolve()),
        "score_statistics_population_sd": {key: v2._stats(values) for key, values in by_arm.items()},
        "fused_per_class_recall": recalls,
        "fused_macro_f1_by_position": positions,
        "error_complementarity_rates": categories,
        "per_seed": rows,
        "matched_shuffle_control": False,
        "interpretation_boundary": (
            "Exploratory late fusion on the fixed validation split; the five fixed seeds 17/29/43/61/89 are paired branch-by-branch, with no best-seed selection, fitted fusion weights, or temperature. "
            "All ten branch runs were verified by config/marker/artifact hashes, and each same-seed pair's sample index, distance, true label, and channel map aligned exactly. "
            "Each branch probability vector and its 0.5/0.5 average was checked for finite normalized probabilities. "
            "Both branches have exact C4 and ideal common first-harmonic U1 invariance, which arithmetic averaging preserves; this is not evidence of physical continuous rotation generalization, a novelty claim, or a formal performance gate. "
            "The descriptive fusion remains below the 0.80 macro-F1 / 0.65-per-class validation gate (RIGHT recall is 0.595); no additional training is warranted for this diagnostic. "
            "There is no matched learned_power_shuffle branch, so the matched fusion-null control is absent. "
            "Both-wrong/exactly-one-wrong/both-correct counts are descriptive error complementarity only; no oracle upper bound is computed. "
            "The dataset NPZ was not opened; no checkpoint was deserialized; test waveforms and test metrics were not accessed."
        ),
    }


def _markdown(summary: dict[str, Any]) -> str:
    stats = summary["score_statistics_population_sd"]
    lines = [
        "# Fixed v4/v6 Validation Probability Fusion Diagnostic", "",
        "Exploratory validation-only late fusion: fixed 0.5/0.5 probability average, then argmax; no weight or temperature search.",
        "The fixed seeds 17/29/43/61/89 are paired branch-by-branch; no best seed was selected. All 320 rows per seed passed identity, probability, marker, config, and artifact-hash checks. Test metrics were not computed.", "",
        "| Metric | Mean ± population SD |",
        "|---|---:|",
        f"| v4 harmonic_gram macro-F1 | {stats['v4_harmonic_gram_macro_f1']['mean']:.4f} ± {stats['v4_harmonic_gram_macro_f1']['std_population']:.4f} |",
        f"| v6 learned_power macro-F1 | {stats['v6_learned_power_macro_f1']['mean']:.4f} ± {stats['v6_learned_power_macro_f1']['std_population']:.4f} |",
        f"| Fused macro-F1 | {stats['fused_macro_f1']['mean']:.4f} ± {stats['fused_macro_f1']['std_population']:.4f} |",
        f"| Fused − v4 paired macro-F1 | {stats['fused_minus_v4']['mean']:+.4f} ± {stats['fused_minus_v4']['std_population']:.4f} |",
        f"| Fused − v6 paired macro-F1 | {stats['fused_minus_v6']['mean']:+.4f} ± {stats['fused_minus_v6']['std_population']:.4f} |",
        "",
        "| Fused recall | Mean ± population SD |",
        "|---|---:|",
    ]
    for direction, stat in summary["fused_per_class_recall"].items():
        lines.append(f"| {direction} | {stat['mean']:.4f} ± {stat['std_population']:.4f} |")
    lines.extend(["", "| Position (mm) | Fused macro-F1 |", "|---:|---:|"])
    for position, stat in summary["fused_macro_f1_by_position"].items():
        lines.append(f"| {position} | {stat['mean']:.4f} ± {stat['std_population']:.4f} |")
    lines.extend(["", "## Error Complementarity", "", "Mean fraction of the 320 paired validation examples per seed; categories are mutually exclusive.", "", "| Category | Mean ± population SD |", "|---|---:|"])
    for category, stat in summary["error_complementarity_rates"].items():
        lines.append(f"| {category} | {stat['mean']:.4f} ± {stat['std_population']:.4f} |")
    lines.extend(["", summary["interpretation_boundary"], ""])
    return "\n".join(lines)


def _write(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "fusion_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "fusion_validation_summary.md").write_text(_markdown(summary), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent / "outputs"
    parser.add_argument("--v4-root", type=Path, default=base / "c170_harmonic_phase_v4_formal_cuda_20260929")
    parser.add_argument("--v6-root", type=Path, default=base / "c170_temporal_phase_invariant_v6_formal_cuda_20260929")
    parser.add_argument("--output-dir", type=Path, default=base / "c170_v4_v6_power_fusion_validation_20260929")
    args = parser.parse_args()
    try:
        runs4, runs6 = _verify_branch_runs(args.v4_root, args.v6_root)
        rows = [_seed_fusion(seed, runs4[seed], runs6[seed]) for seed in SEEDS]
        summary = _aggregate(rows, args.v4_root, args.v6_root)
        _write(args.output_dir.expanduser().resolve(), summary)
        stats = summary["score_statistics_population_sd"]
        print(f"FUSION_VALIDATION_COMPLETE path={args.output_dir.expanduser().resolve()}")
        print(f"fused_macro_f1={stats['fused_macro_f1']['mean']:.6f} ± {stats['fused_macro_f1']['std_population']:.6f}")
        for direction, stat in summary["fused_per_class_recall"].items():
            print(f"fused_recall_{direction.lower()}={stat['mean']:.6f} ± {stat['std_population']:.6f}")
        print(f"fused_minus_v4={stats['fused_minus_v4']['mean']:+.6f} ± {stats['fused_minus_v4']['std_population']:.6f}")
        print(f"fused_minus_v6={stats['fused_minus_v6']['mean']:+.6f} ± {stats['fused_minus_v6']['std_population']:.6f}")
        return 0
    except (AuditError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
