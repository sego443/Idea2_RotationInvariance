#!/usr/bin/env python3
"""Run the frozen round-1 held-out-angle rotation feasibility experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rotation_poc import (  # noqa: E402
    angle_split,
    calibrate_windows,
    filter_windows,
    harmonic_invariants,
    macro_metrics,
    parse_movement_filename,
    predict_temporal_model,
    read_openbci,
    segment_five_strokes,
    select_recordings,
    train_temporal_model,
)


GESTURES = ["up", "down", "left", "right"]
GESTURE_TO_ID = {gesture: index for index, gesture in enumerate(GESTURES)}
SEEDS = [17, 29, 43, 61, 89]
MODEL_KINDS = ["raw", "orbit", "harmonic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    return parser.parse_args()


def longest_constant_run(values: np.ndarray) -> int:
    equal = np.diff(values) == 0
    best = current = 0
    for is_equal in equal:
        current = current + 1 if is_equal else 0
        best = max(best, current)
    return best + 1


def build_dataset(session_dir: Path) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    strokes: list[np.ndarray] = []
    rests: list[np.ndarray] = []
    labels: list[int] = []
    angles: list[int] = []
    blocks: list[int] = []
    repetitions: list[int] = []
    sources: list[str] = []
    qc_rows: list[dict[str, object]] = []

    for path in select_recordings(session_dir, set(range(1, 49))):
        parsed = parse_movement_filename(path)
        if parsed is None:
            raise RuntimeError(f"selected path no longer parses: {path}")
        block, angle, gesture = parsed
        samples, metadata = read_openbci(path)
        if metadata["sample_rate_hz"] != 250 or metadata["declared_channels"] != 8:
            raise ValueError(f"unexpected acquisition metadata in {path}: {metadata}")
        segmented = segment_five_strokes(samples, sample_rate=250, window_samples=500)
        centers_s = segmented.centers / 250.0
        intervals_s = np.diff(centers_s)
        flat_fraction = np.mean(np.diff(samples, axis=0) == 0, axis=0)
        qc_rows.append(
            {
                "block": f"B{block:03d}",
                "angle_deg": angle,
                "gesture": gesture,
                "split": angle_split(angle),
                "source": path.name,
                "samples": len(samples),
                "duration_s": len(samples) / 250.0,
                "protocol_duration_ok": 20.0 <= len(samples) / 250.0 <= 35.0,
                "centers_s": " ".join(f"{value:.3f}" for value in centers_s),
                "interval_cv": float(np.std(intervals_s) / np.mean(intervals_s)),
                "segmentation_score": segmented.score,
                "max_flat_fraction": float(flat_fraction.max()),
                "max_constant_run": max(longest_constant_run(samples[:, channel]) for channel in range(4)),
                "range_ch0": float(np.ptp(samples[:, 0])),
                "range_ch1": float(np.ptp(samples[:, 1])),
                "range_ch2": float(np.ptp(samples[:, 2])),
                "range_ch3": float(np.ptp(samples[:, 3])),
            }
        )
        for repetition in range(5):
            strokes.append(segmented.strokes[repetition])
            rests.append(segmented.rest_windows[repetition])
            labels.append(GESTURE_TO_ID[gesture])
            angles.append(angle)
            blocks.append(block)
            repetitions.append(repetition + 1)
            sources.append(path.name)

    dataset = {
        "strokes": np.stack(strokes),
        "rests": np.stack(rests),
        "labels": np.asarray(labels, dtype=np.int64),
        "angles": np.asarray(angles, dtype=np.int64),
        "blocks": np.asarray(blocks, dtype=np.int64),
        "repetitions": np.asarray(repetitions, dtype=np.int64),
        "sources": np.asarray(sources),
    }
    return dataset, qc_rows


def preprocess(windows: np.ndarray, train_mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    filtered = filter_windows(windows, sample_rate=250)
    _, sensor_scale = calibrate_windows(filtered[train_mask])
    raw, _ = calibrate_windows(filtered, scale=sensor_scale)
    harmonic = harmonic_invariants(raw)
    harmonic_center = np.median(harmonic[train_mask], axis=(0, 1))
    harmonic_scale = np.median(
        np.abs(harmonic[train_mask] - harmonic_center), axis=(0, 1)
    )
    harmonic_scale = np.maximum(harmonic_scale, np.finfo(np.float32).eps)
    harmonic = (harmonic - harmonic_center) / harmonic_scale
    return (
        {"raw": raw, "orbit": raw, "harmonic": harmonic.astype(np.float32)},
        {
            "sensor_scale": sensor_scale,
            "harmonic_center": harmonic_center,
            "harmonic_scale": harmonic_scale,
        },
    )


def metrics_by_angle(labels: np.ndarray, predictions: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    return {
        str(int(angle)): float(np.mean(labels[angles == angle] == predictions[angles == angle]))
        for angle in sorted(set(angles.tolist()))
    }


def run_models(
    representations: dict[str, np.ndarray],
    labels: np.ndarray,
    angles: np.ndarray,
    output_dir: Path,
    max_epochs: int,
    patience: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, list[float]]]:
    split_names = np.asarray([angle_split(int(angle)) for angle in angles])
    masks = {name: split_names == name for name in ["train", "val", "test"]}
    results: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    val_scores: dict[str, list[float]] = {kind: [] for kind in MODEL_KINDS}
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    for model_kind in MODEL_KINDS:
        samples = representations[model_kind]
        for seed in SEEDS:
            trained = train_temporal_model(
                samples[masks["train"]],
                labels[masks["train"]],
                samples[masks["val"]],
                labels[masks["val"]],
                model_kind=model_kind,
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
            )
            torch.save(
                {
                    "model_kind": model_kind,
                    "seed": seed,
                    "best_epoch": trained.best_epoch,
                    "state_dict": trained.model.state_dict(),
                },
                checkpoint_dir / f"{model_kind}_seed{seed}.pt",
            )
            for split_name, mask in masks.items():
                predictions = predict_temporal_model(trained.model, samples[mask])
                split_metrics = macro_metrics(labels[mask], predictions, n_classes=4)
                results.append(
                    {
                        "experiment": "primary_stroke",
                        "model": model_kind,
                        "seed": seed,
                        "split": split_name,
                        "best_epoch": trained.best_epoch,
                        **split_metrics,
                        "accuracy_by_angle": metrics_by_angle(labels[mask], predictions, angles[mask]),
                    }
                )
                if split_name == "val":
                    val_scores[model_kind].append(float(split_metrics["macro_f1"]))
                sample_indices = np.flatnonzero(mask)
                for sample_index, prediction in zip(sample_indices, predictions, strict=True):
                    prediction_rows.append(
                        {
                            "experiment": "primary_stroke",
                            "model": model_kind,
                            "seed": seed,
                            "sample_index": int(sample_index),
                            "split": split_name,
                            "angle_deg": int(angles[sample_index]),
                            "truth": GESTURES[int(labels[sample_index])],
                            "prediction": GESTURES[int(prediction)],
                        }
                    )
    return results, prediction_rows, val_scores


def run_control(
    experiment: str,
    model_kind: str,
    samples: np.ndarray,
    labels: np.ndarray,
    angles: np.ndarray,
    max_epochs: int,
    patience: int,
    shuffle_labels: bool,
) -> list[dict[str, object]]:
    split_names = np.asarray([angle_split(int(angle)) for angle in angles])
    masks = {name: split_names == name for name in ["train", "val", "test"]}
    results: list[dict[str, object]] = []
    for seed in SEEDS:
        train_labels = labels[masks["train"]].copy()
        val_labels = labels[masks["val"]].copy()
        if shuffle_labels:
            rng = np.random.default_rng(seed + 10_000)
            rng.shuffle(train_labels)
            rng.shuffle(val_labels)
        trained = train_temporal_model(
            samples[masks["train"]],
            train_labels,
            samples[masks["val"]],
            val_labels,
            model_kind=model_kind,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
        )
        predictions = predict_temporal_model(trained.model, samples[masks["test"]])
        results.append(
            {
                "experiment": experiment,
                "model": model_kind,
                "seed": seed,
                "split": "test",
                "best_epoch": trained.best_epoch,
                **macro_metrics(labels[masks["test"]], predictions, n_classes=4),
                "accuracy_by_angle": metrics_by_angle(
                    labels[masks["test"]], predictions, angles[masks["test"]]
                ),
            }
        )
    return results


def summarize(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    keys = sorted({(str(row["experiment"]), str(row["model"]), str(row["split"])) for row in results})
    for experiment, model, split in keys:
        rows = [
            row
            for row in results
            if row["experiment"] == experiment and row["model"] == model and row["split"] == split
        ]
        f1 = np.asarray([row["macro_f1"] for row in rows], dtype=float)
        accuracy = np.asarray([row["accuracy"] for row in rows], dtype=float)
        min_recall = np.asarray([min(row["per_class_recall"]) for row in rows], dtype=float)
        summary[f"{experiment}/{model}/{split}"] = {
            "n_seeds": len(rows),
            "macro_f1_mean": float(f1.mean()),
            "macro_f1_std": float(f1.std(ddof=1)) if len(f1) > 1 else 0.0,
            "macro_f1_min": float(f1.min()),
            "macro_f1_max": float(f1.max()),
            "accuracy_mean": float(accuracy.mean()),
            "minimum_class_recall_mean": float(min_recall.mean()),
            "minimum_class_recall_min": float(min_recall.min()),
        }
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(results: list[dict[str, object]], output_dir: Path) -> None:
    primary = [row for row in results if row["experiment"] == "primary_stroke" and row["split"] == "test"]
    controls = [row for row in results if row["experiment"] != "primary_stroke"]
    labels = MODEL_KINDS + sorted({str(row["experiment"]) for row in controls})
    groups = []
    for label in labels:
        if label in MODEL_KINDS:
            groups.append([float(row["macro_f1"]) for row in primary if row["model"] == label])
        else:
            groups.append([float(row["macro_f1"]) for row in controls if row["experiment"] == label])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.axhline(0.80, color="tab:green", linestyle=":", linewidth=1, label="round-1 target")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Held-out-angle macro F1")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "held_out_angle_performance.png", dpi=180)
    plt.close(fig)


def build_report(
    session_dir: Path,
    dataset: dict[str, np.ndarray],
    qc_rows: list[dict[str, object]],
    summary: dict[str, dict[str, float]],
    selected_model: str,
) -> str:
    short_files = sum(not bool(row["protocol_duration_ok"]) for row in qc_rows)
    rejected_files = len(list(session_dir.glob("*.txt.rejected")))
    unlabelled = [path.name for path in session_dir.glob("*.txt") if parse_movement_filename(path) is None and "REST_20S" not in path.name]
    lines = [
        "# Round-1 Rotation-Invariance Feasibility Result",
        "",
        "## Evidence boundary",
        "",
        "Single participant, single session, single donning, four pre-segmented directional strokes. "
        "Training angles were 0/60/120/180/240/300 degrees; validation angles were 30/150/270; "
        "test angles were 90/210/330 and were not used for model selection.",
        "",
        "## Data audit",
        "",
        f"- 48 accepted movement files produced {len(dataset['labels'])} strokes; class counts: "
        f"{np.bincount(dataset['labels'], minlength=4).tolist()}.",
        f"- Rejected recordings retained: {rejected_files}; unlabelled accepted-looking extras: {unlabelled or 'none'}.",
        f"- Files outside the planned 20-35 s duration: {short_files}/48. All 48 nevertheless yielded five "
        "regular action centers; this is retained as a protocol deviation.",
        f"- Maximum action-interval CV: {max(float(row['interval_cv']) for row in qc_rows):.3f}; "
        f"maximum identical-sample run: {max(int(row['max_constant_run']) for row in qc_rows)} rows.",
        "- Only EXG channels 0-3 were used. Raw files were not modified.",
        "",
        "## Model results",
        "",
        "Metrics below are mean over five frozen random seeds.",
        "",
        "| Experiment/model/split | Macro F1 | Accuracy | Mean minimum recall |",
        "|---|---:|---:|---:|",
    ]
    for key, values in summary.items():
        lines.append(
            f"| {key} | {values['macro_f1_mean']:.3f} +/- {values['macro_f1_std']:.3f} | "
            f"{values['accuracy_mean']:.3f} | {values['minimum_class_recall_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Validation-selected model family: `{selected_model}`.",
            "",
            "## Decision rules",
            "",
            "Round-1 support requires test macro F1 >= 0.80, every class recall >= 0.65, a matched shuffled-label "
            "control near 0.25, and no strong classification from pre-action rest windows. This round cannot establish "
            "continuous-angle robustness because the frozen irregular-angle challenge set has not yet been collected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    dataset, qc_rows = build_dataset(args.session_dir.resolve())
    split_names = np.asarray([angle_split(int(angle)) for angle in dataset["angles"]])
    train_mask = split_names == "train"
    stroke_representations, stroke_preprocessing = preprocess(dataset["strokes"], train_mask)
    rest_representations, rest_preprocessing = preprocess(dataset["rests"], train_mask)

    np.savez_compressed(output_dir / "segmented_dataset.npz", **dataset)
    write_csv(output_dir / "recording_qc.csv", qc_rows)
    primary_results, prediction_rows, val_scores = run_models(
        stroke_representations,
        dataset["labels"],
        dataset["angles"],
        output_dir,
        args.max_epochs,
        args.patience,
    )
    selected_model = max(MODEL_KINDS, key=lambda kind: float(np.mean(val_scores[kind])))
    control_results = run_control(
        "shuffled_label_control",
        selected_model,
        stroke_representations[selected_model],
        dataset["labels"],
        dataset["angles"],
        args.max_epochs,
        args.patience,
        shuffle_labels=True,
    )
    rest_models = ["raw"] if selected_model == "raw" else ["raw", selected_model]
    for model_kind in rest_models:
        control_results.extend(
            run_control(
                "rest_only_control",
                model_kind,
                rest_representations[model_kind],
                dataset["labels"],
                dataset["angles"],
                args.max_epochs,
                args.patience,
                shuffle_labels=False,
            )
        )
    results = primary_results + control_results
    summary = summarize(results)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "session_dir": str(args.session_dir.resolve()),
                "frozen_splits": {
                    "train_angles": [0, 60, 120, 180, 240, 300],
                    "val_angles": [30, 150, 270],
                    "test_angles": [90, 210, 330],
                },
                "seeds": SEEDS,
                "selected_model_by_validation": selected_model,
                "summary": summary,
                "runs": results,
                "preprocessing": {
                    "stroke": {key: value.tolist() for key, value in stroke_preprocessing.items()},
                    "rest": {key: value.tolist() for key, value in rest_preprocessing.items()},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(output_dir / "predictions.csv", prediction_rows)
    plot_results(results, output_dir)
    report = build_report(args.session_dir.resolve(), dataset, qc_rows, summary, selected_model)
    (output_dir / "experiment_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
