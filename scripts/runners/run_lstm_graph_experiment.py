#!/usr/bin/env python3
"""Compare independent-LSTM graph fusion with the Round-1 CNN baselines."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import resample_poly
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from rotation_poc import (  # noqa: E402
    angle_split,
    calibrate_windows,
    filter_windows,
    macro_metrics,
    predict_temporal_model,
    train_temporal_model,
)


SEEDS = [17, 29, 43, 61, 89]
TUNING_SEEDS = [17, 29]
BASELINE_DIR = PROJECT_DIR / "outputs" / "round1_rotation_poc_2026-08-27"
GESTURES = ["up", "down", "left", "right"]


def prepare_model_inputs(windows: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Round-1 filtering/calibration, then anti-aliased 4:1 resampling."""
    train_mask = np.asarray([angle_split(int(angle)) == "train" for angle in angles])
    filtered = filter_windows(windows, sample_rate=250)
    _, scale = calibrate_windows(filtered[train_mask])
    normalized, _ = calibrate_windows(filtered, scale=scale)
    reduced = resample_poly(normalized, up=1, down=4, axis=1)
    return reduced.astype(np.float32), scale


def split_masks(angles: np.ndarray) -> dict[str, np.ndarray]:
    labels = np.asarray([angle_split(int(angle)) for angle in angles])
    return {split: labels == split for split in ("train", "val", "test")}


def by_angle_accuracy(labels: np.ndarray, predictions: np.ndarray, angles: np.ndarray) -> dict[str, float]:
    return {
        str(int(angle)): float(np.mean(predictions[angles == angle] == labels[angles == angle]))
        for angle in sorted(set(angles.tolist()))
    }


def tune(
    stage: str,
    candidates: list[tuple[str, dict[str, object]]],
    samples: np.ndarray,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    max_epochs: int,
    patience: int,
) -> tuple[tuple[str, dict[str, object]], list[dict[str, object]]]:
    trials: list[dict[str, object]] = []
    for model_kind, config in candidates:
        seed_rows = []
        for seed in TUNING_SEEDS:
            result = train_temporal_model(
                samples[masks["train"]], labels[masks["train"]],
                samples[masks["val"]], labels[masks["val"]],
                model_kind=model_kind, seed=seed, max_epochs=max_epochs,
                patience=patience, model_config=config,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "best_epoch": result.best_epoch,
                    "val_macro_f1": float(result.best_val_macro_f1),
                    "val_loss": float(result.best_val_loss),
                }
            )
        trial = {
            "stage": stage,
            "model_kind": model_kind,
            "config": config,
            "mean_val_macro_f1": float(np.mean([row["val_macro_f1"] for row in seed_rows])),
            "mean_val_loss": float(np.mean([row["val_loss"] for row in seed_rows])),
            "seeds": seed_rows,
        }
        trials.append(trial)
        print(
            f"{stage} {model_kind} {config}: "
            f"val F1={trial['mean_val_macro_f1']:.3f}, loss={trial['mean_val_loss']:.3f}",
            flush=True,
        )
    winner = max(trials, key=lambda row: (row["mean_val_macro_f1"], -row["mean_val_loss"]))
    return (str(winner["model_kind"]), dict(winner["config"])), trials


def evaluate_family(
    name: str,
    model_kind: str,
    config: dict[str, object] | None,
    samples: np.ndarray,
    labels: np.ndarray,
    angles: np.ndarray,
    masks: dict[str, np.ndarray],
    output_dir: Path,
    max_epochs: int,
    patience: int,
    shuffled_labels: bool = False,
    save_checkpoints: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    predictions_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        train_labels = labels[masks["train"]].copy()
        val_labels = labels[masks["val"]].copy()
        if shuffled_labels:
            rng = np.random.default_rng(seed + 10_000)
            rng.shuffle(train_labels)
            rng.shuffle(val_labels)
        trained = train_temporal_model(
            samples[masks["train"]], train_labels,
            samples[masks["val"]], val_labels,
            model_kind=model_kind, seed=seed, max_epochs=max_epochs,
            patience=patience, model_config=config,
        )
        if save_checkpoints:
            torch.save(
                {
                    "model_kind": model_kind,
                    "model_config": config,
                    "seed": seed,
                    "best_epoch": trained.best_epoch,
                    "state_dict": trained.model.state_dict(),
                },
                output_dir / f"{name}_seed{seed}.pt",
            )
        for split in ("train", "val", "test"):
            if shuffled_labels and split != "test":
                continue
            mask = masks[split]
            predictions = predict_temporal_model(trained.model, samples[mask])
            metrics = macro_metrics(labels[mask], predictions, n_classes=4)
            rows.append(
                {
                    "experiment": name,
                    "model_kind": model_kind,
                    "seed": seed,
                    "split": split,
                    "best_epoch": trained.best_epoch,
                    **metrics,
                    "accuracy_by_angle": by_angle_accuracy(labels[mask], predictions, angles[mask]),
                }
            )
            if split == "test":
                for sample_index, pred in zip(np.flatnonzero(mask), predictions, strict=True):
                    predictions_rows.append(
                        {
                            "experiment": name,
                            "seed": seed,
                            "sample_index": int(sample_index),
                            "angle_deg": int(angles[sample_index]),
                            "truth": GESTURES[int(labels[sample_index])],
                            "prediction": GESTURES[int(pred)],
                        }
                    )
        print(f"final {name} seed={seed} epoch={trained.best_epoch}", flush=True)
    return rows, predictions_rows


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for name in sorted({str(row["experiment"]) for row in rows}):
        for split in ("train", "val", "test"):
            selected = [row for row in rows if row["experiment"] == name and row["split"] == split]
            if not selected:
                continue
            f1 = np.asarray([row["macro_f1"] for row in selected], dtype=float)
            acc = np.asarray([row["accuracy"] for row in selected], dtype=float)
            recalls = np.asarray([row["per_class_recall"] for row in selected], dtype=float)
            all_angles = sorted(selected[0]["accuracy_by_angle"], key=int)
            summary[f"{name}/{split}"] = {
                "macro_f1_mean": float(f1.mean()),
                "macro_f1_std": float(f1.std(ddof=1)),
                "accuracy_mean": float(acc.mean()),
                "per_class_recall_mean": recalls.mean(axis=0).tolist(),
                "minimum_class_recall_mean": float(recalls.min(axis=1).mean()),
                "accuracy_by_angle_mean": {
                    angle: float(np.mean([row["accuracy_by_angle"][angle] for row in selected]))
                    for angle in all_angles
                },
            }
    return summary


def write_report(
    output_dir: Path,
    winner: tuple[str, dict[str, object]],
    trials: list[dict[str, object]],
    summary: dict[str, dict[str, object]],
    baseline: dict[str, object],
) -> None:
    baseline_summary = baseline["summary"]
    graph_f1 = float(summary["graph_selected/test"]["macro_f1_mean"])
    mean_f1 = float(summary["mean_ablation/test"]["macro_f1_mean"])
    matched_cnn_f1 = float(summary["resampled_raw_cnn/test"]["macro_f1_mean"])
    previous_orbit_f1 = float(baseline_summary["primary_stroke/orbit/test"]["macro_f1_mean"])
    up_recall = float(summary["graph_selected/test"]["per_class_recall_mean"][0])
    lines = [
        "# Four independent LSTMs with graph-attention fusion: round-one re-evaluation",
        "",
        "This run reuses the 240 round-one windows from 2026-08-27, four physical channels, and the fixed angle split.",
        "The original test-angle results are already known; this comparison is exploratory and requires new W02 data for independent confirmation.",
        "",
        "## Conclusion",
        "",
        f"After staged tuning, the new model has unseen-angle macro-F1 {graph_f1:.3f}, "
        f"lower than the length-matched CNN by {matched_cnn_f1 - graph_f1:.3f}, "
        f"lower than the original cyclic-orbit CNN by {previous_orbit_f1 - graph_f1:.3f}.",
        f"The same four LSTMs with mean fusion achieve {mean_f1:.3f}; "
        f"graph attention scores lower in this test by {mean_f1 - graph_f1:.3f}.",
        f"The weakest class, up, has mean recall only {up_recall:.3f}."
        "The model fits the training samples, but does not improve cross-angle recognition or meet the predefined gate.",
        "",
        "## Model and selection procedure",
        "",
        "Each channel has an independent bidirectional LSTM. The four nodes interact through scaled dot-product attention at each time step;",
        "node pooling forms a temporal feature h(t), classified by an MLP or a small temporal CNN.",
        "Angle labels are used only for splits and statistics; angle is not a model input.",
        "First compare LSTM hidden sizes 8/16 and depths 1/2 using validation angles;",
        "then compare 1/2 attention heads, fully connected/ring edges, and MLP/CNN classification heads.",
        "Use two fixed seeds per tuning candidate; retrain the selected combination with the original five seeds.",
        f"Selected on validation angles: `{winner[0]}`, parameters `{winner[1]}`. Evaluated {len(trials)} candidate configurations.",
        "",
        "Reuse the original 0.5–40 Hz filtering and training-angle channel scales;",
        "antialias and resample the 500-point movement windows 4:1 to 125 points before the LSTM.",
        "The length-matched CNN control also uses the 125-point input.",
        "",
        "## Five-seed results",
        "",
        "| Model | Train macro-F1 | Validation macro-F1 | Unseen-angle macro-F1 | Test accuracy | Minimum class recall on test |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    display = [
        ("New model: LSTM + graph attention", "graph_selected"),
        ("Same LSTM + mean fusion", "mean_ablation"),
        ("Length-matched raw CNN", "resampled_raw_cnn"),
    ]
    for label, name in display:
        def metric(split: str, key: str) -> float:
            return float(summary[f"{name}/{split}"][key])
        lines.append(
            f"| {label} | {metric('train','macro_f1_mean'):.3f} | "
            f"{metric('val','macro_f1_mean'):.3f} | "
            f"{metric('test','macro_f1_mean'):.3f} ± {metric('test','macro_f1_std'):.3f} | "
            f"{metric('test','accuracy_mean'):.3f} | "
            f"{metric('test','minimum_class_recall_mean'):.3f} |"
        )
    for label, key in [
        ("Original raw CNN, 500 points", "primary_stroke/raw"),
        ("Original cyclic-orbit CNN, 500 points", "primary_stroke/orbit"),
        ("Original harmonic CNN, 500 points", "primary_stroke/harmonic"),
    ]:
        train = baseline_summary[f"{key}/train"]
        val = baseline_summary[f"{key}/val"]
        test = baseline_summary[f"{key}/test"]
        lines.append(
            f"| {label} | {train['macro_f1_mean']:.3f} | {val['macro_f1_mean']:.3f} | "
            f"{test['macro_f1_mean']:.3f} ± {test['macro_f1_std']:.3f} | "
            f"{test['accuracy_mean']:.3f} | {test['minimum_class_recall_mean']:.3f} |"
        )
    selected = summary["graph_selected/test"]
    lines.extend(
        [
            "",
            "Accuracy by test angle: " + ", ".join(
                f"{angle}°={value:.3f}"
                for angle, value in selected["accuracy_by_angle_mean"].items()
            ) + ".",
            "Mean recall (up/down/left/right): " +
            ", ".join(f"{value:.3f}" for value in selected["per_class_recall_mean"]) + ".",
            "",
            "## Controls and interpretation limits",
            "",
            f"Shuffled-label test macro-F1 of the new model: {summary['shuffled_label/test']['macro_f1_mean']:.3f}; "
            f"Rest-window test macro-F1: {summary['rest_only/test']['macro_f1_mean']:.3f}.",
            "The predefined round-one gate is unseen regular-angle macro-F1 >= 0.80 and recall >= 0.65 for every class.",
            "Graph-attention weights are internal model coefficients and cannot be interpreted directly as physical sensor contributions.",
            "The 12 angles were collected sequentially; baseline drift and wearing state remain limitations.",
            "",
            "Per-configuration validation scores, per-seed confusion matrices, per-sample predictions, and checkpoints are saved in this directory.",
            "",
        ]
    )
    (output_dir / "comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=BASELINE_DIR / "segmented_dataset.npz")
    parser.add_argument("--baseline-metrics", type=Path, default=BASELINE_DIR / "metrics.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tune-epochs", type=int, default=100)
    parser.add_argument("--tune-patience", type=int, default=18)
    parser.add_argument("--final-epochs", type=int, default=220)
    parser.add_argument("--final-patience", type=int, default=35)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    with np.load(args.dataset) as dataset:
        strokes = dataset["strokes"].copy()
        rests = dataset["rests"].copy()
        labels = dataset["labels"].copy()
        angles = dataset["angles"].copy()
    if strokes.shape != (240, 500, 4) or sorted(set(angles.tolist())) != list(range(0, 360, 30)):
        raise ValueError("dataset does not match the frozen Round-1 240-stroke corpus")
    masks = split_masks(angles)
    samples, sensor_scale = prepare_model_inputs(strokes, angles)
    baseline = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))

    lstm_candidates = [
        ("lstm_graph_cnn", {"hidden_size": hidden, "num_layers": layers,
                             "graph_heads": 1, "graph_topology": "full"})
        for hidden, layers in product((8, 16), (1, 2))
    ]
    (lstm_winner, lstm_trials) = tune(
        "lstm", lstm_candidates, samples, labels, masks, args.tune_epochs, args.tune_patience
    )
    lstm_config = lstm_winner[1]
    graph_candidates = [
        (f"lstm_graph_{head}", {"hidden_size": lstm_config["hidden_size"],
                                 "num_layers": lstm_config["num_layers"],
                                 "graph_heads": graph_heads, "graph_topology": topology})
        for graph_heads, topology, head in product((1, 2), ("full", "ring"), ("mlp", "cnn"))
    ]
    (winner, graph_trials) = tune(
        "graph", graph_candidates, samples, labels, masks, args.tune_epochs, args.tune_patience
    )
    print(f"VALIDATION SELECTED {winner}", flush=True)

    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for name, kind, config, inputs in [
        ("graph_selected", winner[0], winner[1], samples),
        ("mean_ablation", winner[0].replace("_graph_", "_mean_"),
         {"hidden_size": winner[1]["hidden_size"], "num_layers": winner[1]["num_layers"]}, samples),
        ("resampled_raw_cnn", "raw", None, samples),
    ]:
        result_rows, result_predictions = evaluate_family(
            name, kind, config, inputs, labels, angles, masks, args.output_dir,
            args.final_epochs, args.final_patience, save_checkpoints=True,
        )
        rows.extend(result_rows)
        predictions.extend(result_predictions)

    shuffled_rows, shuffled_predictions = evaluate_family(
        "shuffled_label", winner[0], winner[1], samples, labels, angles, masks,
        args.output_dir, args.final_epochs, args.final_patience, shuffled_labels=True,
    )
    rows.extend(shuffled_rows)
    predictions.extend(shuffled_predictions)
    rest_samples, rest_scale = prepare_model_inputs(rests, angles)
    rest_rows, rest_predictions = evaluate_family(
        "rest_only", winner[0], winner[1], rest_samples, labels, angles, masks,
        args.output_dir, args.final_epochs, args.final_patience,
    )
    rows.extend(rest_rows)
    predictions.extend(rest_predictions)

    summary = summarize(rows)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset.resolve()),
                "baseline_metrics": str(args.baseline_metrics.resolve()),
                "split_angles": {"train": [0,60,120,180,240,300], "val": [30,150,270], "test": [90,210,330]},
                "selection": "two-seed validation macro F1, then validation loss tie-break",
                "tuning_seeds": TUNING_SEEDS,
                "final_seeds": SEEDS,
                "tuning_budget": {"epochs": args.tune_epochs, "patience": args.tune_patience},
                "final_budget": {"epochs": args.final_epochs, "patience": args.final_patience},
                "selected_model_kind": winner[0],
                "selected_model_config": winner[1],
                "sensor_scale": sensor_scale.tolist(),
                "rest_sensor_scale": rest_scale.tolist(),
                "input_length": int(samples.shape[1]),
                "tuning_trials": lstm_trials + graph_trials,
                "summary": summary,
                "runs": rows,
            },
            indent=2,
        ), encoding="utf-8",
    )
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    write_report(args.output_dir, winner, lstm_trials + graph_trials, summary, baseline)
    print((args.output_dir / "comparison_report.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
