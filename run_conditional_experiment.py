#!/usr/bin/env python3
"""Test gesture-conditioned angle and angle-conditioned gesture prediction."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from conditional_rotation import (  # noqa: E402
    ConditionalRotationModel,
    angle_metrics,
    angle_vector_loss,
    decode_angle_deg,
)
from rotation_poc import macro_metrics  # noqa: E402
from run_lstm_graph_experiment import prepare_model_inputs, split_masks  # noqa: E402


GESTURES = ("up", "down", "left", "right")
MODES = ("angle", "gesture", "joint")
BACKBONES = ("cnn", "graph")
TUNING_SEEDS = (17, 29)
FINAL_SEEDS = (17, 29, 43, 61, 89)
DATASET = PROJECT_DIR / "outputs" / "round1_rotation_poc_2026-08-27" / "segmented_dataset.npz"
GRAPH_CONFIGS = (
    {"hidden_size": 8, "num_layers": 1, "graph_heads": 1, "graph_topology": "full"},
    {"hidden_size": 16, "num_layers": 1, "graph_heads": 2, "graph_topology": "ring"},
    {"hidden_size": 16, "num_layers": 2, "graph_heads": 2, "graph_topology": "full"},
    {"hidden_size": 8, "num_layers": 2, "graph_heads": 2, "graph_topology": "ring"},
)


@dataclass
class FitResult:
    model: ConditionalRotationModel
    best_epoch: int
    best_val_score: float
    best_val_loss: float


def evaluate_model(
    model: ConditionalRotationModel,
    samples: np.ndarray,
    gestures: np.ndarray,
    angles: np.ndarray,
    mode: str,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    model.eval()
    x = torch.from_numpy(np.asarray(samples, dtype=np.float32)).permute(0, 2, 1)
    y = torch.from_numpy(np.asarray(gestures, dtype=np.int64))
    a = torch.from_numpy(np.asarray(angles, dtype=np.float32))
    with torch.no_grad():
        if mode == "joint":
            angle_vectors, gesture_logits = model.predict_both(x, y, a)
        elif mode == "angle":
            angle_vectors = model.predict_angle(x, y)
        elif mode == "gesture":
            gesture_logits = model.predict_gesture(x, a)
        else:
            raise ValueError(f"unknown mode: {mode}")

    metrics: dict[str, float] = {}
    predictions: list[dict[str, float | int]] = [{} for _ in range(len(samples))]
    if mode in {"angle", "joint"}:
        predicted_angles = decode_angle_deg(angle_vectors).cpu().numpy()
        angle_result = angle_metrics(predicted_angles, angles)
        metrics.update({f"angle_{key}": value for key, value in angle_result.items()})
        metrics["angle_prediction_30deg_bins"] = int(len(np.unique(np.floor(predicted_angles / 30.0).astype(int) % 12)))
        for row, prediction in zip(predictions, predicted_angles, strict=True):
            row["predicted_angle_deg"] = float(prediction)
    if mode in {"gesture", "joint"}:
        predicted_gestures = gesture_logits.argmax(dim=1).cpu().numpy()
        gesture_result = macro_metrics(gestures, predicted_gestures, n_classes=4)
        metrics["gesture_macro_f1"] = float(gesture_result["macro_f1"])
        metrics["gesture_accuracy"] = float(gesture_result["accuracy"])
        metrics["gesture_min_recall"] = float(min(gesture_result["per_class_recall"]))
        metrics["gesture_unique_predictions"] = int(len(np.unique(predicted_gestures)))
        for row, prediction in zip(predictions, predicted_gestures, strict=True):
            row["predicted_gesture"] = int(prediction)
    return metrics, predictions


def evaluate_condition_shift(
    model: ConditionalRotationModel,
    samples: np.ndarray,
    gestures: np.ndarray,
    angles: np.ndarray,
    mode: str,
) -> dict[str, float]:
    """Post-hoc check that changing only the supplied condition changes predictions."""
    model.eval()
    x = torch.from_numpy(np.asarray(samples, dtype=np.float32)).permute(0, 2, 1)
    y = torch.from_numpy(np.asarray(gestures, dtype=np.int64))
    a = torch.from_numpy(np.asarray(angles, dtype=np.float32))
    with torch.no_grad():
        if mode == "angle":
            correct = decode_angle_deg(model.predict_angle(x, y)).cpu().numpy()
            wrong = decode_angle_deg(model.predict_angle(x, (y + 1) % 4)).cpu().numpy()
            return {
                "correct_mae_deg": angle_metrics(correct, angles)["mae_deg"],
                "wrong_gesture_mae_deg": angle_metrics(wrong, angles)["mae_deg"],
            }
        if mode == "gesture":
            correct = model.predict_gesture(x, a).argmax(dim=1).cpu().numpy()
            shifted = model.predict_gesture(x, torch.remainder(a + 90.0, 360.0)).argmax(dim=1).cpu().numpy()
            return {
                "correct_macro_f1": float(macro_metrics(gestures, correct, n_classes=4)["macro_f1"]),
                "shifted_angle_macro_f1": float(macro_metrics(gestures, shifted, n_classes=4)["macro_f1"]),
                "prediction_agreement": float(np.mean(correct == shifted)),
            }
    raise ValueError(f"unknown diagnostic mode: {mode}")


def validation_score(metrics: dict[str, float], mode: str) -> float:
    if mode == "angle":
        return -metrics["angle_mae_deg"] / 180.0
    if mode == "gesture":
        return metrics["gesture_macro_f1"]
    return 0.5 * (metrics["gesture_macro_f1"] + 1.0 - metrics["angle_mae_deg"] / 180.0)


def permute_train_label_pairs(
    gestures: np.ndarray, angles: np.ndarray, train_mask: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the condition/target pair intact while breaking its waveform alignment."""
    shuffled_gestures = np.asarray(gestures).copy()
    shuffled_angles = np.asarray(angles).copy()
    train_indices = np.flatnonzero(train_mask)
    source_indices = np.random.default_rng(seed + 10_000).permutation(train_indices)
    shuffled_gestures[train_indices] = shuffled_gestures[source_indices]
    shuffled_angles[train_indices] = shuffled_angles[source_indices]
    return shuffled_gestures, shuffled_angles


def fit_model(
    samples: np.ndarray,
    gestures: np.ndarray,
    angles: np.ndarray,
    masks: dict[str, np.ndarray],
    backbone: str,
    mode: str,
    graph_config: dict[str, object] | None,
    angle_weight: float,
    seed: int,
    max_epochs: int,
    patience: int,
    shuffled_training_labels: bool = False,
) -> FitResult:
    torch.manual_seed(seed)
    model = ConditionalRotationModel(backbone, graph_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    train = masks["train"]
    train_gestures, train_angles = (
        permute_train_label_pairs(gestures, angles, train, seed)
        if shuffled_training_labels else (gestures, angles)
    )
    x = torch.from_numpy(np.asarray(samples[train], dtype=np.float32)).permute(0, 2, 1)
    y = torch.from_numpy(np.asarray(train_gestures[train], dtype=np.int64))
    a = torch.from_numpy(np.asarray(train_angles[train], dtype=np.float32))
    best_state = None
    best_epoch = -1
    best_score = -float("inf")
    best_loss = float("inf")
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if mode == "joint":
            angle_vectors, gesture_logits = model.predict_both(x, y, a)
            loss = F.cross_entropy(gesture_logits, y, label_smoothing=0.02)
            loss = loss + angle_weight * angle_vector_loss(angle_vectors, a)
        elif mode == "angle":
            loss = angle_vector_loss(model.predict_angle(x, y), a)
        elif mode == "gesture":
            loss = F.cross_entropy(model.predict_gesture(x, a), y, label_smoothing=0.02)
        else:
            raise ValueError(f"unknown mode: {mode}")
        loss.backward()
        optimizer.step()

        val = masks["val"]
        metrics, _ = evaluate_model(model, samples[val], gestures[val], angles[val], mode)
        score = validation_score(metrics, mode)
        with torch.no_grad():
            val_x = torch.from_numpy(np.asarray(samples[val], dtype=np.float32)).permute(0, 2, 1)
            val_y = torch.from_numpy(np.asarray(gestures[val], dtype=np.int64))
            val_a = torch.from_numpy(np.asarray(angles[val], dtype=np.float32))
            if mode == "joint":
                val_angle, val_gesture = model.predict_both(val_x, val_y, val_a)
                val_loss = F.cross_entropy(val_gesture, val_y, label_smoothing=0.02)
                val_loss = val_loss + angle_weight * angle_vector_loss(val_angle, val_a)
            elif mode == "angle":
                val_loss = angle_vector_loss(model.predict_angle(val_x, val_y), val_a)
            else:
                val_loss = F.cross_entropy(model.predict_gesture(val_x, val_a), val_y, label_smoothing=0.02)
            loss_value = float(val_loss.item())
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and loss_value < best_loss - 1e-6):
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_score = score
            best_loss = loss_value
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return FitResult(model, best_epoch, best_score, best_loss)


def select_trial(trials: list[dict[str, object]]) -> dict[str, object]:
    return max(trials, key=lambda trial: (trial["mean_val_score"], -trial["mean_val_loss"]))


def candidate_configs(backbone: str, mode: str) -> list[dict[str, object]]:
    graph_configs = [None] if backbone == "cnn" else list(GRAPH_CONFIGS)
    weights = (0.5, 2.0) if mode == "joint" else (1.0,)
    return [{"graph_config": graph, "angle_weight": weight} for graph in graph_configs for weight in weights]


def summarize(runs: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary = {}
    for backbone in BACKBONES:
        for mode in MODES:
            for split in ("train", "val", "test"):
                selected = [row for row in runs if row["backbone"] == backbone and row["mode"] == mode and row["split"] == split]
                values = {}
                for metric in ("angle_mae_deg", "angle_within_30_deg", "angle_prediction_30deg_bins", "gesture_macro_f1", "gesture_accuracy", "gesture_min_recall", "gesture_unique_predictions"):
                    if metric not in selected[0]["metrics"]:
                        continue
                    items = np.asarray([row["metrics"][metric] for row in selected], dtype=float)
                    values[f"{metric}_mean"] = float(items.mean())
                    values[f"{metric}_std"] = float(items.std(ddof=1))
                summary[f"{backbone}/{mode}/{split}"] = values
    return summary


def write_report(output_dir: Path, results: dict[str, object]) -> None:
    summary = results["summary"]
    lines = [
        "# W01 conditional bidirectional experiment: movement to wearing angle, angle to movement",
        "",
        "The model receives four-channel movement waveforms and a condition: predict nominal angle given movement, or movement given angle.",
        "Train both directions independently and also train two conditional heads with a shared encoder. Angle estimation uses sin/cos regression and circular absolute error.",
        "Fixed train/validation/test angles: 0/60/120/180/240/300, 30/150/270, and 90/210/330 degrees.",
        "The test angles were evaluated in earlier experiments; every test metric in this report is an exploratory re-evaluation.",
        "",
        "| Backbone | Training mode | Validation angle MAE | Test angle MAE | Validation movement macro-F1 | Test movement macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for backbone in BACKBONES:
        for mode in MODES:
            val = summary[f"{backbone}/{mode}/val"]
            test = summary[f"{backbone}/{mode}/test"]
            angle_val = f"{val['angle_mae_deg_mean']:.1f}°" if "angle_mae_deg_mean" in val else "—"
            angle_test = f"{test['angle_mae_deg_mean']:.1f}° ± {test['angle_mae_deg_std']:.1f}°" if "angle_mae_deg_mean" in test else "—"
            gesture_val = f"{val['gesture_macro_f1_mean']:.3f}" if "gesture_macro_f1_mean" in val else "—"
            gesture_test = f"{test['gesture_macro_f1_mean']:.3f} ± {test['gesture_macro_f1_std']:.3f}" if "gesture_macro_f1_mean" in test else "—"
            lines.append(f"| {backbone} | {mode} | {angle_val} | {angle_test} | {gesture_val} | {gesture_test} |")
    lines += [
        "",
        "Two-seed control with paired training labels shuffled: movement and angle remain paired, while their correspondence to waveforms is broken:",
        "",
        "| Backbone | Training mode | Control angle MAE | Control movement macro-F1 | Predicted 30-degree angle bins on test | Predicted movement classes on test |",
        "|---|---|---:|---:|---:|---:|",
    ]
    controls = results["shuffled_control_summary"]
    for backbone in BACKBONES:
        for mode in MODES:
            control = controls[f"{backbone}/{mode}/test"]
            test = summary[f"{backbone}/{mode}/test"]
            lines.append(
                f"| {backbone} | {mode} | "
                f"{control['angle_mae_deg_mean']:.1f}°" if "angle_mae_deg_mean" in control else f"| {backbone} | {mode} | —"
            )
            lines[-1] += (
                f" | {control['gesture_macro_f1_mean']:.3f}" if "gesture_macro_f1_mean" in control else " | —"
            )
            lines[-1] += (
                f" | {test['angle_prediction_30deg_bins_mean']:.1f}" if "angle_prediction_30deg_bins_mean" in test else " | —"
            )
            lines[-1] += (
                f" | {test['gesture_unique_predictions_mean']:.1f} |" if "gesture_unique_predictions_mean" in test else " | — |"
            )
    lines += [
        "",
        f"Without waveform input, a training-angle median candidate conditioned on the known movement gives test angle MAE: {results['condition_only_angle_baseline_mae_deg']:.1f}°.",
        "Angles are nominal recording settings in 30-degree increments, rather than continuous measured angles.",
        "The five repetitions at an angle share a recording; sequential angle acquisition may confound session drift and contact state.",
        "Configuration selection and early stopping use validation angles only; configurations, per-seed metrics, predictions, and checkpoints are saved in this directory.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tune-epochs", type=int, default=90)
    parser.add_argument("--tune-patience", type=int, default=15)
    parser.add_argument("--final-epochs", type=int, default=220)
    parser.add_argument("--final-patience", type=int, default=35)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output path: {args.output_dir}")
    torch.set_num_threads(1)
    with np.load(args.dataset) as dataset:
        strokes = dataset["strokes"].copy()
        gestures = dataset["labels"].copy()
        angles = dataset["angles"].copy()
        sources = dataset["sources"].copy()
    if strokes.shape != (240, 500, 4) or sorted(set(angles.tolist())) != list(range(0, 360, 30)):
        raise ValueError("dataset does not match frozen W01 corpus")
    masks = split_masks(angles)
    samples, scales = prepare_model_inputs(strokes, angles)
    args.output_dir.mkdir(parents=True)
    trials = []
    winners = {}
    for backbone in BACKBONES:
        for mode in MODES:
            current = []
            for config in candidate_configs(backbone, mode):
                seed_rows = []
                for seed in TUNING_SEEDS:
                    fit = fit_model(samples, gestures, angles, masks, backbone, mode, config["graph_config"], config["angle_weight"], seed, args.tune_epochs, args.tune_patience)
                    seed_rows.append({"seed": seed, "val_score": fit.best_val_score, "val_loss": fit.best_val_loss, "best_epoch": fit.best_epoch})
                row = {
                    "backbone": backbone, "mode": mode, "config": config,
                    "mean_val_score": float(np.mean([item["val_score"] for item in seed_rows])),
                    "mean_val_loss": float(np.mean([item["val_loss"] for item in seed_rows])),
                    "seeds": seed_rows,
                }
                trials.append(row)
                current.append(row)
                print(f"tune {backbone}/{mode} {config}: score={row['mean_val_score']:.3f}", flush=True)
            winners[f"{backbone}/{mode}"] = select_trial(current)["config"]
            print(f"selected {backbone}/{mode}: {winners[f'{backbone}/{mode}']}", flush=True)

    runs = []
    control_runs = []
    predictions = []
    for backbone in BACKBONES:
        for mode in MODES:
            config = winners[f"{backbone}/{mode}"]
            for seed in FINAL_SEEDS:
                fit = fit_model(samples, gestures, angles, masks, backbone, mode, config["graph_config"], config["angle_weight"], seed, args.final_epochs, args.final_patience)
                torch.save(
                    {"backbone": backbone, "mode": mode, "config": config, "seed": seed, "best_epoch": fit.best_epoch, "state_dict": fit.model.state_dict()},
                    args.output_dir / f"{backbone}_{mode}_seed{seed}.pt",
                )
                for split in ("train", "val", "test"):
                    mask = masks[split]
                    metrics, split_predictions = evaluate_model(fit.model, samples[mask], gestures[mask], angles[mask], mode)
                    runs.append({"backbone": backbone, "mode": mode, "seed": seed, "split": split, "best_epoch": fit.best_epoch, "metrics": metrics})
                    if split == "test":
                        for index, prediction in zip(np.flatnonzero(mask), split_predictions, strict=True):
                            predictions.append({
                                "backbone": backbone, "mode": mode, "seed": seed,
                                "sample_index": int(index), "source": str(sources[index]),
                                "known_gesture": GESTURES[int(gestures[index])],
                                "known_angle_deg": int(angles[index]),
                                "predicted_angle_deg": prediction.get("predicted_angle_deg", ""),
                                "predicted_gesture": GESTURES[int(prediction["predicted_gesture"])] if "predicted_gesture" in prediction else "",
                            })
                print(f"final {backbone}/{mode} seed={seed} epoch={fit.best_epoch}", flush=True)

            for seed in TUNING_SEEDS:
                control = fit_model(
                    samples, gestures, angles, masks, backbone, mode,
                    config["graph_config"], config["angle_weight"], seed,
                    args.final_epochs, args.final_patience, shuffled_training_labels=True,
                )
                for split in ("train", "val", "test"):
                    mask = masks[split]
                    metrics, _ = evaluate_model(control.model, samples[mask], gestures[mask], angles[mask], mode)
                    control_runs.append({"backbone": backbone, "mode": mode, "seed": seed, "split": split, "best_epoch": control.best_epoch, "metrics": metrics})
                print(f"shuffled-control {backbone}/{mode} seed={seed} epoch={control.best_epoch}", flush=True)

    train_angles = angles[masks["train"]]
    train_gestures = gestures[masks["train"]]
    test_angles = angles[masks["test"]]
    test_gestures = gestures[masks["test"]]
    medoids = {}
    for gesture in range(4):
        group = train_angles[train_gestures == gesture]
        medoids[gesture] = min(sorted(set(group.tolist())), key=lambda value: angle_metrics(np.full_like(group, value), group)["mae_deg"])
    baseline_angles = np.asarray([medoids[int(gesture)] for gesture in test_gestures])
    baseline_mae = angle_metrics(baseline_angles, test_angles)["mae_deg"]
    results = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "input_shape": list(samples.shape), "sensor_scale": scales.tolist(),
        "split_angles": {"train": [0, 60, 120, 180, 240, 300], "val": [30, 150, 270], "test": [90, 210, 330]},
        "tuning_seeds": TUNING_SEEDS, "final_seeds": FINAL_SEEDS,
        "tune_budget": {"epochs": args.tune_epochs, "patience": args.tune_patience},
        "final_budget": {"epochs": args.final_epochs, "patience": args.final_patience},
        "selection": "mean validation score only; angle=-MAE/180, gesture=macro F1, joint=(F1+1-MAE/180)/2; validation loss tie-break",
        "condition_only_angle_baseline_mae_deg": baseline_mae,
        "condition_only_angle_medoids": {GESTURES[key]: value for key, value in medoids.items()},
        "tuning_trials": trials, "selected_configs": winners,
        "summary": summarize(runs), "runs": runs,
        "shuffled_control_summary": summarize(control_runs), "shuffled_control_runs": control_runs,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    write_report(args.output_dir, results)
    print((args.output_dir / "report.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
