"""Exploratory C170 pilot for the W02 calibration design.

Uses only train and old-validation positions. Registered-test positions (15, 25, 110, 150 mm), all of 70 mm,
and the marked duplicate window are removed immediately after loading and are never filtered, featurized,
predicted or scored. All analyses are leave-one-position-out (LOPO) with a deliberately simple linear model;
they compare protocols relative to each other and are not candidate classifiers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from scipy import signal
from scipy.special import logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed_c170_20260926" / "dataset.npz"
CLASSES = ("UP", "DOWN", "LEFT", "RIGHT")
EXCLUDED_POSITIONS_MM = (15, 25, 110, 150, 70)
OLD_VALIDATION_MM = (5, 40, 90, 130)
CIRCUMFERENCE_MM = 170
N_BINS = 12
SAMPLE_RATE = 250


def load_development(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        positions = archive["distance_mm"].astype(int)
        keep = ~archive["is_duplicate"] & ~np.isin(positions, EXCLUDED_POSITIONS_MM)
        labels = np.array([CLASSES.index(value) for value in archive["direction"][keep]])
        return {"X": archive["X_raw"][keep], "pos": positions[keep], "y": labels,
                "rep": archive["repetition"][keep].astype(int)}


def downsampled_windows(raw: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [0.5, 12], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfiltfilt(sos, raw, axis=1)
    filtered -= filtered.mean(axis=1, keepdims=True)
    return filtered[:, ::10, :]


def features(windows: np.ndarray) -> np.ndarray:
    gain = np.sqrt((windows ** 2).mean(axis=(1, 2), keepdims=True))
    return np.concatenate([(windows / gain).reshape(len(windows), -1), np.log(gain.reshape(-1, 1))], axis=1)


def fit(train_x: np.ndarray, train_y: np.ndarray, weights: np.ndarray | None = None):
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(C=0.05, max_iter=3000).fit(scaler.transform(train_x), train_y, sample_weight=weights)
    return scaler, model


def macro_f1(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(true, predicted, average="macro", labels=list(range(4))))


def summarize(per_position: dict[int, float]) -> dict:
    values = list(per_position.values())
    return {"mean_f1": float(np.mean(values)), "min_f1": float(np.min(values)),
            "per_position": {str(key): float(value) for key, value in per_position.items()}}


def representation_variants(data, windows) -> dict:
    positions, labels = data["pos"], data["y"]

    def per_wear_gain(values):
        values = values.copy()
        for position in np.unique(positions):
            mask = positions == position
            values[mask] /= np.sqrt((values[mask] ** 2).mean(axis=(0, 1)))
        return values

    def canonical(values):
        hottest = (values ** 2).sum(axis=1).argmax(axis=1)
        return np.stack([np.roll(values[i], -hottest[i], axis=1) for i in range(len(values))])

    variants = {"raw": windows, "per_wear_gain": per_wear_gain(windows), "hottest_channel_canonical": canonical(windows),
                "per_wear_gain_canonical": canonical(per_wear_gain(windows))}
    result = {}
    for name, values in variants.items():
        matrix = features(values)
        scores = {}
        for position in np.unique(positions):
            scaler, model = fit(matrix[positions != position], labels[positions != position])
            held = positions == position
            scores[int(position)] = macro_f1(labels[held], model.predict(scaler.transform(matrix[held])))
        result[name] = summarize(scores)
    return result


def block_permuted_labels(positions: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Matched null: permute the four complete direction blocks within every position."""
    rng = np.random.default_rng(seed)
    permuted = labels.copy()
    for position in np.unique(positions):
        mask = positions == position
        permuted[mask] = rng.permutation(4)[labels[mask]]
    return permuted


def few_shot_calibration(data, windows) -> dict:
    positions, labels, repetitions = data["pos"], data["y"], data["rep"]
    matrix = features(windows)
    null_labels = {seed: block_permuted_labels(positions, labels, seed) for seed in (17, 29, 43, 61, 89)}
    result = {}
    for shots in (0, 1, 2, 5):
        pooled, calibration_only, null = {}, {}, {seed: {} for seed in null_labels}
        for position in np.unique(positions):
            calibration = (positions == position) & (repetitions <= shots)
            scored = (positions == position) & (repetitions > 5)
            train = (positions != position) | calibration
            weights = np.where(calibration, 20.0, 1.0)[train]
            scaler, model = fit(matrix[train], labels[train], weights)
            pooled[int(position)] = macro_f1(labels[scored], model.predict(scaler.transform(matrix[scored])))
            for seed, targets in null_labels.items():
                scaler, model = fit(matrix[train], targets[train], weights)
                null[seed][int(position)] = macro_f1(labels[scored], model.predict(scaler.transform(matrix[scored])))
            if shots:
                scaler, model = fit(matrix[calibration], labels[calibration])
                calibration_only[int(position)] = macro_f1(labels[scored], model.predict(scaler.transform(matrix[scored])))
        result[f"shots_{shots}"] = {"pooled_plus_calibration": summarize(pooled),
                                    "matched_null_mean_f1_by_seed": {str(seed): summarize(values)["mean_f1"]
                                                                     for seed, values in null.items()}}
        if shots:
            result[f"shots_{shots}"]["calibration_only"] = summarize(calibration_only)
    return result


def joint_orientation(data, windows, roll_sign: int) -> dict:
    positions, labels, repetitions = data["pos"], data["y"], data["rep"]
    bins = np.round(positions / CIRCUMFERENCE_MM * N_BINS).astype(int) % N_BINS
    scores = {name: {} for name in ("marginal", "wear_unlabeled", "one_labeled_per_class", "oracle_bin")}
    bin_errors = {}
    for position in np.unique(positions):
        train, held = positions != position, positions == position
        rolled = [np.roll(windows[train], shift, axis=2) for shift in range(4)]
        targets = [labels[train] * N_BINS + (bins[train] + roll_sign * 3 * shift) % N_BINS for shift in range(4)]
        scaler, model = fit(features(np.concatenate(rolled)), np.concatenate(targets))
        log_joint = np.full((held.sum(), 4 * N_BINS), -1e9)
        log_joint[:, model.classes_] = model.predict_log_proba(scaler.transform(features(windows[held])))
        log_joint = log_joint.reshape(-1, 4, N_BINS)
        true, scored = labels[held], repetitions[held] > 1
        scores["marginal"][int(position)] = macro_f1(true[scored], logsumexp(log_joint, 2).argmax(1)[scored])
        log_bin = logsumexp(log_joint, axis=1)
        total = log_bin.sum(axis=0)
        predictions = []
        for index in range(len(log_joint)):
            posterior = total - log_bin[index]
            posterior -= logsumexp(posterior)
            predictions.append(logsumexp(log_joint[index] + posterior[None, :], axis=1).argmax())
        scores["wear_unlabeled"][int(position)] = macro_f1(true[scored], np.array(predictions)[scored])
        calibration = repetitions[held] == 1
        posterior = log_joint[calibration][np.arange(calibration.sum()), true[calibration]].sum(axis=0)
        scores["one_labeled_per_class"][int(position)] = macro_f1(
            true[scored], logsumexp(log_joint + (posterior - logsumexp(posterior))[None, None, :], 2).argmax(1)[scored])
        scores["oracle_bin"][int(position)] = macro_f1(true[scored], log_joint[:, :, bins[held][0]].argmax(1)[scored])
        estimate, actual = int(total.argmax()), int(bins[held][0])
        bin_errors[str(int(position))] = int(min((estimate - actual) % N_BINS, (actual - estimate) % N_BINS))
    return {name: summarize(values) for name, values in scores.items()} | {
        "wear_bin_error_30deg": bin_errors, "mean_wear_bin_error_30deg": float(np.mean(list(bin_errors.values())))}


def neighbor_training(data, windows, seed: int = 0) -> dict:
    positions, labels = data["pos"], data["y"]
    matrix = features(windows)
    unique = np.unique(positions)
    rng = np.random.default_rng(seed)

    def circular(a, b):
        distance = abs(int(a) - int(b)) % CIRCUMFERENCE_MM
        return min(distance, CIRCUMFERENCE_MM - distance)

    def score(train_positions, position):
        train, held = np.isin(positions, train_positions), positions == position
        scaler, model = fit(matrix[train], labels[train])
        return macro_f1(labels[held], model.predict(scaler.transform(matrix[held])))

    result = {}
    for count in (2, 4, 6):
        nearest = [score(sorted((q for q in unique if q != p), key=lambda q: circular(p, q))[:count], p) for p in unique]
        random = [score(rng.choice([q for q in unique if q != p], count, replace=False), p) for p in unique for _ in range(5)]
        result[f"k_{count}"] = {"nearest_mean_f1": float(np.mean(nearest)), "random_mean_f1": float(np.mean(random))}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; must not exist")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_development(args.dataset)
    windows = downsampled_windows(data["X"])
    report = {
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "access_boundary": "train and old-validation positions only; registered-test positions, 70 mm and duplicate removed after load",
        "positions_mm": [int(value) for value in np.unique(data["pos"])],
        "n_windows": int(len(data["y"])),
        "classes": list(CLASSES),
        "status": "exploratory; linear LOPO pilot, not a candidate model",
        "representation_variants": representation_variants(data, windows),
        "few_shot_calibration": few_shot_calibration(data, windows),
        "joint_orientation_roll_minus": joint_orientation(data, windows, -1),
        "joint_orientation_roll_plus": joint_orientation(data, windows, 1),
        "neighbor_training": neighbor_training(data, windows),
    }
    (args.output_dir / "calibration_pilot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("positions_mm",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
