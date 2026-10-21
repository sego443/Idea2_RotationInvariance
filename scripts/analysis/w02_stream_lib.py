"""Shared protocol for the W02 causal startup experiments (development wears only).

Protocol: one wear (one donning, six recordings, about 120 strokes) is a stream. Startup calibration takes K strokes
from the prefix of the wear's first recording (for each class in the calibrated subset, occurrence `draw` of that
class in temporal order); they are treated as performed first. Every other stroke of the wear is then predicted in
recording order, using only the calibration and earlier strokes, and is scored.

Also: covariance and waveform features, recording start times from the accepted acquisition plans, stream metrics,
run records with source/data hashes, and overwrite protection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from sklearn.metrics import f1_score

from src import v21_data, w02_cov_fewshot as recipes

SEALED_WEARS = {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}
DEFAULT_DATASET = Path("outputs/w02_dev_20261015/dataset.npz")
PLANS = Path("data/w02_plans")
FS = 250
POSITION_BINS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10_000))
MINUTE_BINS = ((0, 2), (2, 4), (4, 6), (6, 9), (9, 1_000))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dev(path: Path) -> dict:
    data = v21_data.load_w02(path)
    if set(np.unique(data["wear_group"])) & SEALED_WEARS:
        raise RuntimeError("sealed wear present in the development dataset")
    return data


def start_run(output_dir: Path, script: str, dataset: Path, settings: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    run = {"started_utc": datetime.now(timezone.utc).isoformat(), "script": Path(script).name,
           "script_sha256": sha256(Path(script)), "lib_sha256": sha256(Path(__file__)),
           "recipe_source_sha256": sha256(Path(recipes.__file__)), "dataset": str(dataset),
           "dataset_sha256": sha256(dataset), "class_order": list(v21_data.CLASSES), "settings": settings,
           "test_access": "development wears only; sealed wears W07/W10/W12 refused; exploratory"}
    (output_dir / "run.json").write_text(json.dumps(run, indent=2))


def finish_run(output_dir: Path, rows: list, predictions: dict | None = None, extra: dict | None = None) -> None:
    if predictions:
        np.savez_compressed(output_dir / "predictions.npz", **predictions)
    (output_dir / "results.json").write_text(json.dumps({"rows": rows, **(extra or {})}, indent=1))


# ---------------------------------------------------------------- streams


def wear_streams(data: dict) -> dict:
    """wear -> {"order": stroke indices in time order, "recordings": groups in order, "first_prefix": indices}."""
    out = {}
    for wear in np.unique(data["wear_group"]):
        idx = np.flatnonzero(data["wear_group"] == wear)
        idx = idx[np.argsort(data["repetition"][idx], kind="stable")]
        recordings = list(dict.fromkeys(data["group"][idx]))
        first = idx[data["group"][idx] == recordings[0]]
        out[wear] = {"order": idx, "recordings": recordings, "first_prefix": first[data["is_calibration"][first]]}
    return out


def calibration(stream: dict, labels: np.ndarray, classes, draw: int):
    """(anchor indices, anchor labels, remaining stream in time order). draw = occurrence used; 'both' = both."""
    prefix = stream["first_prefix"]
    picks = []
    for c in classes:
        occurrences = prefix[labels[prefix] == c]
        picks.extend(occurrences if draw == "both" else occurrences[draw:draw + 1])
    anchors = np.array(picks, dtype=int)
    rest = stream["order"][~np.isin(stream["order"], anchors)]
    return anchors, labels[anchors], rest


def recording_minutes(data: dict, plans: Path = PLANS) -> dict:
    """group -> minutes from the start of its wear's first recording (None when the plan is unavailable)."""
    starts = {}
    for path in sorted(plans.glob("*/*_plan_accepted.csv")):
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", row.get("actual_filename", ""))
                if row.get("kind") != "trial" or row.get("accepted") != "yes" or not match:
                    continue
                group = f"W02_{row['participant']}_{row['part']}_W{int(row['wear']):02d}_{row['recording']}"
                starts[group] = datetime.strptime(match.group(1), "%Y-%m-%d_%H-%M-%S")
    out = {}
    for wear in np.unique(data["wear_group"]):
        groups = list(dict.fromkeys(data["group"][data["wear_group"] == wear]))
        if all(g in starts for g in groups):
            t0 = min(starts[g] for g in groups)
            out.update({g: (starts[g] - t0).total_seconds() / 60 for g in groups})
        else:
            out.update({g: None for g in groups})
    return out


# ---------------------------------------------------------------- features


def bandpass(windows: np.ndarray, band=(0.5, 12.0)) -> np.ndarray:
    sos = signal.butter(3, band, btype="bandpass", fs=FS, output="sos")
    x = signal.sosfiltfilt(sos, windows, axis=1)
    return x - x.mean(axis=1, keepdims=True)


def covariances(windows, band=(0.5, 12.0), lags=(0, 5, 10, 15, 20, 25, 30), norm="none"):
    """Lagged covariance (as src/w02_cov_fewshot.py). norm: none | trace (unit trace) | corr (unit diagonal)."""
    x = bandpass(windows, band)
    z = np.concatenate([np.roll(x, lag, axis=1) for lag in lags], axis=2)[:, max(lags):]
    covs = np.einsum("nti,ntj->nij", z, z) / z.shape[1]
    if norm == "trace":
        covs = covs / np.trace(covs, axis1=1, axis2=2)[:, None, None]
    elif norm == "corr":
        d = np.sqrt(np.einsum("nii->ni", covs))
        covs = covs / (d[:, :, None] * d[:, None, :])
    return covs + 1e-6 * np.trace(covs, axis1=1, axis2=2)[:, None, None] * np.eye(covs.shape[1])


def tangent(covs, reference):
    return recipes.tangent(covs, reference)


def log_euclid(covs: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(covs.shape[1])
    weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    out = []
    for c in covs:
        w, v = eigh(c)
        out.append((v @ np.diag(np.log(np.maximum(w, 1e-12))) @ v.T)[iu] * weight)
    return np.array(out)


def waveform(windows, band=(0.5, 12.0), step=10, norm="global"):
    """Band-passed, downsampled waveform, flattened. norm: global (stroke RMS) | channel (per-channel RMS) | none."""
    x = bandpass(windows, band)[:, ::step]
    if norm == "global":
        x = x / (np.sqrt((x ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-9)
    elif norm == "channel":
        x = x / (np.sqrt((x ** 2).mean(axis=1, keepdims=True)) + 1e-9)
    return x.reshape(len(x), -1)


# ---------------------------------------------------------------- metrics


def stream_report(data: dict, idx: np.ndarray, pred: np.ndarray, positions: np.ndarray, rec_rank: np.ndarray,
                  minutes: np.ndarray | None = None, actions: np.ndarray | None = None) -> dict:
    """Metrics over the concatenated wear streams. positions are 1-based within each wear's stream."""
    y = data["y"][idx]
    wears, angles = data["wear_group"][idx], data["position"][idx]

    def acc(mask):
        return float(np.mean(y[mask] == pred[mask])) if np.any(mask) else None

    within = np.zeros(len(idx), int)  # strokes since the start of the stroke's recording within its wear stream
    for i in range(1, len(idx)):
        if positions[i] != 1 and rec_rank[i] == rec_rank[i - 1]:
            within[i] = within[i - 1] + 1
    first_after_boundary = (rec_rank > 0) & (within < 3)
    out = {"accuracy": acc(np.ones(len(idx), bool)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=sorted(set(y.tolist())), zero_division=0)),
           "recall": {name: acc(y == c) for c, name in enumerate(v21_data.CLASSES)},
           "prediction_counts": dict(zip(v21_data.CLASSES, np.bincount(pred, minlength=4).tolist())),
           "accuracy_by_position": {(f"{a}-{b}" if b < 10_000 else f"{a}+"): acc((positions >= a) & (positions <= b))
                                    for a, b in POSITION_BINS},
           "accuracy_by_recording": {int(r) + 1: acc(rec_rank == r) for r in np.unique(rec_rank)},
           "accuracy_first3_after_boundary": acc(first_after_boundary),
           "per_wear_accuracy": {w: acc(wears == w) for w in np.unique(wears)},
           "per_angle_accuracy": {int(a): acc(angles == a) for a in np.unique(angles)},
           "n_scored": int(len(idx))}
    if minutes is not None and not np.any(np.isnan(minutes)):
        out["accuracy_by_minutes"] = {(f"{a}-{b}" if b < 1_000 else f"{a}+"): acc((minutes >= a) & (minutes < b))
                                      for a, b in MINUTE_BINS}
    if actions is not None:
        out["user_actions_total_per_wear"] = float(actions.sum() / len(np.unique(wears)))
        out["user_actions_by_position"] = {(f"{a}-{b}" if b < 10_000 else f"{a}+"):
                                           float(actions[(positions >= a) & (positions <= b)].sum()
                                                 / len(np.unique(wears))) for a, b in POSITION_BINS}
    return out


def stream_context(data: dict, streams: dict, pieces: list, minutes_by_group: dict | None):
    """pieces: [(wear, stream indices in order)] -> (idx, positions, rec_rank, minutes)."""
    idx, positions, ranks, minutes = [], [], [], []
    for wear, rest in pieces:
        rank = {g: r for r, g in enumerate(streams[wear]["recordings"])}
        idx.append(rest)
        positions.append(np.arange(1, len(rest) + 1))
        ranks.append(np.array([rank[g] for g in data["group"][rest]]))
        if minutes_by_group is not None:
            minutes.append(np.array([np.nan if minutes_by_group.get(g) is None else minutes_by_group[g]
                                     for g in data["group"][rest]], dtype=float))
    return (np.concatenate(idx), np.concatenate(positions), np.concatenate(ranks),
            np.concatenate(minutes) if minutes else None)


def short(row: dict) -> str:
    bins = [v for v in row["accuracy_by_position"].values()]
    rec = list(row["accuracy_by_recording"].values())
    text = (f"acc {row['accuracy']:.3f} F1 {row['macro_f1']:.3f} pos {[round(v, 2) for v in bins]} "
            f"rec1 {rec[0]:.2f} rec6 {rec[-1]:.2f} pred {list(row['prediction_counts'].values())}")
    if "null_accuracy" in row:
        text += f" null {np.mean(row['null_accuracy']):.3f}"
    if "user_actions_total_per_wear" in row:
        text += f" actions/wear {row['user_actions_total_per_wear']:.1f}"
    return text
