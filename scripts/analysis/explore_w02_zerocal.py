"""Exploratory zero-calibration (no labeled strokes from the test wear) on the W02 development wears.

Leave-one-wear-out over the nine development wears: a classifier is trained on all strokes (prefix and body) of
the other eight wears and scores the body strokes of the held-out wear without any of its labels.

Factors:
  derotation  none | harm+ | harm- | cyc+ | cyc-   canonicalize the ring channels by the measured band angle theta
              (harm: ring-DFT harmonic 1 times exp(-/+ j theta), harmonic 2 times cos(2 theta); cyc: cyclic shift
              by round(theta / 90)); the sign checks the frame convention
  recenter    global | wear_all | rec_prefix | wear_rest   reference covariance for the tangent space: training
              mean only, or per wear/recording from unlabeled inputs (all strokes of the wear, the recording's
              unlabeled prefix, or the wear's REST windows)
Features: lagged covariance (lags 0..30 step 5, 0.5-12 Hz) in tangent space. Classifier: logistic regression.
A training-label permutation null (labels permuted within each training recording, seed 0) is scored per cell.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from scripts.analysis.explore_w02_fewshot import bandpass, feat_cov, summarize, tangent
from src import v21_data

LAGS = (0, 5, 10, 15, 20, 25, 30)


def derotate(windows: np.ndarray, theta_deg: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return windows
    sign = 1 if mode.endswith("+") else -1
    theta = np.deg2rad(theta_deg)[:, None] * sign
    if mode.startswith("cyc"):
        shifts = np.round(np.rad2deg(theta[:, 0]) / 90).astype(int) % 4
        return np.stack([np.roll(w, -s, axis=1) for w, s in zip(windows, shifts)])
    spectrum = np.fft.fft(windows, axis=2)  # ring DFT over the four sensors
    spectrum[:, :, 1] *= np.exp(-1j * theta)
    spectrum[:, :, 3] *= np.exp(1j * theta)
    spectrum[:, :, 2] *= np.cos(2 * theta)
    return np.fft.ifft(spectrum, axis=2).real.astype(np.float32)


def references(covs, data, rest_covs, rest_wear_groups, mode):
    """Reference covariance per stroke."""
    if mode == "global":
        return None
    out = np.empty((len(covs),) + covs.shape[1:])
    for group in np.unique(data["group"]):
        mask = data["group"] == group
        wear = data["wear_group"][mask][0]
        if mode == "wear_all":
            ref = covs[data["wear_group"] == wear].mean(0)
        elif mode == "rec_prefix":
            ref = covs[mask & data["is_calibration"]].mean(0)
        elif mode == "wear_rest":
            ref = rest_covs[rest_wear_groups == wear].mean(0)
        else:
            raise ValueError(mode)
        out[mask] = ref
    return out


def tangent_per_reference(covs, refs):
    if refs is None:
        return None
    out = np.empty((len(covs), covs.shape[1] * (covs.shape[1] + 1) // 2))
    keys = {}
    for i, ref in enumerate(refs):
        keys.setdefault(ref.tobytes(), []).append(i)
    for indices in keys.values():
        out[indices] = tangent(covs[indices], refs[indices[0]])
    return out


def run(data, rest, derot, recenter, null_seed=None, C=0.1):
    theta = data["position"].astype(float)
    covs = feat_cov(derotate(data["X"], theta, derot), lags=LAGS)
    rest_covs = feat_cov(derotate(rest["X"], rest["theta"], derot), lags=LAGS) if recenter == "wear_rest" else None
    refs = references(covs, data, rest_covs, rest.get("wear_group"), recenter)
    local = tangent_per_reference(covs, refs)
    labels = data["y"].copy()
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        for group in np.unique(data["group"]):
            mask = data["group"] == group
            labels[mask] = rng.permutation(4)[labels[mask]]
    predicted = np.full(len(labels), -1)
    for wear in np.unique(data["wear_group"]):
        test = (data["wear_group"] == wear) & ~data["is_calibration"]
        train = data["wear_group"] != wear
        if local is None:
            ref = covs[train].mean(0)
            features = tangent(covs, ref)
        else:
            features = local
        mean, std = features[train].mean(0), features[train].std(0) + 1e-6
        model = LogisticRegression(C=C, max_iter=3000)
        model.fit((features[train] - mean) / std, labels[train])
        predicted[test] = model.predict((features[test] - mean) / std)
    return predicted


def load_rest(path: Path) -> dict:
    with np.load(path) as archive:
        wear, part = archive["rest_wear"].astype(int), archive["rest_part"].astype(str)
        angle = {(p, w): a for p, w, a in zip(archive["part"].astype(str), archive["wear"].astype(int),
                                               archive["actual_angle_deg"].astype(int))}
        return {"X": archive["rest_X_raw"].astype(np.float32),
                "wear_group": np.array([f"W02_P01_{p}_W{w:02d}" for p, w in zip(part, wear)]),
                "theta": np.array([angle[(p, w)] for p, w in zip(part, wear)], dtype=float)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/w02_dev_20261015/dataset.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--derotations", nargs="+", default=["none", "harm+", "harm-", "cyc+", "cyc-"])
    parser.add_argument("--recenters", nargs="+", default=["global", "wear_all", "rec_prefix", "wear_rest"])
    args = parser.parse_args()
    data = v21_data.load_w02(args.dataset)
    if set(np.unique(data["wear_group"])) & {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}:
        raise RuntimeError("sealed wear present in the development dataset")
    rest = load_rest(args.dataset)
    rows = []
    for derot in args.derotations:
        for recenter in args.recenters:
            real = summarize(data, run(data, rest, derot, recenter))
            null = summarize(data, run(data, rest, derot, recenter, null_seed=0))
            rows.append({"derotation": derot, "recenter": recenter, **real, "null_macro_f1": null["macro_f1"]})
            print(f"{derot:6s} {recenter:10s} F1 {real['macro_f1']:.3f} meanW {real['mean_wear_f1']:.3f} "
                  f"minW {real['min_wear_f1']:.3f} minRec {real['min_recall']:.2f} pred {real['pred_counts']} "
                  f"null {null['macro_f1']:.3f}", flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
