"""Exploratory training-free few-shot classifiers on the W02 development wears (sealed wears W07/W10/W12 excluded).

Scores the 12 body strokes of every development trial recording. Calibration (support) scenarios:
  rec    the recording's own labeled 8-stroke prefix (as E1)
  other  the prefixes of the other recordings of the same wear (as E1b)
  wear   all prefixes of the wear, including the recording's own (48 strokes; uses later recordings' prefixes)
  causal the prefixes of this and earlier recordings of the wear
Each (feature, classifier, scenario) cell reports pooled macro-F1, the mean and minimum per-wear macro-F1, the
minimum class recall and prediction counts. A matched support-label permutation (labels permuted within each
support set, seed 0) is scored for every cell as a null. Exploratory: everything here is tuned on the
development wears.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src import v21_data

FS = 250


def bandpass(windows: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = signal.butter(3, [low, high], btype="bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, windows, axis=1)


def feat_wave(windows, low=0.5, high=12.0, step=10, norm="global", crop=(0, 500)):
    x = bandpass(windows, low, high)[:, crop[0]:crop[1]]
    x = x - x.mean(axis=1, keepdims=True)
    x = x[:, ::step, :]
    if norm == "global":
        gain = np.sqrt((x ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
        return np.concatenate([(x / gain).reshape(len(x), -1), np.log(gain.reshape(-1, 1))], axis=1)
    if norm == "channel":
        gain = np.sqrt((x ** 2).mean(axis=1, keepdims=True)) + 1e-12
        return np.concatenate([(x / gain).reshape(len(x), -1), np.log(gain.reshape(len(x), -1))], axis=1)
    return x.reshape(len(x), -1)


def tangent(covs: np.ndarray, reference: np.ndarray) -> np.ndarray:
    w, v = eigh(reference)
    isqrt = v @ np.diag(w ** -0.5) @ v.T
    out = []
    iu = np.triu_indices(covs.shape[1])
    weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    for c in covs:
        m = isqrt @ c @ isqrt
        w2, v2 = eigh(m)
        logm = v2 @ np.diag(np.log(np.maximum(w2, 1e-12))) @ v2.T
        out.append(logm[iu] * weight)
    return np.array(out)


def feat_cov(windows, low=0.5, high=12.0, lags=(0, 10, 20)):
    """Lagged (augmented) covariance matrices; tangent-space mapping happens per support set."""
    x = bandpass(windows, low, high)
    x = x - x.mean(axis=1, keepdims=True)
    stacks = [np.concatenate([np.roll(x, lag, axis=1) for lag in lags], axis=2)[:, max(lags):]]
    z = stacks[0]
    covs = np.einsum("nti,ntj->nij", z, z) / z.shape[1]
    covs += 1e-6 * np.trace(covs, axis1=1, axis2=2)[:, None, None] * np.eye(covs.shape[1])
    return covs


def classify(name, train_x, train_y, test_x):
    if name == "nc":  # nearest centroid after z-scoring with support statistics (E4 rule)
        mean, std = train_x.mean(0), train_x.std(0) + 1e-6
        a, b = (train_x - mean) / std, (test_x - mean) / std
        cent = np.stack([a[train_y == c].mean(0) for c in range(4)])
        return ((b[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
    if name == "nc_cos":
        mean = train_x.mean(0)
        a, b = train_x - mean, test_x - mean
        cent = np.stack([a[train_y == c].mean(0) for c in range(4)])
        cent /= np.linalg.norm(cent, axis=1, keepdims=True) + 1e-12
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
        return (b @ cent.T).argmax(1)
    if name == "lda":
        mean, std = train_x.mean(0), train_x.std(0) + 1e-6
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        return model.fit((train_x - mean) / std, train_y).predict((test_x - mean) / std)
    if name == "logreg":
        mean, std = train_x.mean(0), train_x.std(0) + 1e-6
        model = LogisticRegression(C=0.1, max_iter=2000)
        return model.fit((train_x - mean) / std, train_y).predict((test_x - mean) / std)
    if name == "knn1":
        mean, std = train_x.mean(0), train_x.std(0) + 1e-6
        a, b = (train_x - mean) / std, (test_x - mean) / std
        return train_y[((b[:, None, :] - a[None]) ** 2).sum(2).argmin(1)]
    raise ValueError(name)


def support_index(data, group, scenario):
    wear = data["wear_group"][data["group"] == group][0]
    trial = int(group.rsplit("_T", 1)[1])
    in_wear = (data["wear_group"] == wear) & data["is_calibration"]
    trials = np.array([int(g.rsplit("_T", 1)[1]) for g in data["group"]])
    if scenario == "rec":
        mask = in_wear & (data["group"] == group)
    elif scenario == "other":
        mask = in_wear & (data["group"] != group)
    elif scenario == "wear":
        mask = in_wear
    elif scenario == "causal":
        mask = in_wear & (trials <= trial)
    else:
        raise ValueError(scenario)
    return np.flatnonzero(mask)


def evaluate(data, features, clf, scenario, permute_seed=None, cov=False):
    predicted = np.full(len(data["y"]), -1)
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    for group in np.unique(data["group"]):
        body = np.flatnonzero((data["group"] == group) & ~data["is_calibration"])
        support = support_index(data, group, scenario)
        labels = data["y"][support].copy()
        if rng is not None:
            labels = rng.permutation(labels)
        if cov:
            reference = features[support].mean(0)
            train_x, test_x = tangent(features[support], reference), tangent(features[body], reference)
        else:
            train_x, test_x = features[support], features[body]
        predicted[body] = classify(clf, train_x, labels, test_x)
    return predicted


def summarize(data, predicted):
    scored = ~data["is_calibration"]
    y, p, wears = data["y"][scored], predicted[scored], data["wear_group"][scored]
    per_wear = {w[-3:]: float(f1_score(y[wears == w], p[wears == w], average="macro", labels=list(range(4))))
                for w in np.unique(wears)}
    return {"macro_f1": float(f1_score(y, p, average="macro", labels=list(range(4)))),
            "mean_wear_f1": float(np.mean(list(per_wear.values()))), "min_wear_f1": float(min(per_wear.values())),
            "min_recall": float(min(np.mean(p[y == c] == c) for c in range(4))),
            "pred_counts": np.bincount(p, minlength=4).tolist(), "per_wear": per_wear}


FEATURES = {
    "wave_0.5-12_s10_global": lambda X: feat_wave(X),
    "wave_0.5-12_s5_global": lambda X: feat_wave(X, step=5),
    "wave_0.5-12_s10_channel": lambda X: feat_wave(X, norm="channel"),
    "wave_0.5-12_s10_none": lambda X: feat_wave(X, norm="none"),
    "wave_0.5-6_s10_global": lambda X: feat_wave(X, high=6.0),
    "wave_0.5-25_s5_global": lambda X: feat_wave(X, high=25.0, step=5),
    "wave_1-12_s10_global": lambda X: feat_wave(X, low=1.0),
    "wave_0.1-12_s10_global": lambda X: feat_wave(X, low=0.1),
    "wave_0.5-12_s10_mid300": lambda X: feat_wave(X, crop=(100, 400)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/w02_dev_20261015/dataset.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classifiers", nargs="+", default=["nc", "nc_cos", "lda", "logreg", "knn1"])
    parser.add_argument("--scenarios", nargs="+", default=["rec", "other", "causal", "wear"])
    args = parser.parse_args()
    data = v21_data.load_w02(args.dataset)
    if set(np.unique(data["wear_group"])) & {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}:
        raise RuntimeError("sealed wear present in the development dataset")
    rows = []
    feature_sets = {name: (fn(data["X"]), False) for name, fn in FEATURES.items()}
    feature_sets["cov_0.5-12_lag0-10-20"] = (feat_cov(data["X"]), True)
    feature_sets["cov_0.5-12_lag0"] = (feat_cov(data["X"], lags=(0,)), True)
    for fname, (features, cov) in feature_sets.items():
        for clf in args.classifiers:
            if cov and clf == "nc":
                pass
            for scenario in args.scenarios:
                real = summarize(data, evaluate(data, features, clf, scenario, cov=cov))
                null = summarize(data, evaluate(data, features, clf, scenario, permute_seed=0, cov=cov))
                rows.append({"feature": fname, "classifier": clf, "scenario": scenario, **real,
                             "null_macro_f1": null["macro_f1"]})
                print(f"{fname:28s} {clf:7s} {scenario:6s} F1 {real['macro_f1']:.3f} meanW {real['mean_wear_f1']:.3f}"
                      f" minW {real['min_wear_f1']:.3f} minRec {real['min_recall']:.2f} null {null['macro_f1']:.3f}",
                      flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
