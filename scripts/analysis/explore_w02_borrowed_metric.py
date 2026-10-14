"""Exploratory calibrated classifiers that borrow a metric from the other development wears.

For each held-out development wear (leave-one-wear-out), every recording is mapped to tangent space at its own
reference covariance (mean of the recording's unlabeled prefix, or of its 20 unlabeled strokes). The within-class
scatter of the other wears (strokes minus their recording-class means) gives a shrunken shared covariance S
(Ledoit-Wolf on a PCA basis fitted to the other wears). The test recording's labeled support strokes give the
class means; strokes are assigned by Mahalanobis distance under S (LDA rule). Optional transductive refinement
re-estimates the class means from the recording's body strokes with soft assignments (two steps, support kept).
Support: k = 1 or 2 strokes per class from the recording's prefix (first k of each class). Compared with
nearest-centroid on z-scored support features (the training-free baseline). Optional ring derotation (harm+/-).
A matched null permutes the support labels within each recording (seed 0).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from scripts.analysis.explore_w02_fewshot import classify, feat_cov, summarize, tangent
from scripts.analysis.explore_w02_zerocal import LAGS, derotate
from src import v21_data


def recording_tangent(covs, data, reference):
    out = np.empty((len(covs), covs.shape[1] * (covs.shape[1] + 1) // 2))
    for group in np.unique(data["group"]):
        mask = data["group"] == group
        ref_mask = mask & data["is_calibration"] if reference == "prefix" else mask
        out[mask] = tangent(covs[mask], covs[ref_mask].mean(0))
    return out


def within_scatter(features, data, mask):
    residuals = []
    for group in np.unique(data["group"][mask]):
        for c in range(4):
            m = mask & (data["group"] == group) & (data["y"] == c)
            residuals.append(features[m] - features[m].mean(0))
    return np.concatenate(residuals)


def run(data, shots, method, reference="prefix", derot="none", n_components=60, refine=0, null_seed=None):
    covs = feat_cov(derotate(data["X"], data["position"].astype(float), derot), lags=LAGS)
    features = recording_tangent(covs, data, reference)
    rng = np.random.default_rng(null_seed) if null_seed is not None else None
    predicted = np.full(len(data["y"]), -1)
    for wear in np.unique(data["wear_group"]):
        test_wear = data["wear_group"] == wear
        if method == "borrowed":
            train = ~test_wear
            pca = PCA(n_components=n_components).fit(features[train])
            z = pca.transform(features)
            scatter = within_scatter(z, data, train)
            precision = LedoitWolf().fit(scatter).precision_
        for group in np.unique(data["group"][test_wear]):
            prefix = np.flatnonzero((data["group"] == group) & data["is_calibration"])
            body = np.flatnonzero((data["group"] == group) & ~data["is_calibration"])
            support = np.concatenate([prefix[data["y"][prefix] == c][:shots] for c in range(4)])
            labels = data["y"][support].copy()
            if rng is not None:
                labels = rng.permutation(labels)
            if method == "nc":
                predicted[body] = classify("nc", features[support], labels, features[body])
                continue
            means = np.stack([z[support][labels == c].mean(0) for c in range(4)])
            for _ in range(refine):
                diff = z[body][:, None, :] - means[None]
                d2 = np.einsum("nci,ij,ncj->nc", diff, precision, diff)
                weights = np.exp(-(d2 - d2.min(1, keepdims=True)) / 2)
                weights /= weights.sum(1, keepdims=True)
                support_onehot = np.eye(4)[labels]
                total = support_onehot.T @ z[support] + weights.T @ z[body]
                means = total / (support_onehot.sum(0) + weights.sum(0))[:, None]
            diff = z[body][:, None, :] - means[None]
            predicted[body] = np.einsum("nci,ij,ncj->nc", diff, precision, diff).argmin(1)
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/w02_dev_20261015/dataset.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = v21_data.load_w02(args.dataset)
    if set(np.unique(data["wear_group"])) & {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}:
        raise RuntimeError("sealed wear present in the development dataset")
    configs = [("nc", {}), ("borrowed", {}), ("borrowed", {"n_components": 30}), ("borrowed", {"n_components": 100}),
               ("borrowed", {"refine": 2}), ("borrowed", {"reference": "recording"}),
               ("borrowed", {"reference": "recording", "refine": 2}), ("borrowed", {"derot": "harm+"}),
               ("borrowed", {"derot": "harm-"})]
    rows = []
    for shots in (1, 2):
        for method, options in configs:
            real = summarize(data, run(data, shots, method, **options))
            null = summarize(data, run(data, shots, method, null_seed=0, **options))
            rows.append({"shots": shots, "method": method, **options, **real, "null_macro_f1": null["macro_f1"]})
            print(f"{shots}-shot {method:8s} {json.dumps(options):45s} F1 {real['macro_f1']:.3f} "
                  f"minW {real['min_wear_f1']:.3f} minRec {real['min_recall']:.2f} null {null['macro_f1']:.3f}",
                  flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
