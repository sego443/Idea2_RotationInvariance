"""Exploratory label-free adaptation and calibration-size sweep on the W02 development wears.

Label-free (no labels from the test wear), leave-one-wear-out over the nine development wears:
  zero        logistic regression trained on the other wears (derotation harm-, per-wear re-centering) predicts
              each stroke on its own (as explore_w02_zerocal)
  cluster     the test unit's unlabeled strokes are clustered into four groups (k-means on tangent features
              re-centered within the unit); clusters are named by a one-to-one Hungarian assignment that maximizes
              the summed log-probabilities of the zero model
  Units: the whole wear (all 120 strokes; transductive over the wear) or one recording (its 20 strokes).
Calibration-size sweep (labeled strokes from the same recording's prefix, training-free): k = 1 or 2 strokes
per class from the prefix, nearest centroid on tangent features (lags 0..30 step 5, 0.5-12 Hz).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from scripts.analysis.explore_w02_fewshot import classify, feat_cov, summarize, tangent
from scripts.analysis.explore_w02_zerocal import LAGS, derotate
from src import v21_data


def recentered(covs, masks):
    out = np.empty((len(covs), covs.shape[1] * (covs.shape[1] + 1) // 2))
    for mask in masks:
        out[mask] = tangent(covs[mask], covs[mask].mean(0))
    return out


def label_free(data, unit, derot="harm-", seed=0, null_seed=None):
    covs = feat_cov(derotate(data["X"], data["position"].astype(float), derot), lags=LAGS)
    wear_masks = [data["wear_group"] == w for w in np.unique(data["wear_group"])]
    features = recentered(covs, wear_masks)
    labels = data["y"].copy()
    if null_seed is not None:
        rng = np.random.default_rng(null_seed)
        for group in np.unique(data["group"]):
            mask = data["group"] == group
            labels[mask] = rng.permutation(4)[labels[mask]]
    zero = np.full(len(labels), -1)
    clustered = np.full(len(labels), -1)
    for wear in np.unique(data["wear_group"]):
        test_wear = data["wear_group"] == wear
        train = ~test_wear
        mean, std = features[train].mean(0), features[train].std(0) + 1e-6
        model = LogisticRegression(C=0.1, max_iter=3000).fit((features[train] - mean) / std, labels[train])
        logp = model.predict_log_proba((features - mean) / std)
        zero[test_wear] = logp[test_wear].argmax(1)
        units = [test_wear] if unit == "wear" else [data["group"] == g for g in np.unique(data["group"][test_wear])]
        for mask in units:
            local = tangent(covs[mask], covs[mask].mean(0))
            local = (local - local.mean(0)) / (local.std(0) + 1e-6)
            assignment = KMeans(4, n_init=20, random_state=seed).fit_predict(local)
            cost = np.zeros((4, 4))
            for cluster in range(4):
                cost[cluster] = -logp[mask][assignment == cluster].sum(0)
            rows, cols = linear_sum_assignment(cost)
            mapping = dict(zip(rows, cols))
            clustered[mask] = [mapping[a] for a in assignment]
    return zero, clustered


def calibration_sweep(data, shots):
    covs = feat_cov(data["X"], lags=LAGS)
    predicted = np.full(len(data["y"]), -1)
    for group in np.unique(data["group"]):
        prefix = np.flatnonzero((data["group"] == group) & data["is_calibration"])
        body = np.flatnonzero((data["group"] == group) & ~data["is_calibration"])
        support = np.concatenate([prefix[data["y"][prefix] == c][:shots] for c in range(4)])
        ref = covs[support].mean(0)
        predicted[body] = classify("nc", tangent(covs[support], ref), data["y"][support], tangent(covs[body], ref))
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/w02_dev_20261015/dataset.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = v21_data.load_w02(args.dataset)
    if set(np.unique(data["wear_group"])) & {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}:
        raise RuntimeError("sealed wear present in the development dataset")
    rows = {}
    for unit in ("wear", "recording"):
        for derot in ("none", "harm-", "harm+", "cyc-"):
            zero, clustered = label_free(data, unit, derot)
            _, clustered_null = label_free(data, unit, derot, null_seed=0)
            for name, pred in (("zero", zero), ("cluster", clustered), ("cluster_null", clustered_null)):
                s = summarize(data, pred)
                rows[f"{unit}/{derot}/{name}"] = s
                print(f"{unit:9s} {derot:5s} {name:12s} F1 {s['macro_f1']:.3f} meanW {s['mean_wear_f1']:.3f} "
                      f"minW {s['min_wear_f1']:.3f} minRec {s['min_recall']:.2f} pred {s['pred_counts']}",
                      flush=True)
            print("   per wear (cluster):", {k: round(v, 2) for k, v in rows[f'{unit}/{derot}/cluster']['per_wear'].items()})
    for shots in (1, 2):
        s = summarize(data, calibration_sweep(data, shots))
        rows[f"calibration/{shots}shot"] = s
        print(f"calibration {shots} stroke(s) per class: F1 {s['macro_f1']:.3f} minW {s['min_wear_f1']:.3f} "
              f"minRec {s['min_recall']:.2f}")
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
