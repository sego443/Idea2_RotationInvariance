"""Exploratory family A: calibration-only prototypes with online adaptation (W02 development wears, causal streams).

The startup-curve run showed that 4 calibration strokes classify well at first (nearest centroid 0.93 on the first
10 stream strokes) but decay over the wear (0.55-0.60 by the sixth recording), and that population prototypes from
other wears are wrong. Here prototypes start from the calibration strokes only and adapt during use.

Calibration: K = 4 (one stroke per class; draws 0 and 1) and K = 8 (both prefix strokes of each class, reference).
Feature space:
  tangent    tangent space at the mean of the calibration covariances (reproduces nc4 with metric anchor_z)
  logeuclid  log-Euclidean vectors (no reference), optionally re-centered online
Metric (all distances are Euclidean after the transform):
  anchor_z  z-score with the calibration strokes' mean/std (nc4 rule)
  pop_z     divide by the per-feature std of the training wears (labels unused; leave-one-wear-out)
  mahal     PCA 30 + Ledoit-Wolf within-class precision from the training wears (training labels used)
  none      plain Euclidean
Update after each prediction (prototype of the predicted class only; margin = (d2nd - d1st) / d2nd):
  none; cum_g{g}: cumulative mean (calibration strokes count 1 each) when margin >= g;
  ema_a{a}_g{g}: exponential moving average with rate a when margin >= g
Re-centering (logeuclid only): rc{b}: subtract an exponential moving average (rate b) of all past stream features,
initialized at the calibration mean; prototypes are stored relative to it. Unsupervised common-mode drift tracking.
Matched null: calibration labels replaced by a seeded random permutation of the classes (per wear and seed).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from scripts.analysis import w02_stream_lib as lib

UPDATES = (["none"] + [f"cum_g{g}" for g in (0.0, 0.1, 0.2, 0.3)]
           + [f"ema_a{a}_g{g}" for a in (0.1, 0.2, 0.3) for g in (0.0, 0.2)])


def population(features_by_recording, data, train_mask):
    """Label-free std and within-class whitening from training recordings (features centered per recording)."""
    rows = np.flatnonzero(train_mask)
    x = features_by_recording[rows]
    std = x.std(0) + 1e-6
    pca = PCA(n_components=30).fit(x)
    z = pca.transform(x)
    y, groups = data["y"][rows], data["group"][rows]
    residuals = [z[(groups == g) & (y == c)] - z[(groups == g) & (y == c)].mean(0)
                 for g in np.unique(groups) for c in range(4)]
    precision = LedoitWolf().fit(np.concatenate(residuals)).precision_
    return {"std": std, "pca": pca, "whiten": np.linalg.cholesky(precision)}


def run_stream(x, anchors_x, anchor_y, metric, update, beta, pop):
    """x: raw stream features in order; returns predictions."""
    mu = anchors_x.mean(0) if beta else 0.0
    rel_anchor = anchors_x - mu
    if metric == "anchor_z":
        m, s = rel_anchor.mean(0), rel_anchor.std(0) + 1e-6
        f = lambda v: (v - m) / s  # noqa: E731
    elif metric == "pop_z":
        f = lambda v: v / pop["std"]  # noqa: E731
    elif metric == "mahal":
        f = lambda v: pop["pca"].transform(np.atleast_2d(v)) @ pop["whiten"]  # noqa: E731
    else:
        f = lambda v: v  # noqa: E731
    fa = np.atleast_2d(f(rel_anchor))
    protos = np.stack([fa[anchor_y == c].mean(0) for c in range(4)])
    counts = np.bincount(anchor_y, minlength=4).astype(float)
    kind, rate, gate = "none", 0.0, 0.0
    if update.startswith("cum"):
        kind, gate = "cum", float(update.split("_g")[1])
    elif update.startswith("ema"):
        kind, rate, gate = "ema", float(update.split("_a")[1].split("_g")[0]), float(update.split("_g")[1])
    pred = np.empty(len(x), dtype=int)
    for t in range(len(x)):
        v = np.atleast_2d(f(x[t] - mu))[0]
        d2 = ((protos - v) ** 2).sum(1)
        order = np.argsort(d2)
        c = order[0]
        pred[t] = c
        margin = (d2[order[1]] - d2[c]) / max(d2[order[1]], 1e-12)
        if kind != "none" and margin >= gate:
            if kind == "cum":
                counts[c] += 1
                protos[c] += (v - protos[c]) / counts[c]
            else:
                protos[c] += rate * (v - protos[c])
        if beta:
            mu = (1 - beta) * mu + beta * x[t]
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=lib.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    data = lib.load_dev(args.dataset)
    lib.start_run(args.output_dir, __file__, args.dataset, {"updates": UPDATES, "null_seeds": args.null_seeds})
    streams, minutes = lib.wear_streams(data), lib.recording_minutes(data)
    covs = lib.covariances(data["X"])
    logeuclid = lib.log_euclid(covs)
    centered = {"tangent": np.empty((len(covs), logeuclid.shape[1])), "logeuclid": np.empty_like(logeuclid)}
    for group in np.unique(data["group"]):
        mask = data["group"] == group
        centered["tangent"][mask] = lib.tangent(covs[mask], covs[mask].mean(0))
        centered["logeuclid"][mask] = logeuclid[mask] - logeuclid[mask].mean(0)
    wears = list(streams)
    pops = {(space, w): population(centered[space], data, data["wear_group"] != w)
            for space in centered for w in wears}
    print("population statistics ready", flush=True)

    configs = [("tangent", m, u, 0.0) for m in ("anchor_z", "pop_z", "mahal", "none") for u in UPDATES]
    configs += [("logeuclid", m, u, b) for m in ("anchor_z", "pop_z", "mahal", "none") for u in UPDATES
                for b in (0.0, 0.02, 0.05, 0.1)]
    calibrations = [(4, 0), (4, 1), (8, "both")]
    rows, saved = [], {}
    for space, metric, update, beta in configs:
        for k, draw in calibrations:
            results = {}
            for seed in [None] + list(args.null_seeds):
                pieces, preds = [], []
                for w_i, wear in enumerate(wears):
                    anchors, anchor_y, rest = lib.calibration(streams[wear], data["y"], range(4), draw)
                    if seed is not None:
                        anchor_y = np.random.default_rng([seed, w_i]).permutation(4)[anchor_y]
                    if space == "tangent":
                        ref = covs[anchors].mean(0)
                        x, ax = lib.tangent(covs[rest], ref), lib.tangent(covs[anchors], ref)
                    else:
                        x, ax = logeuclid[rest], logeuclid[anchors]
                    pieces.append((wear, rest))
                    preds.append(run_stream(x, ax, anchor_y, metric, update, beta, pops[space, wear]))
                results[seed] = (pieces, np.concatenate(preds))
            pieces, pred = results[None]
            idx, positions, ranks, mins = lib.stream_context(data, streams, pieces, minutes)
            row = {"space": space, "metric": metric, "update": update, "recenter": beta, "startup_actions": k,
                   "draw": str(draw), **lib.stream_report(data, idx, pred, positions, ranks, mins)}
            row["null_accuracy"] = [float(np.mean(data["y"][idx] == results[s][1])) for s in args.null_seeds]
            row["null_prediction_counts"] = [np.bincount(results[s][1], minlength=4).tolist() for s in args.null_seeds]
            key = f"{space}|{metric}|{update}|rc{beta}|K{k}|d{draw}"
            full = np.full(len(data["y"]), -1, np.int8)
            full[idx] = pred
            saved[key] = full
            rows.append(row)
            print(f"{space:9s} {metric:8s} {update:14s} rc{beta:<4} K={k} d{draw!s:4s} {lib.short(row)}", flush=True)

    print("\nTop 25 K=4 configurations (mean of draws 0 and 1):")
    groups = {}
    for row in rows:
        if row["startup_actions"] == 4:
            groups.setdefault((row["space"], row["metric"], row["update"], row["recenter"]), []).append(row)
    ranked = sorted(groups.items(), key=lambda kv: -np.mean([r["accuracy"] for r in kv[1]]))
    for key, rs in ranked[:25]:
        pos = np.mean([list(r["accuracy_by_position"].values()) for r in rs], axis=0)
        rec = np.mean([list(r["accuracy_by_recording"].values()) for r in rs], axis=0)
        print(f"{key} acc {np.mean([r['accuracy'] for r in rs]):.3f} pos {np.round(pos, 2).tolist()} "
              f"rec {np.round(rec, 2).tolist()} null {np.mean([r['null_accuracy'] for r in rs]):.3f}")
    lib.finish_run(args.output_dir, rows, saved)


if __name__ == "__main__":
    main()
