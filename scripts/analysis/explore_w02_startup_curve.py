"""Exploratory causal startup curve on the W02 development wears: K calibration actions, then streaming recognition.

Question: how few startup actions (K = 0..4 strokes, labeled or not) are needed before recognition is usable, when
every later stroke is predicted from the calibration and past strokes only? Leave-one-wear-out over the nine
development wears; the sealed wears are refused.

Protocol per held-out wear (one donning, about 120 strokes):
  calibration  K strokes taken from the prefix of the wear's first recording: for each class in the labeled subset,
               occurrence `draw` (0 or 1) of that class in temporal order. They are treated as performed first.
  stream       every other stroke of the wear (all recordings, prefix and body) in recording order; each one is
               predicted before the next is seen. All stream strokes are scored.
Training wears use the same construction on every training recording (anchors from its own prefix, both draws),
so the cross-wear model sees features of the same form as the held-out wear.

Features: lagged covariance (src/w02_cov_fewshot.py: 0.5-12 Hz, lags 0..30 step 5, 28 x 28), tangent space at a
reference covariance:
  anchor   (K >= 1) the mean of the K calibration covariances
  global   (K = 0) the mean covariance of the training wears; nothing from the held-out wear
  running  (K = 0) the mean of the stream strokes seen so far (the first stroke uses global); training recordings
           use the mean of their own 20 strokes
Optional cyclic derotation by the measured band angle (--derotate cyclic).

Methods:
  logreg         logistic regression (C = 0.1) on the training features, standardized with training statistics
  pop_lda        PCA 30 of the training features; class prototypes = training class means; within-class precision
                 from training residuals (recording-class means removed, Ledoit-Wolf); nearest prototype by
                 Mahalanobis distance. A calibrated class's prototype is blended with its calibration stroke
                 ((n0 * population + stroke) / (n0 + 1), n0 = --prior-count).
  pop_lda_adapt  pop_lda whose prototypes are updated after each prediction with the stroke's soft assignment
                 (posterior weights; prototypes start with pseudo-count n0, plus 1 for a calibrated class)
  nc4            (K = 4 only) training-free nearest centroid on the 4 calibration strokes (tangent at their mean,
                 z-scored with their statistics): the startup-only baseline of the W02 development record
Matched nulls (--null-seeds): labels block-permuted within each recording (v21_data.block_permuted_labels) for both
the training wears and the held-out wear's calibration selection; scored against the true labels.

Reported per method x reference x labeled subset x draw: accuracy, macro-F1, per-class recall, prediction counts,
accuracy by stream position (1-5, 6-10, 11-20, 21-40, 41+), by recording within the wear, per wear and per band
angle, plus null accuracy, null first-10 accuracy and null prediction counts. Predictions are saved per stroke.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from src import v21_data, w02_cov_fewshot as recipes

SEALED_WEARS = {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}
BINS = ((1, 5), (6, 10), (11, 20), (21, 40), (41, 10_000))
METHODS = ("logreg", "pop_lda", "pop_lda_adapt", "nc4")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mahalanobis(z, means, precision):
    diff = z[:, None, :] - means[None]
    return np.einsum("nci,ij,ncj->nc", diff, precision, diff)


def anchors_of(prefix, labels, classes, draw):
    return np.array([prefix[labels[prefix] == c][draw] for c in classes], dtype=int)


def fit_models(x, y, units):
    mean, std = x.mean(0), x.std(0) + 1e-6
    logreg = LogisticRegression(C=0.1, max_iter=3000).fit((x - mean) / std, y)
    pca = PCA(n_components=30).fit(x)
    z = pca.transform(x)
    prototypes = np.stack([z[y == c].mean(0) for c in range(4)])
    residuals = [z[(units == u) & (y == c)] - z[(units == u) & (y == c)].mean(0)
                 for u in np.unique(units) for c in range(4) if np.any((units == u) & (y == c))]
    precision = LedoitWolf().fit(np.concatenate(residuals)).precision_
    return {"mean": mean, "std": std, "logreg": logreg, "pca": pca, "prototypes": prototypes, "precision": precision}


def predict_stream(method, models, features, anchor_features, classes, prior_count):
    if method == "logreg":
        return models["logreg"].predict((features - models["mean"]) / models["std"])
    if method == "nc4":
        return recipes.nearest_centroid(anchor_features, np.array(classes), features)
    z = models["pca"].transform(features)
    counts = np.full(4, float(prior_count))
    sums = models["prototypes"] * prior_count
    if len(classes):
        za = models["pca"].transform(anchor_features)
        for k, c in enumerate(classes):
            sums[c] += za[k]
            counts[c] += 1
    predicted = np.empty(len(z), dtype=int)
    for t in range(len(z)):
        d2 = mahalanobis(z[t:t + 1], sums / counts[:, None], models["precision"])[0]
        predicted[t] = d2.argmin()
        if method == "pop_lda_adapt":
            w = np.exp(-(d2 - d2.min()) / 2)
            w /= w.sum()
            sums += w[:, None] * z[t]
            counts += w
    return predicted


def stream_report(y, p, positions, wears, angles, rec_rank) -> dict:
    return {"accuracy": float(np.mean(y == p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=list(range(4)))),
            "recall": {name: float(np.mean(p[y == c] == c)) for c, name in enumerate(v21_data.CLASSES)},
            "prediction_counts": dict(zip(v21_data.CLASSES, np.bincount(p, minlength=4).tolist())),
            "accuracy_by_position": {f"{a}-{b}" if b < 10_000 else f"{a}+":
                                     float(np.mean(y[(positions >= a) & (positions <= b)]
                                                   == p[(positions >= a) & (positions <= b)]))
                                     for a, b in BINS},
            "accuracy_by_recording": {int(r) + 1: float(np.mean(y[rec_rank == r] == p[rec_rank == r]))
                                      for r in np.unique(rec_rank)},
            "per_wear_accuracy": {w: float(np.mean(y[wears == w] == p[wears == w])) for w in np.unique(wears)},
            "per_angle_accuracy": {int(a): float(np.mean(y[angles == a] == p[angles == a])) for a in np.unique(angles)},
            "n_scored": int(len(y))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/w02_dev_20261015/dataset.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--derotate", choices=("none", "cyclic"), default="none")
    parser.add_argument("--prior-count", type=float, default=3.0)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = v21_data.load_w02(args.dataset)
    if set(np.unique(data["wear_group"])) & SEALED_WEARS:
        raise RuntimeError("sealed wear present in the development dataset")
    run = {"started_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset),
           "dataset_sha256": sha256(args.dataset), "script_sha256": sha256(Path(__file__)),
           "recipe_source_sha256": sha256(Path(recipes.__file__)), "class_order": list(v21_data.CLASSES),
           "derotate": args.derotate, "prior_count": args.prior_count, "null_seeds": args.null_seeds,
           "test_access": "development wears only (leave-one-wear-out); sealed wears W07/W10/W12 refused; exploratory"}
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2))

    y, groups, wears = data["y"], data["group"], data["wear_group"]
    windows = data["X"] if args.derotate == "none" else recipes.cyclic_canonical(data["X"], data["position"])
    covs = recipes.covariances(windows)

    recordings = {}
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        idx = idx[np.argsort(data["repetition"][idx], kind="stable")]
        recordings[group] = {"idx": idx, "prefix": idx[data["is_calibration"][idx]]}
    wear_order, wear_recordings = {}, {}
    for wear in np.unique(wears):
        idx = np.flatnonzero(wears == wear)
        idx = idx[np.argsort(data["repetition"][idx], kind="stable")]
        wear_order[wear] = idx
        wear_recordings[wear] = list(dict.fromkeys(groups[idx]))

    label_sets = {None: y} | {s: v21_data.block_permuted_labels(data, s) for s in args.null_seeds}
    configs = [((), "global"), ((), "running")] + [
        (classes, "anchor") for b in range(1, 5) for classes in itertools.combinations(range(4), b)]

    feature_cache = {}  # (seed, classes, reference, group) -> tangent features; fold-independent except global

    def training_set(seed, classes, reference, train_wears, global_ref):
        labels = label_sets[seed]
        x, ys, units = [], [], []
        for wear in train_wears:
            for group in wear_recordings[wear]:
                rec = recordings[group]
                key = (seed, classes, reference, group)
                if reference == "global":
                    blocks = [recipes.tangent(covs[rec["idx"]], global_ref)]
                elif key in feature_cache:
                    blocks = feature_cache[key]
                else:
                    if reference == "anchor":
                        refs = [covs[anchors_of(rec["prefix"], labels, classes, d)].mean(0) for d in (0, 1)]
                    else:
                        refs = [covs[rec["idx"]].mean(0)]
                    blocks = feature_cache[key] = [recipes.tangent(covs[rec["idx"]], ref) for ref in refs]
                for d, block in enumerate(blocks):
                    x.append(block)
                    ys.append(labels[rec["idx"]])
                    units.append(np.full(len(rec["idx"]), f"{group}/{d}"))
        return np.concatenate(x), np.concatenate(ys), np.concatenate(units)

    rows, saved = [], {}
    for classes, reference in configs:
        subset = "+".join(v21_data.CLASSES[c] for c in classes) or "-"
        draws = (0, 1) if classes else (0,)
        methods = [m for m in METHODS if m != "nc4" or len(classes) == 4]
        collected = {(m, d, s): [] for m in methods for d in draws for s in label_sets}
        for wear in np.unique(wears):
            train_wears = [w for w in np.unique(wears) if w != wear]
            train_mask = np.isin(wears, train_wears)
            global_ref = covs[train_mask].mean(0)
            first = recordings[wear_recordings[wear][0]]
            for seed, labels in label_sets.items():
                models = fit_models(*training_set(seed, classes, reference, train_wears, global_ref))
                for draw in draws:
                    anchors = anchors_of(first["prefix"], labels, classes, draw) if classes else np.array([], int)
                    stream = wear_order[wear][~np.isin(wear_order[wear], anchors)]
                    if reference == "anchor":
                        ref = covs[anchors].mean(0)
                        features = recipes.tangent(covs[stream], ref)
                        anchor_features = recipes.tangent(covs[anchors], ref)
                    elif reference == "global":
                        features, anchor_features = recipes.tangent(covs[stream], global_ref), None
                    else:
                        features = np.concatenate([recipes.tangent(covs[stream[t:t + 1]],
                                                                   covs[stream[:t]].mean(0) if t else global_ref)
                                                   for t in range(len(stream))])
                        anchor_features = None
                    for method in methods:
                        p = predict_stream(method, models, features, anchor_features, classes, args.prior_count)
                        collected[method, draw, seed].append((stream, p))
            print(f"{subset:20s} {reference:8s} fold {wear} done", flush=True)

        for method in methods:
            for draw in draws:
                def flatten(seed):
                    stream = np.concatenate([s for s, _ in collected[method, draw, seed]])
                    pred = np.concatenate([p for _, p in collected[method, draw, seed]])
                    positions = np.concatenate([np.arange(1, len(s) + 1) for s, _ in collected[method, draw, seed]])
                    return stream, pred, positions
                stream, pred, positions = flatten(None)
                rank = {g: r for w in wear_recordings for r, g in enumerate(wear_recordings[w])}
                rec_rank = np.array([rank[g] for g in groups[stream]])
                row = {"method": method, "reference": reference, "subset": subset, "startup_actions": len(classes),
                       "draw": draw, **stream_report(y[stream], pred, positions, wears[stream],
                                                     data["position"][stream], rec_rank)}
                key = f"{method}|{reference}|{subset}|d{draw}"
                full = np.full(len(y), -1, dtype=np.int8)
                full[stream] = pred
                saved[key] = full
                nulls = []
                for seed in args.null_seeds:
                    s_stream, s_pred, s_pos = flatten(seed)
                    full = np.full(len(y), -1, dtype=np.int8)
                    full[s_stream] = s_pred
                    saved[f"{key}|null{seed}"] = full
                    nulls.append({"accuracy": float(np.mean(y[s_stream] == s_pred)),
                                  "first10_accuracy": float(np.mean(y[s_stream][s_pos <= 10] == s_pred[s_pos <= 10])),
                                  "prediction_counts": np.bincount(s_pred, minlength=4).tolist()})
                row["null"] = nulls
                rows.append(row)
                bins = row["accuracy_by_position"]
                print(f"{method:14s} {reference:8s} K={len(classes)} {subset:20s} d{draw} acc {row['accuracy']:.3f} "
                      f"F1 {row['macro_f1']:.3f} by position {[round(v, 2) for v in bins.values()]} "
                      f"pred {list(row['prediction_counts'].values())} "
                      f"null {np.mean([n['accuracy'] for n in nulls]):.3f}", flush=True)

    print("\nmean over labeled subsets and draws: overall accuracy | accuracy at stream positions 1-5, 6-10, 11-20")
    summary = []
    for method in METHODS:
        for reference in ("global", "running", "anchor"):
            for k in range(5):
                sel = [r for r in rows if r["method"] == method and r["reference"] == reference
                       and r["startup_actions"] == k]
                if not sel:
                    continue
                entry = {"method": method, "reference": reference, "startup_actions": k, "n_rows": len(sel),
                         "mean_accuracy": float(np.mean([r["accuracy"] for r in sel])),
                         "min_accuracy": float(np.min([r["accuracy"] for r in sel])),
                         "mean_accuracy_by_position": {b: float(np.mean([r["accuracy_by_position"][b] for r in sel]))
                                                       for b in sel[0]["accuracy_by_position"]},
                         "mean_null_accuracy": float(np.mean([n["accuracy"] for r in sel for n in r["null"]]))}
                summary.append(entry)
                pos = list(entry["mean_accuracy_by_position"].values())
                print(f"{method:14s} {reference:8s} K={k} acc {entry['mean_accuracy']:.3f} "
                      f"(min {entry['min_accuracy']:.3f}) | {pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} "
                      f"null {entry['mean_null_accuracy']:.3f}")
    np.savez_compressed(args.output_dir / "predictions.npz", y=y.astype(np.int8), **saved)
    (args.output_dir / "results.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
