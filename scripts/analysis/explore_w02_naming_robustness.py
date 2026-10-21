"""Exploratory robustness check of k-shot cluster naming on the W02 development wears (balanced-batch assumption).

explore_w02_kshot_naming clustered all 20 strokes of a recording: exactly five per class, k = 4 known. This script
asks whether cluster_zero / cluster_flat survive smaller, incomplete or unbalanced unlabeled pools. Leave-one-wear-out
over the nine development wears; the sealed wears are refused.

Per recording, the pool clustered and named is
  support strokes (first prefix stroke of each labeled class)
  + the other prefix strokes, unlabeled            (--prefix with)   or nothing   (--prefix without)
  + a perturbed set of body strokes (the only strokes scored).
Body perturbations (body strokes in recording order):
  full              all 12 body strokes
  first{m}          the first m body strokes (m = 4, 8): classes need not be present or balanced
  drop{pp}_s{seed}  a random pp% of the body strokes removed (pp = 25, 50; seeds from --drop-seeds)
  dropclass_{C}_{k} the first k body strokes of class C removed (k = 1, 2, 3; k = 3 removes C from the body)
Cluster count: fixed4 (k-means, k = 4) or silhouette (k in 2..4 chosen by silhouette on the pool).
Naming: as explore_w02_kshot_naming, generalized to k <= 4 clusters (rectangular Hungarian assignment; each cluster
gets a distinct class, a labeled stroke forbids any other name for its cluster). cluster_zero adds the zero-model
log-probabilities; cluster_flat uses the support constraints only.

Zero model: as explore_w02_kshot_naming (logistic regression on the training wears, cyclic ring shift
-round(theta/90), training recordings re-centered at their own mean). The held-out pool is re-centered at the mean of
the pool only, so no stroke outside the pool is used. k-means features: tangent at the pool mean, z-scored within
the pool.

Labeled class subsets: --subsets (default eight pre-specified subsets, b = 0..4). Matched nulls (--null-seeds): zero
model retrained with block-permuted training labels and support labels relabeled by a seeded four-class permutation.

Reported per method x condition x subset: accuracy, macro-F1 over the classes present among scored strokes, per-class
recall, prediction counts, per-wear and per-position accuracy/F1, naming failures, support collisions (recordings
where differently labeled support strokes share a cluster), chosen cluster counts, null scores and null prediction
counts. Per-stroke real and null predictions are saved to predictions.npz. No bootstrap intervals.

Sanity check: the full / with / fixed4 rows use the same pool, features, support strokes and zero model as
explore_w02_kshot_naming, so they should reproduce its cluster_zero and cluster_flat rows for the same subsets.
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, silhouette_score

from scripts.analysis.explore_w02_kshot_naming import PENALTY, PERMUTATIONS, per_recording_tangent, sha256
from src import v21_data, w02_cov_fewshot as recipes

SEALED_WEARS = {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}
METHODS = ("cluster_zero", "cluster_flat")
DEFAULT_SUBSETS = ("-", "UP", "UP+DOWN", "UP+RIGHT", "DOWN+RIGHT", "UP+DOWN+LEFT", "UP+DOWN+RIGHT",
                   "UP+DOWN+LEFT+RIGHT")


def parse_subset(text: str) -> tuple[int, ...]:
    return () if text == "-" else tuple(sorted(v21_data.CLASSES.index(name) for name in text.split("+")))


def body_conditions(drop_seeds):
    yield "full", ("full", None)
    for m in (4, 8):
        yield f"first{m}", ("first", m)
    for fraction in (0.25, 0.5):
        for seed in drop_seeds:
            yield f"drop{round(fraction * 100)}_s{seed}", ("drop", (fraction, seed))
    for c, name in enumerate(v21_data.CLASSES):
        for k in (1, 2, 3):
            yield f"dropclass_{name}_{k}", ("dropclass", (c, k))


def perturb(body, y, kind, param, record_index):
    if kind == "full":
        return body
    if kind == "first":
        return body[:param]
    if kind == "drop":
        fraction, seed = param
        rng = np.random.default_rng([seed, record_index])
        keep = np.sort(rng.choice(len(body), size=len(body) - round(fraction * len(body)), replace=False))
        return body[keep]
    c, k = param
    removed = body[y[body] == c][:k]
    return body[~np.isin(body, removed)]


def cluster(features, k_mode, seed):
    local = (features - features.mean(0)) / (features.std(0) + 1e-6)
    if k_mode == "fixed4":
        return KMeans(4, n_init=20, random_state=seed).fit_predict(local)
    best = None
    for k in range(2, min(4, len(local) - 1) + 1):
        assignment = KMeans(k, n_init=20, random_state=seed).fit_predict(local)
        score = silhouette_score(local, assignment)
        if best is None or score > best[0]:
            best = (score, assignment)
    return best[1]


def name_clusters(assignment, logp, support_local, support_labels, prior):
    n_clusters = assignment.max() + 1
    cost = np.zeros((n_clusters, 4))
    for k in range(n_clusters):
        cost[k] = -prior * logp[assignment == k].sum(0)
    for index, label in zip(support_local, support_labels):
        cost[assignment[index]] += PENALTY
        cost[assignment[index], label] -= PENALTY
    rows, cols = linear_sum_assignment(cost)
    return dict(zip(rows, cols))


def report(data, predicted) -> dict:
    scored = predicted >= 0
    y, p = data["y"][scored], predicted[scored]
    wears, positions, groups = data["wear_group"][scored], data["position"][scored], data["group"][scored]
    present = sorted(set(y.tolist()))

    def f1(mask):
        labels = sorted(set(y[mask].tolist()))
        return float(f1_score(y[mask], p[mask], average="macro", labels=labels, zero_division=0))

    failures = []
    for group in np.unique(groups):
        m = groups == group
        best = max(np.mean(perm[p[m]] == y[m]) for perm in PERMUTATIONS)
        if best - np.mean(p[m] == y[m]) >= 0.25:
            failures.append(str(group))
    everything = np.ones(len(y), bool)
    return {"accuracy": float(np.mean(y == p)), "macro_f1_present": f1(everything),
            "classes_present": [v21_data.CLASSES[c] for c in present],
            "recall": {name: (float(np.mean(p[y == c] == c)) if np.any(y == c) else None)
                       for c, name in enumerate(v21_data.CLASSES)},
            "true_counts": dict(zip(v21_data.CLASSES, np.bincount(y, minlength=4).tolist())),
            "prediction_counts": dict(zip(v21_data.CLASSES, np.bincount(p, minlength=4).tolist())),
            "per_wear": {w: {"accuracy": float(np.mean(y[wears == w] == p[wears == w])), "f1": f1(wears == w)}
                         for w in np.unique(wears)},
            "per_position": {int(a): {"accuracy": float(np.mean(y[positions == a] == p[positions == a])),
                                      "f1": f1(positions == a)} for a in np.unique(positions)},
            "naming_failures": failures, "n_recordings": int(len(np.unique(groups))), "n_scored": int(scored.sum())}


def null_summary(data, predicted) -> dict:
    scored = predicted >= 0
    y, p = data["y"][scored], predicted[scored]
    return {"accuracy": float(np.mean(y == p)),
            "macro_f1_present": float(f1_score(y, p, average="macro", labels=sorted(set(y.tolist())),
                                               zero_division=0)),
            "prediction_counts": dict(zip(v21_data.CLASSES, np.bincount(p, minlength=4).tolist()))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/w02_dev_20261015/dataset.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subsets", nargs="*", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--prefix", nargs="*", choices=("with", "without"), default=["with", "without"])
    parser.add_argument("--k-modes", nargs="*", choices=("fixed4", "silhouette"), default=["fixed4", "silhouette"])
    parser.add_argument("--drop-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--kmeans-seed", type=int, default=0)
    args = parser.parse_args()
    subsets = {text: parse_subset(text) for text in args.subsets}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = v21_data.load_w02(args.dataset)
    if set(np.unique(data["wear_group"])) & SEALED_WEARS:
        raise RuntimeError("sealed wear present in the development dataset")
    conditions = list(body_conditions(args.drop_seeds))
    run = {"started_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset),
           "dataset_sha256": sha256(args.dataset), "script_sha256": sha256(Path(__file__)),
           "kshot_source_sha256": sha256(Path(__file__).with_name("explore_w02_kshot_naming.py")),
           "recipe_source_sha256": sha256(Path(recipes.__file__)), "class_order": list(v21_data.CLASSES),
           "subsets": args.subsets, "prefix": args.prefix, "k_modes": args.k_modes,
           "body_conditions": [name for name, _ in conditions], "drop_seeds": args.drop_seeds,
           "null_seeds": args.null_seeds, "kmeans_seed": args.kmeans_seed,
           "test_access": "development wears only (leave-one-wear-out); sealed wears W07/W10/W12 refused; exploratory"}
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2))

    y, groups, wears = data["y"], data["group"], data["wear_group"]
    covs = recipes.covariances(data["X"])
    zero_covs = recipes.covariances(recipes.cyclic_canonical(data["X"], data["position"]))
    zero_train = per_recording_tangent(zero_covs, groups)
    label_sets = {None: y} | {s: v21_data.block_permuted_labels(data, s) for s in args.null_seeds}
    zero_models = {}
    for wear in np.unique(wears):
        train = wears != wear
        mean, std = zero_train[train].mean(0), zero_train[train].std(0) + 1e-6
        for seed, labels in label_sets.items():
            model = LogisticRegression(C=0.1, max_iter=3000).fit((zero_train[train] - mean) / std, labels[train])
            zero_models[wear, seed] = (mean, std, model)
        print(f"fold {wear} ready", flush=True)

    recordings = []
    for record_index, group in enumerate(np.unique(groups)):
        idx = np.flatnonzero(groups == group)
        prefix = idx[data["is_calibration"][idx]]  # dataset order, as support_for in explore_w02_kshot_naming
        ordered = idx[np.argsort(data["repetition"][idx], kind="stable")]
        recordings.append({"index": record_index, "group": group, "wear": wears[idx[0]], "prefix": prefix,
                           "body": ordered[~data["is_calibration"][ordered]],
                           "first_of_class": {c: prefix[y[prefix] == c][0] for c in range(4)}})

    cluster_cache, logp_cache = {}, {}

    def pool_clusters(pool, k_mode):
        key = (tuple(pool), k_mode)
        if key not in cluster_cache:
            features = recipes.tangent(covs[pool], covs[pool].mean(0))
            cluster_cache[key] = cluster(features, k_mode, args.kmeans_seed)
        return cluster_cache[key]

    def pool_logp(pool, wear, seed):
        key = (tuple(pool), seed)
        if key not in logp_cache:
            mean, std, model = zero_models[wear, seed]
            features = recipes.tangent(zero_covs[pool], zero_covs[pool].mean(0))
            logp_cache[key] = model.predict_log_proba((features - mean) / std)
        return logp_cache[key]

    seeds = [None] + list(args.null_seeds)
    rows, saved = [], {}
    for condition, (kind, param) in conditions:
        bodies = {r["group"]: perturb(r["body"], y, kind, param, r["index"]) for r in recordings}
        for prefix_mode in args.prefix:
            for k_mode in args.k_modes:
                for subset_name, classes in subsets.items():
                    predicted = {(m, s): np.full(len(y), -1) for m in METHODS for s in seeds}
                    collisions, cluster_counts = 0, Counter()
                    for r in recordings:
                        support = np.array([r["first_of_class"][c] for c in classes], dtype=int)
                        body = bodies[r["group"]]
                        unlabeled_prefix = r["prefix"] if prefix_mode == "with" else np.array([], dtype=int)
                        pool = np.unique(np.concatenate([support, unlabeled_prefix, body]))
                        assignment = pool_clusters(pool, k_mode)
                        cluster_counts[int(assignment.max() + 1)] += 1
                        support_local = np.searchsorted(pool, support)
                        if len({assignment[i] for i in support_local}) < len(support_local):
                            collisions += 1
                        body_local = np.searchsorted(pool, body)
                        for seed in seeds:
                            labels = np.array(classes, dtype=int)
                            if seed is not None:
                                labels = np.random.default_rng([seed, r["index"]]).permutation(4)[labels]
                            logp = pool_logp(pool, r["wear"], seed)
                            for method in METHODS:
                                mapping = name_clusters(assignment, logp, support_local, labels,
                                                        prior=1.0 if method == "cluster_zero" else 0.0)
                                predicted[method, seed][body] = [mapping[a] for a in assignment[body_local]]
                    for method in METHODS:
                        row = {"method": method, "condition": condition, "prefix": prefix_mode, "k_mode": k_mode,
                               "subset": subset_name, "budget": len(classes), **report(data, predicted[method, None]),
                               "support_collisions": collisions,
                               "cluster_counts": {str(k): v for k, v in sorted(cluster_counts.items())}}
                        nulls = [null_summary(data, predicted[method, s]) for s in args.null_seeds]
                        row["null_accuracy"] = [n["accuracy"] for n in nulls]
                        row["null_macro_f1_present"] = [n["macro_f1_present"] for n in nulls]
                        row["null_prediction_counts"] = [n["prediction_counts"] for n in nulls]
                        key = f"{method}|{condition}|{prefix_mode}|{k_mode}|{subset_name}"
                        saved[key] = predicted[method, None].astype(np.int8)
                        for s in args.null_seeds:
                            saved[f"{key}|null{s}"] = predicted[method, s].astype(np.int8)
                        rows.append(row)
                        print(f"{method:12s} {condition:18s} {prefix_mode:7s} {k_mode:10s} {subset_name:18s} "
                              f"acc {row['accuracy']:.3f} F1 {row['macro_f1_present']:.3f} "
                              f"pred {list(row['prediction_counts'].values())} "
                              f"fail {len(row['naming_failures'])}/{row['n_recordings']} coll {collisions} "
                              f"k {row['cluster_counts']} null {np.mean(row['null_accuracy']):.3f}", flush=True)

    # Summary: accuracy per condition family (drop seeds averaged), one column per subset.
    summary = {}
    for row in rows:
        family = row["condition"].split("_s")[0] if row["condition"].startswith("drop") and "_s" in row["condition"] \
            else row["condition"]
        key = (row["method"], family, row["prefix"], row["k_mode"], row["subset"])
        summary.setdefault(key, []).append((row["accuracy"], np.mean(row["null_accuracy"])))
    table = [{"method": k[0], "condition": k[1], "prefix": k[2], "k_mode": k[3], "subset": k[4],
              "mean_accuracy": float(np.mean([v[0] for v in vals])), "min_accuracy": float(np.min([v[0] for v in vals])),
              "mean_null_accuracy": float(np.mean([v[1] for v in vals])), "n_runs": len(vals)}
             for k, vals in summary.items()]
    print("\nmean accuracy (drop seeds averaged); columns:", " | ".join(args.subsets))
    for method in METHODS:
        for prefix_mode in args.prefix:
            for k_mode in args.k_modes:
                families = dict.fromkeys(t["condition"] for t in table)
                for family in families:
                    cells = {t["subset"]: t["mean_accuracy"] for t in table if t["method"] == method
                             and t["condition"] == family and t["prefix"] == prefix_mode and t["k_mode"] == k_mode}
                    print(f"{method:12s} {prefix_mode:7s} {k_mode:10s} {family:18s} "
                          + " ".join(f"{cells[s]:.3f}" for s in args.subsets))
    np.savez_compressed(args.output_dir / "predictions.npz", y=y.astype(np.int8), **saved)
    (args.output_dir / "results.json").write_text(json.dumps({"rows": rows, "summary": table}, indent=1))


if __name__ == "__main__":
    main()
