"""Exploratory k-shot naming curve on the W02 development wears: fewer than one labeled stroke per class.

Question: can 0-3 labeled strokes per recording (instead of rec1_borrowed's 4, one per class) keep macro-F1 near
0.93? Leave-one-wear-out over the nine development wears. The sealed wears are refused.

Labeled budget b = 0..4 strokes per recording. For every subset S of b classes (all C(4, b) subsets), the support
is the first prefix stroke of each class in S. All other strokes of the recording are unlabeled. Scoring is on the
recording's 12 body strokes only (the prefix is never scored).

Methods (all transductive over one recording's unlabeled strokes; none can label the first stroke after donning):
  cluster_zero       k-means (4 clusters) on the recording's 20 strokes (tangent at the recording mean, z-scored
                     within the recording). Clusters are named by a one-to-one Hungarian assignment: a labeled
                     stroke forbids naming its cluster anything but its label (large penalty), and the remaining
                     choice maximizes the summed log-probabilities of the zero model.
  cluster_flat       as cluster_zero without the zero-model prior (only the labeled constraints; ties arbitrary).
  anchored_zero      rec1_borrowed generalized to b < 4: Mahalanobis distance under the borrowed within-class
                     metric (src/w02_cov_fewshot.py, PCA 30 + Ledoit-Wolf from the training wears). Labeled classes
                     start at their support stroke; each unlabeled class starts at the zero-model-weighted mean of
                     the body strokes (softmax over the unlabeled classes only). Two soft refinement passes over the
                     body strokes, as rec1_borrowed.
  anchored_farthest  as anchored_zero, but each unlabeled class starts at the body stroke farthest (Mahalanobis)
                     from all means placed so far; after refinement the unlabeled slots are named by a Hungarian
                     assignment on the soft-weighted zero-model log-probabilities.
  cluster_oracle     (budget-independent ceiling) k-means clusters named by the best permutation of the true labels.
At b = 4 the anchored methods reduce to rec1_borrowed; the script checks agreement with src.w02_cov_fewshot.

Zero model (the prior): logistic regression (C = 0.1) on the training wears; ring channels cyclically shifted by
-round(theta/90); tangent at the mean of each recording's strokes (unlabeled; per recording, not per wear).

Matched nulls (seeds from --null-seeds): the zero model is retrained with block-permuted training labels
(v21_data.block_permuted_labels) and every support set is relabeled by a random permutation of the four classes
(seeded per recording). Both use the same seed.

Reported per method x budget x subset: macro-F1, accuracy, per-class recall, prediction counts, per-wear and
per-position (band angle) macro-F1, confusion matrix, naming failures (recordings whose body accuracy is at least
0.25 below their best-permutation accuracy), null macro-F1s; real rows also get a wear-cluster bootstrap 95% CI.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from src import v21_data, w02_cov_fewshot as recipes

SEALED_WEARS = {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}
METHODS = ("cluster_zero", "cluster_flat", "anchored_zero", "anchored_farthest")
PENALTY = 1e6
PERMUTATIONS = np.array(list(itertools.permutations(range(4))))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", labels=list(range(4))))


def report(data, predicted, bootstrap: int) -> dict:
    scored = ~data["is_calibration"]
    y, p = data["y"][scored], predicted[scored]
    wears, positions, groups = data["wear_group"][scored], data["position"][scored], data["group"][scored]
    failures = []
    for group in np.unique(groups):
        m = groups == group
        best = max(np.mean(perm[p[m]] == y[m]) for perm in PERMUTATIONS)
        if best - np.mean(p[m] == y[m]) >= 0.25:
            failures.append(str(group))
    out = {"macro_f1": macro_f1(y, p), "accuracy": float(np.mean(y == p)),
           "recall": {name: float(np.mean(p[y == c] == c)) for c, name in enumerate(v21_data.CLASSES)},
           "prediction_counts": dict(zip(v21_data.CLASSES, np.bincount(p, minlength=4).tolist())),
           "per_wear_f1": {w: macro_f1(y[wears == w], p[wears == w]) for w in np.unique(wears)},
           "per_position_f1": {int(a): macro_f1(y[positions == a], p[positions == a]) for a in np.unique(positions)},
           "confusion_true_rows": confusion_matrix(y, p, labels=list(range(4))).tolist(),
           "naming_failures": failures, "n_recordings": int(len(np.unique(groups))), "n_scored": int(scored.sum())}
    if bootstrap:
        rng = np.random.default_rng(20261015)
        unique = np.unique(wears)
        members = {w: np.flatnonzero(wears == w) for w in unique}
        boot = [macro_f1(y[i], p[i]) for i in (np.concatenate([members[w] for w in rng.choice(unique, len(unique))])
                                               for _ in range(bootstrap))]
        out["bootstrap_95ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return out


def per_recording_tangent(covs, groups):
    out = np.empty((len(covs), covs.shape[1] * (covs.shape[1] + 1) // 2))
    for group in np.unique(groups):
        mask = groups == group
        out[mask] = recipes.tangent(covs[mask], covs[mask].mean(0))
    return out


def zero_log_proba(features, labels, train):
    mean, std = features[train].mean(0), features[train].std(0) + 1e-6
    model = LogisticRegression(C=0.1, max_iter=3000).fit((features[train] - mean) / std, labels[train])
    return model.predict_log_proba((features - mean) / std)


def support_for(data, group, classes):
    prefix = np.flatnonzero((data["group"] == group) & data["is_calibration"])
    return np.array([prefix[data["y"][prefix] == c][0] for c in classes], dtype=int)


def name_clusters(assignment, logp, support_local, support_labels, prior):
    cost = np.zeros((4, 4))
    for cluster in range(4):
        cost[cluster] = -prior * logp[assignment == cluster].sum(0)
    for index, label in zip(support_local, support_labels):
        cost[assignment[index]] += PENALTY
        cost[assignment[index], label] -= PENALTY
    rows, cols = linear_sum_assignment(cost)
    return dict(zip(rows, cols))


def mahalanobis(z, means, precision):
    diff = z[:, None, :] - means[None]
    return np.einsum("nci,ij,ncj->nc", diff, precision, diff)


def soft(d2):
    weights = np.exp(-(d2 - d2.min(1, keepdims=True)) / 2)
    return weights / weights.sum(1, keepdims=True)


def anchored(z_support, support_labels, z_body, precision, logp_body, init):
    """rec1_borrowed with anchors only for the labeled classes. Returns class predictions for the body strokes."""
    labeled = list(support_labels)
    unlabeled = [c for c in range(4) if c not in labeled]
    means = np.zeros((4, z_body.shape[1]))
    for z, c in zip(z_support, labeled):
        means[c] = z
    if init == "zero" and unlabeled:
        lp = logp_body[:, unlabeled]
        w = np.exp(lp - lp.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        for k, c in enumerate(unlabeled):
            means[c] = w[:, k] @ z_body / max(w[:, k].sum(), 1e-12)
    elif init == "farthest":
        placed = list(labeled)
        for c in unlabeled:
            anchors = means[placed] if placed else z_body.mean(0, keepdims=True)
            means[c] = z_body[mahalanobis(z_body, anchors, precision).min(1).argmax()]
            placed.append(c)
    onehot = np.zeros((len(labeled), 4))
    onehot[np.arange(len(labeled)), labeled] = 1
    for _ in range(2):
        weights = soft(mahalanobis(z_body, means, precision))
        total = onehot.T @ z_support + weights.T @ z_body
        count = onehot.sum(0) + weights.sum(0)
        means = np.where(count[:, None] > 1e-12, total / np.maximum(count, 1e-12)[:, None], means)
    d2 = mahalanobis(z_body, means, precision)
    slots = d2.argmin(1)
    if init != "farthest" or not unlabeled:
        return slots
    weights = soft(d2)
    cost = np.array([[-(weights[:, s] * logp_body[:, c]).sum() for c in unlabeled] for s in unlabeled])
    rows, cols = linear_sum_assignment(cost)
    mapping = {s: s for s in labeled} | {unlabeled[r]: unlabeled[k] for r, k in zip(rows, cols)}
    return np.array([mapping[s] for s in slots])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/w02_dev_20261015/dataset.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--kmeans-seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000, help="wear-cluster resamples for real rows (0: off)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = v21_data.load_w02(args.dataset)
    if set(np.unique(data["wear_group"])) & SEALED_WEARS:
        raise RuntimeError("sealed wear present in the development dataset")
    run = {"started_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset),
           "dataset_sha256": sha256(args.dataset), "script_sha256": sha256(Path(__file__)),
           "recipe_source_sha256": sha256(Path(recipes.__file__)), "class_order": list(v21_data.CLASSES),
           "null_seeds": args.null_seeds, "kmeans_seed": args.kmeans_seed, "bootstrap": args.bootstrap,
           "test_access": "development wears only (leave-one-wear-out); sealed wears W07/W10/W12 refused; exploratory"}
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2))

    groups, wears = data["group"], data["wear_group"]
    covs = recipes.covariances(data["X"])
    prefix_tangent = recipes._recording_tangent_at_prefix(covs, data)
    zero_features = per_recording_tangent(
        recipes.covariances(recipes.cyclic_canonical(data["X"], data["position"])), groups)

    # Label-independent per-recording k-means; per-fold borrowed metric and zero models (true + null labels).
    clusters = {}
    for group in np.unique(groups):
        mask = groups == group
        local = recipes.tangent(covs[mask], covs[mask].mean(0))
        local = (local - local.mean(0)) / (local.std(0) + 1e-6)
        clusters[group] = KMeans(4, n_init=20, random_state=args.kmeans_seed).fit_predict(local)
    label_sets = {None: data["y"]} | {s: v21_data.block_permuted_labels(data, s) for s in args.null_seeds}
    z = np.empty((len(covs), 30))
    precision, logp = {}, {s: np.empty((len(covs), 4)) for s in label_sets}
    rec1_reference = np.full(len(covs), -1)
    for wear in np.unique(wears):
        test, train = wears == wear, wears != wear
        train_data = v21_data.subset(data, train)
        pca, precision[wear] = recipes.borrowed_metric(covs[train], train_data)
        z[test] = pca.transform(prefix_tangent[test])
        for seed, labels in label_sets.items():
            logp[seed][test] = zero_log_proba(zero_features, labels, train)[test]
        rec1_reference[test] = recipes.predict("rec1_borrowed", v21_data.subset(data, test), train_data)
        print(f"fold {wear} ready", flush=True)

    def predict(method, classes, seed):
        rng = np.random.default_rng(seed) if seed is not None else None
        predicted = np.full(len(data["y"]), -1)
        for group in np.unique(groups):
            mask = groups == group
            idx = np.flatnonzero(mask)
            body = np.flatnonzero(mask & ~data["is_calibration"])
            support = support_for(data, group, classes)
            labels = np.array(classes, dtype=int)
            if rng is not None:
                labels = rng.permutation(4)[labels]
            if method.startswith("cluster"):
                assignment = clusters[group]
                local_support = np.searchsorted(idx, support)
                mapping = name_clusters(assignment, logp[seed][idx], local_support, labels,
                                        prior=1.0 if method == "cluster_zero" else 0.0)
                named = np.array([mapping[a] for a in assignment])
                predicted[body] = named[np.searchsorted(idx, body)]
            else:
                wear = wears[group == groups][0]
                predicted[body] = anchored(z[support], labels, z[body], precision[wear], logp[seed][body],
                                           init=method.split("_")[1])
        return predicted

    scored = ~data["is_calibration"]
    results = {"rows": [], "summary": {}}
    oracle = np.full(len(data["y"]), -1)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        assignment, y = clusters[group], data["y"][idx]
        best = max(PERMUTATIONS, key=lambda perm: np.sum(perm[assignment] == y))
        oracle[idx] = best[assignment]
    oracle[data["is_calibration"]] = -1
    results["cluster_oracle"] = report(data, oracle, args.bootstrap)
    print(f"cluster_oracle (ceiling)      F1 {results['cluster_oracle']['macro_f1']:.3f}", flush=True)

    for method in METHODS:
        for budget in range(5):
            f1s, nulls_all = [], []
            for classes in itertools.combinations(range(4), budget):
                predicted = predict(method, classes, None)
                row = {"method": method, "budget": budget, "labeled_classes": [v21_data.CLASSES[c] for c in classes],
                       **report(data, predicted, args.bootstrap)}
                row["null_macro_f1"] = [macro_f1(data["y"][scored], predict(method, classes, s)[scored])
                                        for s in args.null_seeds]
                if budget == 4 and method.startswith("anchored"):
                    row["rec1_borrowed_agreement"] = float(np.mean(predicted[scored] == rec1_reference[scored]))
                results["rows"].append(row)
                f1s.append(row["macro_f1"])
                nulls_all.extend(row["null_macro_f1"])
                print(f"{method:18s} b={budget} {'+'.join(row['labeled_classes']) or '-':18s} "
                      f"F1 {row['macro_f1']:.3f} min recall {min(row['recall'].values()):.2f} "
                      f"pred {list(row['prediction_counts'].values())} naming failures "
                      f"{len(row['naming_failures'])}/{row['n_recordings']} "
                      f"null {np.mean(row['null_macro_f1']) if row['null_macro_f1'] else float('nan'):.3f}"
                      + (f" agree(rec1) {row['rec1_borrowed_agreement']:.3f}" if "rec1_borrowed_agreement" in row
                         else ""), flush=True)
            results["summary"][f"{method}/b{budget}"] = {
                "mean_f1_over_subsets": float(np.mean(f1s)), "min_f1": float(np.min(f1s)),
                "max_f1": float(np.max(f1s)), "n_subsets": len(f1s),
                "mean_null_f1": float(np.mean(nulls_all)) if nulls_all else None}
    for key, s in results["summary"].items():
        print(f"{key:24s} mean F1 {s['mean_f1_over_subsets']:.3f} [{s['min_f1']:.3f}, {s['max_f1']:.3f}] "
              f"null {s['mean_null_f1'] if s['mean_null_f1'] is None else round(s['mean_null_f1'], 3)}")
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
