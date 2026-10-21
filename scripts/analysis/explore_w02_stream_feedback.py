"""Exploratory family D: user corrections as the source of labels (W02 development wears, causal streams).

Question: if the user corrects wrong actions (or answers an occasional query) instead of calibrating up front, how
many total actions does it take to reach usable accuracy? Every label the user gives counts as one action, whether
it is a startup stroke, a correction or a query answer.

Pipeline: log-Euclidean covariance vectors, online re-centering (EMA of past stream features, rate --beta; starts
at the calibration mean, or at the first stroke when K = 0), per-feature scaling by the training wears' std (labels
unused), nearest prototype among the classes that already have one. A class gets its first prototype only from a
user label. After each prediction the predicted class's prototype is updated (cumulative mean) unless the user
supplies a label, in which case the labeled class is updated instead. With no active class the system cannot act
(counted as an error).
Startup: K = 0, 1 (each class), 2 (all pairs), 4; draws 0 and 1.
Policies:
  none         no corrections (K = 4 reference)
  correct      the user corrects every error
  correct_p50  the user corrects each error with probability 0.5 (seeded); uncorrected errors self-update
  query{m}     instead of acting, the system asks for the label when its margin (d2nd - d1st) / d2nd < m, or when a
               stroke is farther than twice the median prototype distance while classes are missing (an abstention:
               scored as not correct); errors on the remaining strokes are corrected
Accuracy counts a stroke as correct only when the system acted and was right (abstentions and strokes with no
active class count as not correct). Metrics add user actions per wear (startup + corrections + queries) overall and
by stream position, wrong actions and abstentions.
Matched null: the user's labels (startup, corrections, query answers) and the user's judgement of what is an error
both follow a seeded random label per stroke instead of the true label.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from scripts.analysis import w02_stream_lib as lib
from src import v21_data

POLICIES = ("correct", "correct_p50", "query0.1", "query0.2", "query0.3")


def stream(x, y_teacher, anchors_x, anchor_labels, std, beta, policy, rng):
    mu = anchors_x.mean(0) if len(anchors_x) else x[0].copy()
    protos, counts = np.zeros((4, x.shape[1])), np.zeros(4)
    for v, c in zip((anchors_x - mu) / std, anchor_labels):
        counts[c] += 1
        protos[c] += (v - protos[c]) / counts[c]
    pred, actions, abstain = np.full(len(x), -1), np.zeros(len(x)), np.zeros(len(x), bool)
    margin_q = float(policy[5:]) if policy.startswith("query") else None
    for t in range(len(x)):
        v = (x[t] - mu) / std
        active = counts > 0
        label = None
        if active.any():
            d2 = np.where(active, ((protos - v) ** 2).sum(1), np.inf)
            order = np.argsort(d2)
            pred[t] = order[0]
            if margin_q is not None:
                second = d2[order[1]] if np.isfinite(d2[order[1]]) else np.inf
                margin = 1.0 if not np.isfinite(second) else (second - d2[order[0]]) / max(second, 1e-12)
                spread = (np.median([np.sqrt(((protos[a] - protos[b]) ** 2).sum())
                                     for a, b in itertools.combinations(np.flatnonzero(active), 2)])
                          if active.sum() > 1 else np.sqrt(d2[order[0]]) + 1.0)
                far = (not active.all()) and np.sqrt(d2[order[0]]) > 2 * spread
                if margin < margin_q or far:
                    label, abstain[t] = y_teacher[t], True
        if label is None and pred[t] != y_teacher[t] and policy != "none":
            if policy != "correct_p50" or rng.random() < 0.5:
                label = y_teacher[t]
        if label is not None:
            actions[t] = 1
            c = label
        elif pred[t] >= 0:
            c = pred[t]
        else:
            c = None
        if c is not None:
            counts[c] += 1
            protos[c] += (v - protos[c]) / counts[c]
        mu = (1 - beta) * mu + beta * x[t]
    pred[abstain] = -1
    return pred, actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=lib.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    data = lib.load_dev(args.dataset)
    lib.start_run(args.output_dir, __file__, args.dataset,
                  {"beta": args.beta, "policies": POLICIES, "null_seeds": args.null_seeds})
    streams, minutes = lib.wear_streams(data), lib.recording_minutes(data)
    wears = list(streams)
    vectors = lib.log_euclid(lib.covariances(data["X"]))
    centered = vectors.copy()
    for group in np.unique(data["group"]):
        m = data["group"] == group
        centered[m] = vectors[m] - vectors[m].mean(0)
    stds = {w: centered[data["wear_group"] != w].std(0) + 1e-6 for w in wears}

    subsets = [()] + [(c,) for c in range(4)] + list(itertools.combinations(range(4), 2)) + [(0, 1, 2, 3)]
    rows, saved = [], {}
    for classes in subsets:
        k = len(classes)
        policies = (("none",) if k == 4 else ()) + POLICIES
        for policy in policies:
            for draw in ([0, 1] if k else ["-"]):
                results = {}
                for seed in [None] + list(args.null_seeds):
                    pieces, preds, acts = [], [], []
                    for w_i, wear in enumerate(wears):
                        if k:
                            anchors, anchor_y, rest = lib.calibration(streams[wear], data["y"], classes, draw)
                        else:
                            anchors, anchor_y, rest = np.array([], int), np.array([], int), streams[wear]["order"]
                        rng = np.random.default_rng([w_i, 0 if seed is None else seed + 1, k, draw if k else 9])
                        teacher = data["y"][rest]
                        if seed is not None:
                            noise = np.random.default_rng([seed, w_i])
                            teacher = noise.integers(0, 4, len(rest))
                            anchor_y = noise.integers(0, 4, len(anchor_y))
                        pred, actions = stream(vectors[rest], teacher, vectors[anchors], anchor_y,
                                               stds[wear], args.beta, policy, rng)
                        pieces.append((wear, rest))
                        preds.append(pred)
                        acts.append(actions)
                    results[seed] = (pieces, np.concatenate(preds), np.concatenate(acts))
                pieces, pred, actions = results[None]
                idx, positions, ranks, mins = lib.stream_context(data, streams, pieces, minutes)
                scored_pred = np.where(pred >= 0, pred, (data["y"][idx] + 1) % 4)  # no action = error
                subset = "+".join(v21_data.CLASSES[c] for c in classes) or "-"
                row = {"policy": policy, "subset": subset, "startup_actions": k, "draw": str(draw),
                       "no_action_strokes": int(np.sum(pred < 0)),
                       "wrong_actions": int(np.sum((pred >= 0) & (pred != data["y"][idx]))),
                       **lib.stream_report(data, idx, scored_pred, positions, ranks, mins, actions)}
                row["user_actions_total_per_wear"] += k
                row["null_accuracy"] = [float(np.mean(data["y"][idx] == results[s][1])) for s in args.null_seeds]
                row["null_actions_per_wear"] = [float(results[s][2].sum() / len(wears) + k) for s in args.null_seeds]
                full = np.full(len(data["y"]), -1, np.int8)
                full[idx] = pred
                saved[f"{policy}|{subset}|d{draw}"] = full
                rows.append(row)
                print(f"{policy:11s} K={k} {subset:10s} d{draw!s:2s} {lib.short(row)} "
                      f"actions 1-20 {sum(list(row['user_actions_by_position'].values())[:3]):.1f}", flush=True)

    print("\nmean over subsets and draws: accuracy | actions per wear (total, in first 20 strokes incl. startup)")
    summary = []
    for policy in ("none",) + POLICIES:
        for k in (0, 1, 2, 4):
            sel = [r for r in rows if r["policy"] == policy and r["startup_actions"] == k]
            if not sel:
                continue
            entry = {"policy": policy, "startup_actions": k, "mean_accuracy": float(np.mean([r["accuracy"] for r in sel])),
                     "mean_actions_per_wear": float(np.mean([r["user_actions_total_per_wear"] for r in sel])),
                     "mean_actions_first20": float(np.mean([k + sum(list(r["user_actions_by_position"].values())[:3])
                                                            for r in sel])),
                     "mean_accuracy_after_20": float(np.mean([np.mean([r["accuracy_by_position"]["21-40"],
                                                                       r["accuracy_by_position"]["41+"]]) for r in sel])),
                     "mean_null_accuracy": float(np.mean([r["null_accuracy"] for r in sel]))}
            summary.append(entry)
            print(f"{policy:11s} K={k} acc {entry['mean_accuracy']:.3f} after20 {entry['mean_accuracy_after_20']:.3f} "
                  f"actions {entry['mean_actions_per_wear']:.1f} first20 {entry['mean_actions_first20']:.1f} "
                  f"null {entry['mean_null_accuracy']:.3f}")
    lib.finish_run(args.output_dir, rows, saved, {"summary": summary})


if __name__ == "__main__":
    main()
