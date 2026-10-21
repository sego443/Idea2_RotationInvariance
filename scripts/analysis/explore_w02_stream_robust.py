"""Exploratory family G: fixes for the two failure modes of the 4-action adaptive recipe (W02 development wears).

Base recipe (family A primary): K = 4 calibration strokes, log-Euclidean covariance vectors scaled by the training
wears' per-feature std (labels unused), nearest prototype, cumulative update of the predicted class's prototype,
online re-centering. Failure modes found by family F: unbalanced use (re-centering on an EMA of all strokes assumes
balanced use) and a calibration stroke performed as the wrong direction (self-adaptation reinforces it).

Part 1, re-centering variants x usage scenarios (scenarios as family F; miscal here enumerates all six label swaps
of two calibration strokes per wear):
  static, adapt                   no updates / prototype updates without re-centering
  rc{b}                           EMA (rate b) of all past strokes
  rcconf{b}                       as rc, but the reference moves only when the margin (d2nd - d1st) / d2nd >= 0.2
  rcres{b}                        residual tracking: after predicting class c the reference moves by b times the
                                  part of the stroke not explained by c's prototype (x - mu - std * p_c), so the
                                  class mix does not bias the reference
  rcresconf{b}                    rcres, gated by margin >= 0.2
  b in 0.05, 0.1, 0.15, 0.2, 0.3.
Part 2, calibration check: before use, the 4 calibration strokes (relative to their mean, scaled) are matched
one-to-one to the training wears' class prototypes (Hungarian assignment; training labels used). A calibration whose
labels disagree with the assignment is flagged. Reported: false-alarm rate on correct calibrations, detection rate
on each swapped calibration, and the stream accuracy / user actions of "check + redo" (a flagged calibration is
redone correctly with the other prefix occurrence of each class: 4 more actions) for the base recipe.
Part 3, repair by corrections: the base recipe with miscalibration, with the user correcting every error or half of
the errors (family D loop), versus correct calibration.
Matched nulls: calibration labels permuted (seeded) for part 1; part 2 reports the check on permuted labels.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.analysis import w02_stream_lib as lib
from scripts.analysis.explore_w02_stream_feedback import stream as feedback_stream
from scripts.analysis.explore_w02_stream_usage import apply, scenarios

BETAS = (0.05, 0.1, 0.15, 0.2, 0.3)
PIPELINES = ["static", "adapt"] + [f"{kind}{b}" for kind in ("rc", "rcconf", "rcres", "rcresconf") for b in BETAS]


def run(x, ax, ay, std, pipeline):
    kind = pipeline.rstrip("0123456789.")
    beta = float(pipeline[len(kind):]) if kind not in ("static", "adapt") else 0.0
    mu = ax.mean(0)
    protos = np.stack([((ax[ay == c] - mu) / std).mean(0) for c in range(4)])
    counts = np.bincount(ay, minlength=4).astype(float)
    pred = np.empty(len(x), dtype=int)
    for t in range(len(x)):
        v = (x[t] - mu) / std
        d2 = ((protos - v) ** 2).sum(1)
        order = np.argsort(d2)
        c = int(order[0])
        pred[t] = c
        if kind == "static":
            continue
        margin = (d2[order[1]] - d2[c]) / max(d2[order[1]], 1e-12)
        residual = x[t] - mu - std * protos[c]
        counts[c] += 1
        protos[c] += (v - protos[c]) / counts[c]
        if kind == "rc" or (kind == "rcconf" and margin >= 0.2):
            mu = (1 - beta) * mu + beta * x[t]
        elif kind == "rcres" or (kind == "rcresconf" and margin >= 0.2):
            mu = mu + beta * residual
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=lib.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    data = lib.load_dev(args.dataset)
    lib.start_run(args.output_dir, __file__, args.dataset, {"pipelines": PIPELINES, "null_seeds": args.null_seeds})
    streams, minutes = lib.wear_streams(data), lib.recording_minutes(data)
    wears = list(streams)
    y = data["y"]
    rank = np.zeros(len(y), int)
    for s in streams.values():
        for r, g in enumerate(s["recordings"]):
            rank[data["group"] == g] = r
    vectors = lib.log_euclid(lib.covariances(data["X"]))
    centered = vectors.copy()
    for group in np.unique(data["group"]):
        m = data["group"] == group
        centered[m] = vectors[m] - vectors[m].mean(0)
    stds, pops = {}, {}
    for w in wears:
        train = data["wear_group"] != w
        stds[w] = centered[train].std(0) + 1e-6
        pops[w] = np.stack([centered[train & (y == c)].mean(0) for c in range(4)]) / stds[w]
    swaps = list(itertools.combinations(range(4), 2))

    def calibration_variants(spec, anchor_y, w_i, draw):
        """List of (variant name, anchor labels). miscal enumerates all six swaps of two calibration labels."""
        if spec is None or spec[0] != "miscal":
            return [("-", anchor_y)]
        out = []
        for i, j in swaps:
            a = anchor_y.copy()
            a[[i, j]] = a[[j, i]]
            out.append((f"swap{i}{j}", a))
        return out

    rows, saved = [], {}

    def record(fields, pieces, pred, null_preds, actions=None, key=None):
        idx, positions, ranks, mins = lib.stream_context(data, streams, pieces, minutes)
        row = {**fields, "true_counts": np.bincount(y[idx], minlength=4).tolist(),
               **lib.stream_report(data, idx, pred, positions, ranks, mins, actions)}
        if null_preds:
            row["null_accuracy"] = [float(np.mean(y[idx] == p)) for p in null_preds]
            row["null_prediction_counts"] = [np.bincount(p, minlength=4).tolist() for p in null_preds]
        full = np.full(len(y), -1, np.int8)
        full[idx] = pred
        saved[key or "|".join(str(v) for v in fields.values())] = full
        rows.append(row)
        print(" ".join(f"{v}" for v in fields.values()), lib.short(row), flush=True)
        return row

    # Part 1: re-centering variants x usage scenarios.
    for scenario, spec in scenarios():
        for pipeline in PIPELINES:
            for draw in (0, 1):
                n_variants = 6 if spec is not None and spec[0] == "miscal" else 1
                for v_i in range(n_variants):
                    results = {}
                    for seed in [None] + list(args.null_seeds):
                        pieces, preds = [], []
                        for w_i, wear in enumerate(wears):
                            anchors, anchor_y, rest = lib.calibration(streams[wear], y, range(4), draw)
                            rest = apply(rest, y, rank, spec, np.random.default_rng([w_i, draw, 7]))
                            _, anchor_y = calibration_variants(spec, anchor_y, w_i, draw)[v_i]
                            if seed is not None:
                                anchor_y = np.random.default_rng([seed, w_i]).permutation(4)[anchor_y]
                            pieces.append((wear, rest))
                            preds.append(run(vectors[rest], vectors[anchors], anchor_y, stds[wear], pipeline))
                        results[seed] = (pieces, np.concatenate(preds))
                    variant = calibration_variants(spec, np.arange(4), 0, draw)[v_i][0]
                    record({"part": 1, "scenario": scenario, "pipeline": pipeline, "draw": draw, "variant": variant},
                           results[None][0], results[None][1], [results[s][1] for s in args.null_seeds])

    # Part 2: calibration check (Hungarian match to population prototypes) and check + redo.
    def flagged(wear, anchors, labels):
        rel = (vectors[anchors] - vectors[anchors].mean(0)) / stds[wear]
        cost = ((rel[:, None] - pops[wear][None]) ** 2).sum(2)
        rows_, cols = linear_sum_assignment(cost)
        assigned = np.empty(len(anchors), int)
        assigned[rows_] = cols
        return bool(np.any(assigned != labels))

    check = {"correct": [], "swapped": {}, "permuted_null": []}
    for w_i, wear in enumerate(wears):
        for draw in (0, 1):
            anchors, anchor_y, _ = lib.calibration(streams[wear], y, range(4), draw)
            check["correct"].append(flagged(wear, anchors, anchor_y))
            for i, j in swaps:
                a = anchor_y.copy()
                a[[i, j]] = a[[j, i]]
                check["swapped"].setdefault(f"{i}{j}", []).append(flagged(wear, anchors, a))
            for seed in args.null_seeds:
                check["permuted_null"].append(flagged(wear, anchors, np.random.default_rng([seed, w_i]).permutation(4)[anchor_y]))
    check_summary = {"false_alarm_rate": float(np.mean(check["correct"])),
                     "detection_rate_by_swap": {k: float(np.mean(v)) for k, v in check["swapped"].items()},
                     "detection_rate_all_swaps": float(np.mean([f for v in check["swapped"].values() for f in v])),
                     "flag_rate_permuted_null": float(np.mean(check["permuted_null"]))}
    print("calibration check:", check_summary, flush=True)

    for condition in ("correct", "miscal"):
        for draw in (0, 1):
            n_variants = 6 if condition == "miscal" else 1
            for v_i in range(n_variants):
                pieces, preds, acts = [], [], []
                for w_i, wear in enumerate(wears):
                    anchors, anchor_y, rest = lib.calibration(streams[wear], y, range(4), draw)
                    labels = anchor_y.copy()
                    if condition == "miscal":
                        i, j = swaps[v_i]
                        labels[[i, j]] = labels[[j, i]]
                    actions = np.zeros(len(rest))
                    if flagged(wear, anchors, labels):  # redo with the other occurrence, performed correctly
                        anchors, labels, _ = lib.calibration(streams[wear], y, range(4), 1 - draw)
                        rest = rest[~np.isin(rest, anchors)]
                        actions = np.zeros(len(rest))
                        actions[0] = 4
                    pieces.append((wear, rest))
                    preds.append(run(vectors[rest], vectors[anchors], labels, stds[wear], "rc0.1"))
                    acts.append(actions)
                variant = f"swap{swaps[v_i][0]}{swaps[v_i][1]}" if condition == "miscal" else "-"
                row = record({"part": 2, "scenario": condition, "pipeline": "check_redo+rc0.1", "draw": draw,
                              "variant": variant}, pieces, np.concatenate(preds), [], np.concatenate(acts))
                row["user_actions_total_per_wear"] += 4

    # Part 3: repair by corrections (family D loop, re-centering rate 0.1).
    for condition in ("correct", "miscal"):
        for policy in ("none", "correct", "correct_p50"):
            for draw in (0, 1):
                n_variants = 6 if condition == "miscal" else 1
                for v_i in range(n_variants):
                    pieces, preds, acts = [], [], []
                    for w_i, wear in enumerate(wears):
                        anchors, anchor_y, rest = lib.calibration(streams[wear], y, range(4), draw)
                        labels = anchor_y.copy()
                        if condition == "miscal":
                            i, j = swaps[v_i]
                            labels[[i, j]] = labels[[j, i]]
                        rng = np.random.default_rng([w_i, draw, v_i, 13])
                        pred, actions = feedback_stream(vectors[rest], y[rest], vectors[anchors], labels, stds[wear],
                                                        0.1, policy, rng)
                        pieces.append((wear, rest))
                        preds.append(pred)
                        acts.append(actions)
                    variant = f"swap{swaps[v_i][0]}{swaps[v_i][1]}" if condition == "miscal" else "-"
                    row = record({"part": 3, "scenario": condition, "pipeline": f"feedback_{policy}", "draw": draw,
                                  "variant": variant}, pieces, np.concatenate(preds), [], np.concatenate(acts))
                    row["user_actions_total_per_wear"] += 4

    print("\nPart 1: mean accuracy over draws (and swaps); columns:", " ".join(PIPELINES))
    summary = []
    for scenario, _ in scenarios():
        cells = []
        for p in PIPELINES:
            sel = [r for r in rows if r["part"] == 1 and r["scenario"] == scenario and r["pipeline"] == p]
            cells.append(float(np.mean([r["accuracy"] for r in sel])))
            summary.append({"part": 1, "scenario": scenario, "pipeline": p, "mean_accuracy": cells[-1],
                            "min_accuracy": float(np.min([r["accuracy"] for r in sel])),
                            "mean_null_accuracy": float(np.mean([r["null_accuracy"] for r in sel]))})
        print(f"{scenario:14s} " + " ".join(f"{c:.3f}" for c in cells))
    print("\nParts 2-3: mean accuracy | mean user actions per wear")
    for part in (2, 3):
        for key in dict.fromkeys((r["scenario"], r["pipeline"]) for r in rows if r["part"] == part):
            sel = [r for r in rows if r["part"] == part and (r["scenario"], r["pipeline"]) == key]
            entry = {"part": part, "scenario": key[0], "pipeline": key[1],
                     "mean_accuracy": float(np.mean([r["accuracy"] for r in sel])),
                     "min_accuracy": float(np.min([r["accuracy"] for r in sel])),
                     "mean_actions_per_wear": float(np.mean([r["user_actions_total_per_wear"] for r in sel]))}
            summary.append(entry)
            print(f"part {part} {key[0]:8s} {key[1]:22s} acc {entry['mean_accuracy']:.3f} "
                  f"(min {entry['min_accuracy']:.3f}) actions {entry['mean_actions_per_wear']:.1f}")
    lib.finish_run(args.output_dir, rows, saved, {"summary": summary, "calibration_check": check_summary,
                                                  "calibration_check_raw": check})


if __name__ == "__main__":
    main()
