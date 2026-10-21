"""Exploratory family F: robustness of the adaptive pipelines to unbalanced use (W02 development wears, causal).

The recorded streams are prompt-balanced (about 30 strokes per class per wear). Online re-centering subtracts a
running mean of all strokes, which assumes balanced use. Here the stream after calibration is subsampled (time order
kept) to imitate realistic use, and the pipelines are compared.

Usage scenarios (per wear, seeded):
  balanced          the full stream
  sparse50          a random half of the stream
  skew_{C}_{p}      class C keeps all its strokes; the others are subsampled so that C makes up share p
                    (p = 0.4, 0.55, 0.7)
  absent_{C}        class C is removed from recordings 2-4 (temporarily unused)
  two_class         only UP and DOWN strokes after recording 1
  miscal            full stream, but two calibration strokes have their labels swapped (a calibration gesture
                    performed as the wrong direction; the pair is drawn per wear)
Pipelines (K = 4 calibration, draws 0 and 1; log-Euclidean vectors scaled by the training-wear std, labels unused):
  static            nearest prototype, no updates
  adapt             cumulative prototype updates on every prediction, no re-centering
  rc{b}             adapt + re-centering on an EMA (rate b) of all past strokes (b = 0.02, 0.05, 0.1)
  rcbal{b}          adapt + class-balanced re-centering: an EMA (rate 4b) of the raw features of each predicted
                    class, re-centering on their average, so frequent classes do not dominate the reference
Matched null: calibration labels permuted (seeded).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.analysis import w02_stream_lib as lib

PIPELINES = ["static", "adapt"] + [f"rc{b}" for b in (0.02, 0.05, 0.1)] + [f"rcbal{b}" for b in (0.02, 0.05, 0.1)]


def scenarios():
    yield "balanced", None
    yield "sparse50", ("sparse", 0.5)
    for c in range(4):
        for p in (0.4, 0.55, 0.7):
            yield f"skew_{'UDLR'[c]}_{p}", ("skew", c, p)
        yield f"absent_{'UDLR'[c]}", ("absent", c)
    yield "two_class", ("two",)
    yield "miscal", ("miscal",)


def apply(rest, y, rank, spec, rng):
    if spec is None or spec[0] == "miscal":
        return rest
    if spec[0] == "sparse":
        return np.sort(rng.choice(rest, size=len(rest) // 2, replace=False))
    if spec[0] == "skew":
        _, c, p = spec
        dominant = rest[y[rest] == c]
        others = rest[y[rest] != c]
        n_keep = min(len(others), int(round(len(dominant) * (1 - p) / p)))
        keep = np.concatenate([dominant, rng.choice(others, size=n_keep, replace=False)])
        return rest[np.isin(rest, keep)]
    if spec[0] == "absent":
        _, c = spec
        return rest[~((y[rest] == c) & np.isin(rank[rest], (1, 2, 3)))]
    return rest[(rank[rest] == 0) | np.isin(y[rest], (0, 1))]


def stream(x, ax, ay, std, pipeline):
    beta = float(pipeline.lstrip("rcbal")) if pipeline.startswith("rc") else 0.0
    balanced = pipeline.startswith("rcbal")
    class_raw = np.stack([ax[ay == c].mean(0) for c in range(4)])
    mu = class_raw.mean(0)
    protos = np.stack([((ax[ay == c] - mu) / std).mean(0) for c in range(4)])
    counts = np.bincount(ay, minlength=4).astype(float)
    pred = np.empty(len(x), dtype=int)
    for t in range(len(x)):
        v = (x[t] - mu) / std
        c = int(np.argmin(((protos - v) ** 2).sum(1)))
        pred[t] = c
        if pipeline == "static":
            continue
        counts[c] += 1
        protos[c] += (v - protos[c]) / counts[c]
        if balanced:
            rate = min(1.0, 4 * beta)
            class_raw[c] = (1 - rate) * class_raw[c] + rate * x[t]
            mu = class_raw.mean(0)
        elif beta:
            mu = (1 - beta) * mu + beta * x[t]
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
    stds = {w: centered[data["wear_group"] != w].std(0) + 1e-6 for w in wears}

    rows, saved = [], {}
    for scenario, spec in scenarios():
        for pipeline in PIPELINES:
            for draw in (0, 1):
                results = {}
                for seed in [None] + list(args.null_seeds):
                    pieces, preds = [], []
                    for w_i, wear in enumerate(wears):
                        anchors, anchor_y, rest = lib.calibration(streams[wear], y, range(4), draw)
                        rest = apply(rest, y, rank, spec, np.random.default_rng([w_i, draw, 7]))
                        if spec is not None and spec[0] == "miscal":
                            i, j = np.random.default_rng([w_i, draw, 11]).choice(len(anchor_y), 2, replace=False)
                            anchor_y = anchor_y.copy()
                            anchor_y[[i, j]] = anchor_y[[j, i]]
                        if seed is not None:
                            anchor_y = np.random.default_rng([seed, w_i]).permutation(4)[anchor_y]
                        pieces.append((wear, rest))
                        preds.append(stream(vectors[rest], vectors[anchors], anchor_y, stds[wear], pipeline))
                    results[seed] = (pieces, np.concatenate(preds))
                pieces, pred = results[None]
                idx, positions, ranks, mins = lib.stream_context(data, streams, pieces, minutes)
                row = {"scenario": scenario, "pipeline": pipeline, "startup_actions": 4, "draw": str(draw),
                       "true_counts": np.bincount(y[idx], minlength=4).tolist(),
                       **lib.stream_report(data, idx, pred, positions, ranks, mins)}
                row["null_accuracy"] = [float(np.mean(y[idx] == results[s][1])) for s in args.null_seeds]
                row["null_prediction_counts"] = [np.bincount(results[s][1], minlength=4).tolist()
                                                 for s in args.null_seeds]
                full = np.full(len(y), -1, np.int8)
                full[idx] = pred
                saved[f"{scenario}|{pipeline}|d{draw}"] = full
                rows.append(row)
                print(f"{scenario:14s} {pipeline:10s} d{draw} {lib.short(row)}", flush=True)

    print("\nmean accuracy over draws; columns:", " ".join(PIPELINES))
    for scenario, _ in scenarios():
        cells = [np.mean([r["accuracy"] for r in rows if r["scenario"] == scenario and r["pipeline"] == p])
                 for p in PIPELINES]
        print(f"{scenario:14s} " + " ".join(f"{c:.3f}" for c in cells))
    lib.finish_run(args.output_dir, rows, saved)


if __name__ == "__main__":
    main()
