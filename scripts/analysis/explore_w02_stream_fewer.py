"""Exploratory family C: fewer than 4 startup actions under the causal protocol (W02 development wears).

Pipeline (from family A): log-Euclidean covariance vectors, unsupervised online re-centering (exponential moving
average of past stream features, rate --beta), per-feature scaling by the training wears' std (pop_z), nearest
prototype with cumulative updates of the predicted class's prototype. Calibrated classes start at their calibration
strokes; the remaining classes need a starting prototype from somewhere:
  pop       the training wears' class mean, in the same re-centered space (features minus each training recording's
            mean), relative to the current stream reference
  synth_T   the calibration stroke of a source class transformed by T (applied to the raw window before the
            covariance), with T and the source chosen on the training wears only (leave-one-wear-out): for each
            missing class, the (source, T) whose transformed training-wear class means lie nearest to the target
            class means. T in: identity, time reversal, negation, cyclic ring shift by 1/2/3, ring reflections,
            and time reversal combined with each ring operation.
  lazy      no starting prototype: a missing class cannot be predicted until it is created; a stroke farther than
            --new-class-factor times the median calibration distance from every prototype creates the missing class
            named by the population prototype nearest to it (one-to-one)
The stream reference starts at the wear centre implied by the calibration strokes: mean over calibration strokes of
(stroke - std * population prototype of its class); with K = 0 it starts at the first stream stroke. Filled
prototypes start with pseudo-count --seed-count; calibration strokes count 1. Every prediction updates the
predicted class's prototype (cumulative mean).
Calibrated subsets: all 1-, 2- and 3-class subsets (both draws), K = 0 (pop only) and K = 4 (no fill).
Also writes a transform diagnostic: for each wear, which class each transformed class mean lands nearest to.
Matched null: calibration labels permuted and training labels block-permuted (seeded).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from scripts.analysis import w02_stream_lib as lib
from src import v21_data

TRANSFORMS = ["id", "rev", "neg", "ring1", "ring2", "ring3", "refl0", "refl1",
              "rev+ring1", "rev+ring2", "rev+ring3", "rev+refl0", "rev+refl1"]


def transform(windows, name):
    x = windows
    for part in name.split("+"):
        if part == "rev":
            x = x[:, ::-1]
        elif part == "neg":
            x = -x
        elif part.startswith("ring"):
            x = np.roll(x, int(part[4:]), axis=2)
        elif part == "refl0":  # sensor i -> -i mod 4
            x = x[:, :, [0, 3, 2, 1]]
        elif part == "refl1":  # sensor i -> 1 - i mod 4
            x = x[:, :, [1, 0, 3, 2]]
    return np.ascontiguousarray(x)


def stream(x, protos, active, std, beta, mu0, counts, lazy=None):
    """Causal nearest prototype in re-centered pop_z space. protos are relative to the stream reference."""
    mu = mu0.copy()
    pred = np.empty(len(x), dtype=int)
    protos, active, counts = protos.copy(), active.copy(), counts.copy()
    for t in range(len(x)):
        v = (x[t] - mu) / std
        d2 = np.where(active, ((protos - v) ** 2).sum(1), np.inf)
        c = int(np.argmin(d2))
        if lazy is not None and not active.all():
            threshold, pop_protos = lazy
            if np.sqrt(d2[c]) > threshold:
                missing = np.flatnonzero(~active)
                c = int(missing[np.argmin(((pop_protos[missing] - v) ** 2).sum(1))])
                active[c], protos[c], counts[c] = True, v, 0.0
        pred[t] = c
        counts[c] += 1
        protos[c] += (v - protos[c]) / counts[c]
        mu = (1 - beta) * mu + beta * x[t]
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=lib.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--new-class-factor", type=float, default=1.0)
    parser.add_argument("--seed-count", type=float, default=1.0)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    data = lib.load_dev(args.dataset)
    lib.start_run(args.output_dir, __file__, args.dataset,
                  {"beta": args.beta, "new_class_factor": args.new_class_factor, "seed_count": args.seed_count, "transforms": TRANSFORMS,
                   "null_seeds": args.null_seeds})
    streams, minutes = lib.wear_streams(data), lib.recording_minutes(data)
    wears = list(streams)
    groups = data["group"]
    feats = {name: lib.log_euclid(lib.covariances(transform(data["X"], name))) for name in TRANSFORMS}
    base = feats["id"]
    centered = {}
    for name, f in feats.items():  # minus each recording's mean of the untransformed features
        c = np.empty_like(f)
        for g in np.unique(groups):
            m = groups == g
            c[m] = f[m] - base[m].mean(0)
        centered[name] = c
    print("features ready", flush=True)
    label_sets = {None: data["y"]} | {s: v21_data.block_permuted_labels(data, s) for s in args.null_seeds}

    # Transform diagnostic: per wear, nearest class mean to each transformed class mean (true labels, all strokes).
    diagnostic = {}
    for wear in wears:
        m = data["wear_group"] == wear
        means = np.stack([centered["id"][m & (data["y"] == c)].mean(0) for c in range(4)])
        for name in TRANSFORMS:
            tm = np.stack([centered[name][m & (data["y"] == c)].mean(0) for c in range(4)])
            nearest = ((tm[:, None] - means[None]) ** 2).sum(2).argmin(1)
            diagnostic.setdefault(name, {})[wear] = [v21_data.CLASSES[i] for i in nearest]
    for name in TRANSFORMS:
        maps = diagnostic[name]
        print(f"transform {name:10s}", " ".join(f"{w[-3:]}:{''.join(c[0] for c in v)}" for w, v in maps.items()))

    def fold_stats(wear, labels):
        train = data["wear_group"] != wear
        std = centered["id"][train].std(0) + 1e-6
        pop = np.stack([centered["id"][train & (labels == c)].mean(0) for c in range(4)]) / std
        choice = {}
        for target, source in itertools.permutations(range(4), 2):
            best = None
            for name in TRANSFORMS:
                err = 0.0
                for w in np.unique(data["wear_group"][train]):
                    m = data["wear_group"] == w
                    src = centered[name][m & (labels == source)].mean(0)
                    tgt = centered["id"][m & (labels == target)].mean(0)
                    others = [centered["id"][m & (labels == c)].mean(0) for c in range(4)]
                    err += float(np.argmin([((src - o) ** 2).sum() for o in others]) != target)
                err += 1e-3 * TRANSFORMS.index(name)
                if best is None or err < best[0]:
                    best = (err, name)
            choice[target, source] = best
        return std, pop, choice

    stats = {(w, s): fold_stats(w, labels) for w in wears for s, labels in label_sets.items()}
    print("fold statistics ready", flush=True)
    chosen = {w: {f"{v21_data.CLASSES[t]}<-{v21_data.CLASSES[s]}": stats[w, None][2][t, s][1]
                  for t, s in itertools.permutations(range(4), 2)} for w in wears}

    subsets = [()] + [c for b in (1, 2, 3, 4) for c in itertools.combinations(range(4), b)]
    rows, saved = [], {}
    for classes in subsets:
        k = len(classes)
        fills = ["pop"] if k == 0 else (["none"] if k == 4 else ["pop", "synth", "lazy"])
        draws = ["-"] if k == 0 else [0, 1]
        for fill in fills:
            for draw in draws:
                results = {}
                for seed, labels in label_sets.items():
                    pieces, preds = [], []
                    for w_i, wear in enumerate(wears):
                        std, pop, choice = stats[wear, seed]
                        if k:
                            anchors, anchor_y, rest = lib.calibration(streams[wear], labels, classes, draw)
                            if seed is not None:
                                anchor_y = np.random.default_rng([seed, w_i]).permutation(4)[anchor_y]
                        else:
                            anchors, anchor_y, rest = np.array([], int), np.array([], int), streams[wear]["order"]
                        x = base[rest]
                        if k:
                            mu0 = np.mean([base[a] - std * pop[c] for a, c in zip(anchors, anchor_y)], 0)
                        else:
                            mu0 = base[rest[0]].copy()
                        protos, active = pop.copy(), np.ones(4, bool)
                        counts = np.full(4, args.seed_count)
                        lazy = None
                        if k:
                            anchor_v = (base[anchors] - mu0) / std
                            for i, c in enumerate(anchor_y):
                                protos[c], counts[c] = anchor_v[i], 1.0
                            missing = [c for c in range(4) if c not in set(anchor_y.tolist())]
                            if fill == "synth":
                                for target in missing:
                                    options = [(choice[target, int(c)][0], i, int(c))
                                               for i, c in enumerate(anchor_y)]
                                    _, i, source = min(options)
                                    name = choice[target, source][1]
                                    protos[target] = (feats[name][anchors[i]] - mu0) / std
                                    counts[target] = args.seed_count
                            if fill == "lazy":
                                active = np.zeros(4, bool)
                                active[list(set(anchor_y.tolist()))] = True
                                spread = (np.median([np.sqrt(((a - b) ** 2).sum()) for a, b in
                                                     itertools.combinations(anchor_v, 2)]) if k > 1
                                          else np.sqrt((pop ** 2).sum(1)).mean())
                                lazy = (args.new_class_factor * spread, pop)
                        pieces.append((wear, rest))
                        preds.append(stream(x, protos, active, std, args.beta, mu0, counts, lazy))
                    results[seed] = (pieces, np.concatenate(preds))
                pieces, pred = results[None]
                idx, positions, ranks, mins = lib.stream_context(data, streams, pieces, minutes)
                subset = "+".join(v21_data.CLASSES[c] for c in classes) or "-"
                row = {"fill": fill, "subset": subset, "startup_actions": k, "draw": str(draw),
                       **lib.stream_report(data, idx, pred, positions, ranks, mins)}
                row["null_accuracy"] = [float(np.mean(data["y"][idx] == results[s][1])) for s in args.null_seeds]
                row["null_prediction_counts"] = [np.bincount(results[s][1], minlength=4).tolist()
                                                 for s in args.null_seeds]
                full = np.full(len(data["y"]), -1, np.int8)
                full[idx] = pred
                saved[f"{fill}|{subset}|d{draw}"] = full
                rows.append(row)
                print(f"{fill:9s} K={k} {subset:16s} d{draw!s:2s} {lib.short(row)}", flush=True)

    print("\nmean accuracy over subsets and draws by fill and K")
    summary = []
    for fill in ("pop", "synth", "lazy", "none"):
        for k in range(5):
            sel = [r for r in rows if r["fill"] == fill and r["startup_actions"] == k]
            if sel:
                entry = {"fill": fill, "startup_actions": k, "mean_accuracy": float(np.mean([r["accuracy"] for r in sel])),
                         "min_accuracy": float(np.min([r["accuracy"] for r in sel])),
                         "mean_first10": float(np.mean([np.mean([r["accuracy_by_position"]["1-5"],
                                                                 r["accuracy_by_position"]["6-10"]]) for r in sel])),
                         "mean_null": float(np.mean([r["null_accuracy"] for r in sel]))}
                summary.append(entry)
                print(f"{fill:9s} K={k} mean {entry['mean_accuracy']:.3f} min {entry['min_accuracy']:.3f} "
                      f"first10 {entry['mean_first10']:.3f} null {entry['mean_null']:.3f}")
    lib.finish_run(args.output_dir, rows, saved, {"summary": summary, "transform_diagnostic": diagnostic,
                                                  "chosen_transforms": chosen})


if __name__ == "__main__":
    main()
