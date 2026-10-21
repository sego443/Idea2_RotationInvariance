"""Exploratory family E: what drifts within a wear? (W02 development wears; diagnostic, uses all labels.)

Within each wear, recording 1 is the reference and recordings 2-6 follow over about 10-12 minutes (start times from
the accepted acquisition plans). This script describes the drift and tests which correction removes it. It is a
diagnostic: labels of every stroke are used, so none of it is a deployable startup result.

1. Timing: minutes from the wear's first recording to each recording.
2. Channel gain: RMS of each band-passed channel per recording, relative to recording 1 (all strokes).
3. Feature drift (log-Euclidean covariance vectors scaled by the training-wear std): for recording r,
   common  = || mean_r - mean_1 ||                         (shift shared by all classes)
   class   = mean over classes of || (m_rc - mean_r) - (m_1c - mean_1) ||   (class-specific change)
   separation = mean pairwise distance of recording-1 class means after centering
   noise floor = mean over classes of the distance between class means from the first and the second half of
   that class's strokes in the same recording (sampling noise of a 5-stroke class mean, roughly)
4. Static transfer: class means from all 20 strokes of recording 1, applied to each later recording with
     plain        no correction
     recenter     subtract each recording's own mean (batch, unsupervised) from features and prototypes
     corr         correlation-normalized covariances (per-channel gain removed), no re-centering
     corr+recenter
   Accuracy per recording index; matched null with recording-1 labels permuted (seeds).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from scripts.analysis import w02_stream_lib as lib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=lib.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    data = lib.load_dev(args.dataset)
    lib.start_run(args.output_dir, __file__, args.dataset, {"null_seeds": args.null_seeds})
    streams, minutes = lib.wear_streams(data), lib.recording_minutes(data)
    y, groups = data["y"], data["group"]
    x = lib.bandpass(data["X"])
    rms = np.sqrt((x ** 2).mean(1))  # (n, 4)
    vectors = {"plain": lib.log_euclid(lib.covariances(data["X"])),
               "corr": lib.log_euclid(lib.covariances(data["X"], norm="corr"))}
    stds = {}
    for name, v in vectors.items():
        centered = v.copy()
        for g in np.unique(groups):
            m = groups == g
            centered[m] = v[m] - v[m].mean(0)
        stds[name] = {w: centered[data["wear_group"] != w].std(0) + 1e-6 for w in streams}

    out = {"timing": {}, "gain": {}, "feature_drift": {}, "transfer": {}}
    for wear, s in streams.items():
        recs = s["recordings"]
        out["timing"][wear] = [minutes.get(g) for g in recs]
        base_rms = rms[groups == recs[0]].mean(0)
        out["gain"][wear] = [(rms[groups == g].mean(0) / base_rms).round(4).tolist() for g in recs]
        v = vectors["plain"] / stds["plain"][wear]
        m1 = v[groups == recs[0]].mean(0)
        c1 = {c: v[(groups == recs[0]) & (y == c)].mean(0) - m1 for c in range(4)}
        sep = np.mean([np.linalg.norm(c1[a] - c1[b]) for a, b in itertools.combinations(range(4), 2)])
        drift = []
        for g in recs:
            mr = v[groups == g].mean(0)
            cls = np.mean([np.linalg.norm((v[(groups == g) & (y == c)].mean(0) - mr) - c1[c]) for c in range(4)])
            halves = []
            for c in range(4):
                rows = np.flatnonzero((groups == g) & (y == c))
                rows = rows[np.argsort(data["repetition"][rows])]
                half = len(rows) // 2
                halves.append(np.linalg.norm(v[rows[:half]].mean(0) - v[rows[half:]].mean(0)))
            drift.append({"common": float(np.linalg.norm(mr - m1)), "class": float(cls),
                          "noise_floor": float(np.mean(halves)), "separation_rec1": float(sep)})
        out["feature_drift"][wear] = drift
        print(f"{wear} minutes {[None if t is None else round(t, 1) for t in out['timing'][wear]]}")
        print(f"   gain (channels 0-3, relative to rec1): " +
              " | ".join(" ".join(f"{q:.2f}" for q in r) for r in out["gain"][wear]))
        print(f"   drift common {[round(d['common'], 2) for d in drift]} class {[round(d['class'], 2) for d in drift]}"
              f" noise floor {[round(d['noise_floor'], 2) for d in drift]} separation {sep:.2f}")

    for variant in ("plain", "recenter", "corr", "corr+recenter"):
        name = "corr" if variant.startswith("corr") else "plain"
        recenter = variant.endswith("recenter")
        results = {}
        for seed in [None] + list(args.null_seeds):
            per_rec = {r: [] for r in range(6)}
            for w_i, (wear, s) in enumerate(streams.items()):
                v = vectors[name] / stds[name][wear]
                recs = s["recordings"]
                ref = groups == recs[0]
                labels = y.copy()
                if seed is not None:
                    labels[ref] = np.random.default_rng([seed, w_i]).permutation(4)[y[ref]]
                offset = v[ref].mean(0) if recenter else 0.0
                protos = np.stack([v[ref & (labels == c)].mean(0) - offset for c in range(4)])
                for r, g in enumerate(recs[1:], start=1):
                    m = groups == g
                    f = v[m] - (v[m].mean(0) if recenter else 0.0)
                    pred = ((f[:, None] - protos[None]) ** 2).sum(2).argmin(1)
                    per_rec[r].append(np.mean(pred == y[m]))
            results[seed] = {r: float(np.mean(a)) for r, a in per_rec.items() if a}
        out["transfer"][variant] = {"accuracy_by_recording": results[None],
                                    "null_mean_accuracy": float(np.mean([np.mean(list(results[s].values()))
                                                                         for s in args.null_seeds]))}
        print(f"transfer {variant:14s} recordings 2-6 {[round(a, 3) for a in results[None].values()]} "
              f"null {out['transfer'][variant]['null_mean_accuracy']:.3f}")
    lib.finish_run(args.output_dir, [], None, out)


if __name__ == "__main__":
    main()
