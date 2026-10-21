"""Exploratory family B: feature variants under the causal startup protocol (W02 development wears).

Question: does a different representation make a 4-stroke calibration last longer within a wear (gain or contact
drift), or let other wears' models transfer (K = 0)?

Features:
  cov_*   lagged covariance variants (band, lags, normalization: none | trace = unit trace | corr = unit diagonal,
          removing per-channel gain), tangent space at the calibration mean (log-Euclidean for K = 0)
  wave_*  band-passed waveform downsampled by `step`, normalized by stroke RMS (global), per channel, or not
  logpow  log band power per channel (bands 0.5-2, 2-4, 4-8, 8-12, 12-20 Hz), 20 values
Classifiers:
  nc        nearest centroid on the calibration strokes, z-scored with their statistics (K = 4 draws 0/1; K = 8)
  nc_adapt  nc with cumulative prototype updates when margin >= 0.1 (explore_w02_stream_adapt rule cum_g0.1)
  rc_adapt  re-centered pipeline from explore_w02_stream_adapt: vectors (log-Euclidean for covariances) minus an
            exponential moving average of past stream features (rate 0.1, started at the calibration mean), scaled
            by the training wears' per-feature std (features minus recording means; labels unused), cumulative
            prototype updates on every prediction (pop_z, cum_g0.0, rc0.1)
  xcorr     (wave_* only) maximum normalized cross-correlation with the calibration templates over +/- 100 ms shifts
  zero      (K = 0) logistic regression (C = 0.1) on the training wears; covariance features are log-Euclidean
            minus the recording mean in training and minus the running mean of past stream strokes in the test wear
            (first stroke: training mean); other features are standardized with training statistics only
Matched nulls: calibration labels permuted (calibrated classifiers); block-permuted training labels (zero).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.linear_model import LogisticRegression

from scripts.analysis import w02_stream_lib as lib
from scripts.analysis.explore_w02_stream_adapt import run_stream
from src import v21_data

COV = {  # name: (band, lags, norm)
    "cov_base": ((0.5, 12.0), (0, 5, 10, 15, 20, 25, 30), "none"),
    "cov_trace": ((0.5, 12.0), (0, 5, 10, 15, 20, 25, 30), "trace"),
    "cov_corr": ((0.5, 12.0), (0, 5, 10, 15, 20, 25, 30), "corr"),
    "cov_band0.5-6": ((0.5, 6.0), (0, 5, 10, 15, 20, 25, 30), "none"),
    "cov_band1-20": ((1.0, 20.0), (0, 5, 10, 15, 20, 25, 30), "none"),
    "cov_band0.1-12": ((0.1, 12.0), (0, 5, 10, 15, 20, 25, 30), "none"),
    "cov_lag0": ((0.5, 12.0), (0,), "none"),
    "cov_lags0-20": ((0.5, 12.0), (0, 10, 20), "none"),
    "cov_lags0-50": ((0.5, 12.0), (0, 10, 20, 30, 40, 50), "none"),
    "cov_corr_lags0-20": ((0.5, 12.0), (0, 10, 20), "corr"),
}
WAVE = {  # name: (band, step, norm)
    "wave_global": ((0.5, 12.0), 10, "global"),
    "wave_channel": ((0.5, 12.0), 10, "channel"),
    "wave_none": ((0.5, 12.0), 10, "none"),
    "wave_band0.5-6": ((0.5, 6.0), 10, "global"),
    "wave_step5": ((0.5, 12.0), 5, "global"),
}
POWER_BANDS = ((0.5, 2), (2, 4), (4, 8), (8, 12), (12, 20))


def log_power(windows):
    f, p = signal.welch(windows, fs=lib.FS, nperseg=250, axis=1)
    return np.concatenate([np.log(p[:, (f >= a) & (f < b)].mean(1) + 1e-12) for a, b in POWER_BANDS], axis=1)


def xcorr_predict(stream, templates, labels, max_shift):
    """stream (n, T, C), templates (k, T, C): class score = max normalized correlation over templates and shifts."""
    def norm(a):
        a = a - a.mean(1, keepdims=True)
        return a
    s, tp = norm(stream), norm(templates)
    scores = np.full((len(stream), 4), -np.inf)
    T = stream.shape[1]
    for shift in range(-max_shift, max_shift + 1):
        a = s[:, max(shift, 0):T + min(shift, 0)]
        b = tp[:, max(-shift, 0):T + min(-shift, 0)]
        num = np.einsum("ntc,ktc->nk", a, b)
        den = np.sqrt(np.einsum("ntc,ntc->n", a, a)[:, None] * np.einsum("ktc,ktc->k", b, b)[None]) + 1e-12
        r = num / den
        for c in range(4):
            if np.any(labels == c):
                scores[:, c] = np.maximum(scores[:, c], r[:, labels == c].max(1))
    return scores.argmax(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=lib.DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    data = lib.load_dev(args.dataset)
    lib.start_run(args.output_dir, __file__, args.dataset,
                  {"cov": {k: [list(v[0]), list(v[1]), v[2]] for k, v in COV.items()},
                   "wave": {k: [list(v[0]), v[1], v[2]] for k, v in WAVE.items()}, "null_seeds": args.null_seeds})
    streams, minutes = lib.wear_streams(data), lib.recording_minutes(data)
    wears = list(streams)
    label_sets = {None: data["y"]} | {s: v21_data.block_permuted_labels(data, s) for s in args.null_seeds}
    rows, saved = [], {}

    def record(feature, classifier, k, draw, results):
        pieces, pred = results[None]
        idx, positions, ranks, mins = lib.stream_context(data, streams, pieces, minutes)
        row = {"feature": feature, "classifier": classifier, "startup_actions": k, "draw": str(draw),
               **lib.stream_report(data, idx, pred, positions, ranks, mins)}
        row["null_accuracy"] = [float(np.mean(data["y"][idx] == results[s][1])) for s in args.null_seeds]
        row["null_prediction_counts"] = [np.bincount(results[s][1], minlength=4).tolist() for s in args.null_seeds]
        full = np.full(len(data["y"]), -1, np.int8)
        full[idx] = pred
        saved[f"{feature}|{classifier}|K{k}|d{draw}"] = full
        rows.append(row)
        print(f"{feature:18s} {classifier:9s} K={k} d{draw!s:4s} {lib.short(row)}", flush=True)

    def fold_std(vectors):
        out = {}
        for wear in wears:
            train = data["wear_group"] != wear
            centered = vectors.copy()
            for group in np.unique(data["group"][train]):
                m = data["group"] == group
                centered[m] = vectors[m] - vectors[m].mean(0)
            out[wear] = {"std": centered[train].std(0) + 1e-6}
        return out

    def calibrated(feature, classifier, make_x, k, draw, pops=None):
        results = {}
        for seed in [None] + list(args.null_seeds):
            pieces, preds = [], []
            for w_i, wear in enumerate(wears):
                anchors, anchor_y, rest = lib.calibration(streams[wear], data["y"], range(4), draw)
                if seed is not None:
                    anchor_y = np.random.default_rng([seed, w_i]).permutation(4)[anchor_y]
                pieces.append((wear, rest))
                if classifier == "xcorr":
                    preds.append(make_x(rest, anchors, anchor_y))
                    continue
                x, ax = make_x(rest, anchors)
                if classifier == "rc_adapt":
                    preds.append(run_stream(x, ax, anchor_y, "pop_z", "cum_g0.0", 0.1, pops[wear]))
                    continue
                update = "cum_g0.1" if classifier == "nc_adapt" else "none"
                preds.append(run_stream(x, ax, anchor_y, "anchor_z", update, 0.0, None))
            results[seed] = (pieces, np.concatenate(preds))
        record(feature, classifier, k, draw, results)

    def zero(feature, features, running):
        """features: per-stroke vectors; running: subtract recording mean (train) / running stream mean (test)."""
        results = {}
        for seed, labels in label_sets.items():
            pieces, preds = [], []
            for wear in wears:
                train = data["wear_group"] != wear
                x = features.copy()
                if running:
                    for group in np.unique(data["group"][train]):
                        m = data["group"] == group
                        x[m] = features[m] - features[m].mean(0)
                mean, std = x[train].mean(0), x[train].std(0) + 1e-6
                model = LogisticRegression(C=0.1, max_iter=3000).fit((x[train] - mean) / std, labels[train])
                rest = streams[wear]["order"]
                f = features[rest]
                if running:
                    past = np.cumsum(f, 0) - f
                    counts = np.arange(len(f))[:, None]
                    ref = np.where(counts > 0, past / np.maximum(counts, 1), features[train].mean(0))
                    f = f - ref
                pieces.append((wear, rest))
                preds.append(model.predict((f - mean) / std))
            results[seed] = (pieces, np.concatenate(preds))
        record(feature, "zero", 0, "-", results)

    calibrations = [(4, 0), (4, 1), (8, "both")]
    for name, (band, lags, norm) in COV.items():
        covs = lib.covariances(data["X"], band, lags, norm)

        def make_cov(rest, anchors, covs=covs):
            ref = covs[anchors].mean(0)
            return lib.tangent(covs[rest], ref), lib.tangent(covs[anchors], ref)
        vectors = lib.log_euclid(covs)
        pops = fold_std(vectors)

        def make_le(rest, anchors, vectors=vectors):
            return vectors[rest], vectors[anchors]
        for k, draw in calibrations:
            for classifier in ("nc", "nc_adapt"):
                calibrated(name, classifier, make_cov, k, draw)
            calibrated(name, "rc_adapt", make_le, k, draw, pops)
        zero(name, vectors, running=True)

    vector_features = {name: lib.waveform(data["X"], band, step, norm) for name, (band, step, norm) in WAVE.items()}
    vector_features["logpow"] = log_power(lib.bandpass(data["X"], (0.1, 30.0)))
    for name, features in vector_features.items():
        def make_vec(rest, anchors, features=features):
            return features[rest], features[anchors]
        pops = fold_std(features)
        for k, draw in calibrations:
            for classifier in ("nc", "nc_adapt"):
                calibrated(name, classifier, make_vec, k, draw)
            calibrated(name, "rc_adapt", make_vec, k, draw, pops)
            if name.startswith("wave"):
                band, step, norm = WAVE[name]
                shaped = features.reshape(len(features), -1, 4)
                shift = round(25 / step)
                calibrated(name, "xcorr", lambda rest, anchors, anchor_y, shaped=shaped, shift=shift:
                           xcorr_predict(shaped[rest], shaped[anchors], anchor_y, shift), k, draw)
        zero(name, features, running=False)
    lib.finish_run(args.output_dir, rows, saved)


if __name__ == "__main__":
    main()
