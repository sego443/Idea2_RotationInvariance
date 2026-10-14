"""Exploratory window variants for the W02 development wears, cut directly from the raw recordings.

Only plan rows of development wears are read; sealed wears (W07, W10, W12) are filtered out before any raw file is
opened. Windows (250 Hz samples):
  peak2.0     2 s centred on the slot energy maximum (the current segmentation)
  peak2.5     2.5 s centred on the slot energy maximum
  peak1.5     1.5 s centred on the slot energy maximum
  slot3.0     the full 3 s grid slot
  slot+0.5    3 s grid slot shifted 0.5 s later
Every variant is scored with the 2-shot in-recording nearest centroid on lagged-covariance tangent features (lags
0..30 step 5, 0.5-12 Hz) and with the wear-level logistic classifier on all prefixes of the wear.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.analysis.explore_w02_fewshot import evaluate, feat_cov, summarize, tangent, classify
from scripts.data.prepare_c170_session import active_channels, load_recording
from scripts.data.prepare_w02_session import LETTERS, PREPARE_SECONDS, PROMPT_RE, SAMPLE_RATE, SLOT_SAMPLES, find_file, stroke_energy

SEALED = {7, 10, 12}
PLANS = [Path("data/w02_plans/P01_AD1_timed/w02_P01_AD1_plan_accepted.csv"),
         Path("data/w02_plans/P01_BD1_timed/w02_P01_BD1_plan_accepted.csv")]
SESSIONS = sorted(Path("data/raw/w02_data").glob("P01_*D1/OpenBCISession_*"))
VARIANTS = {"peak2.0": ("peak", 500, 0), "peak2.5": ("peak", 625, 0), "peak1.5": ("peak", 375, 0),
            "slot3.0": ("slot", 750, 0), "slot+0.5": ("slot", 750, 125)}


def cut(variant):
    kind, length, shift = VARIANTS[variant]
    X, y, group, wear_group, calib = [], [], [], [], []
    for plan in PLANS:
        with plan.open(encoding="utf-8") as file:
            rows = [r for r in csv.DictReader(file) if r["accepted"] == "yes" and int(r["wear"]) not in SEALED]
        for row in rows:
            match = PROMPT_RE.fullmatch(row["prompt"])
            if match["recording"] == "REST":
                continue
            path = find_file(SESSIONS, row["filename_label"], row["actual_filename"])
            samples = load_recording(path)
            channels = active_channels(samples, path)
            sequence = match["prefix"] + match["body"]
            energy = stroke_energy(samples, channels)
            for k, letter in enumerate(sequence):
                start = PREPARE_SECONDS * SAMPLE_RATE + k * SLOT_SAMPLES
                if kind == "peak":
                    centre = start + int(np.argmax(energy[start:start + SLOT_SAMPLES]))
                    begin = int(np.clip(centre - length // 2, 0, len(samples) - length))
                else:
                    begin = int(min(start + shift, len(samples) - length))
                X.append(samples[begin:begin + length, channels].astype(np.float32))
                y.append(LETTERS[letter])
                group.append(f"{match['part']}_W{match['wear']}_{match['recording']}")
                wear_group.append(f"{match['part']}_W{match['wear']}")
                calib.append(k < len(match["prefix"]))
    return {"X": np.stack(X), "y": np.array(y), "group": np.array(group), "wear_group": np.array(wear_group),
            "is_calibration": np.array(calib)}


def two_shot(data, covs):
    predicted = np.full(len(data["y"]), -1)
    for g in np.unique(data["group"]):
        pre = np.flatnonzero((data["group"] == g) & data["is_calibration"])
        body = np.flatnonzero((data["group"] == g) & ~data["is_calibration"])
        sup = np.concatenate([pre[data["y"][pre] == c][:2] for c in range(4)])
        ref = covs[sup].mean(0)
        predicted[body] = classify("nc", tangent(covs[sup], ref), data["y"][sup], tangent(covs[body], ref))
    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for variant in VARIANTS:
        data = cut(variant)
        covs = feat_cov(data["X"], lags=(0, 5, 10, 15, 20, 25, 30))
        a = summarize(data, two_shot(data, covs))
        b = summarize(data, evaluate(data, covs, "logreg", "wear", cov=True))
        rows[variant] = {"two_shot": a, "wear_logreg": b}
        print(f"{variant:9s} n={len(data['y'])} 2-shot F1 {a['macro_f1']:.3f} minW {a['min_wear_f1']:.3f} | "
              f"wear logreg F1 {b['macro_f1']:.3f} minW {b['min_wear_f1']:.3f}", flush=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
