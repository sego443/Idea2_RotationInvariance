"""Summarize the v22 window/normalization ablation grid from per-run predictions.

For every (protocol, window, norm) and mode it reports the seed mean/SD of macro-F1, the 3-seed majority vote,
pooled per-class recall, prediction counts, the share of errors that pick the opposite direction (UP<->DOWN,
LEFT<->RIGHT) and the matched block-permuted null. C170 held-out scores are split into the old-validation and
exposed registered-test positions. Writes JSON and a Markdown table; the output files must not exist.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from src import v21_data as data_module

MODES = ("zero", "cal_tau1", "unsup_tau1", "calnb_tau4")
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
SUBSETS = {
    "c170_val": lambda row: row["evaluation"] == "c170_heldout"
    and int(row["group"][6:]) in data_module.C170_OLD_VALIDATION_MM,
    "c170_test": lambda row: row["evaluation"] == "c170_heldout"
    and int(row["group"][6:]) in data_module.C170_EXPOSED_TEST_MM,
    "w01": lambda row: row["evaluation"] == "w01",
}


def macro_f1(true, predicted) -> float:
    return float(f1_score(true, predicted, average="macro", labels=list(range(4))))


def describe(true: np.ndarray, predicted: np.ndarray) -> dict:
    errors = predicted != true
    return {
        "recall": {c: float(np.mean(predicted[true == i] == i)) for i, c in enumerate(data_module.CLASSES)},
        "prediction_counts": dict(zip(data_module.CLASSES, np.bincount(predicted, minlength=4).tolist())),
        "opposite_error_share": float(np.mean([OPPOSITE[t] == p for t, p in zip(true[errors], predicted[errors])]))
        if errors.any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_json, args.output_md):
        if path.exists():
            raise FileExistsError(path)

    runs = defaultdict(list)
    for marker in sorted(args.root.glob("*/RUN_COMPLETE.json")):
        arguments = json.loads((marker.parent / "run_manifest.json").read_text())["arguments"]
        null = arguments["null_seed"] not in (None, "None")
        runs[(arguments["protocol"], arguments["window"], arguments["norm"], null)].append(marker.parent)

    report = {}
    for (protocol, window, norm, null), members in sorted(runs.items()):
        key = f"{protocol}__{window}__{norm}" + ("__null" if null else "")
        entry = {"n_runs": len(members), "runs": [m.name for m in members], "subsets": {}}
        rows_by_run = []
        for run in members:
            with (run / "predictions.csv").open() as file:
                rows_by_run.append(list(csv.DictReader(file)))
        for subset, keep in SUBSETS.items():
            selected = [[row for row in rows if keep(row)] for rows in rows_by_run]
            if not selected[0]:
                continue
            true = np.array([int(r["true"]) for r in selected[0]])
            modes = {}
            for mode in MODES:
                if mode not in selected[0][0]:
                    continue
                per_seed = [np.array([int(r[mode]) for r in rows]) for rows in selected]
                scores = [macro_f1(true, p) for p in per_seed]
                vote = np.array([np.bincount(v, minlength=4).argmax() for v in np.stack(per_seed).T])
                pooled_true = np.tile(true, len(per_seed))
                modes[mode] = {"seed_mean": float(np.mean(scores)), "seed_sd": float(np.std(scores)),
                               "per_seed": scores, "vote": macro_f1(true, vote),
                               **describe(pooled_true, np.concatenate(per_seed))}
            entry["subsets"][subset] = {"n_scored": int(len(true)), "modes": modes}
        report[key] = entry
    args.output_json.write_text(json.dumps(report, indent=2))

    lines = ["| Protocol / subset | Window | Norm | Runs | " + " | ".join(MODES) + " | Null ZERO / CAL τ=1 |",
             "|---|---|---|---:|" + "---:|" * len(MODES) + "---|"]
    for key, entry in report.items():
        if key.endswith("__null"):
            continue
        protocol, window, norm = key.split("__")
        null = report.get(key + "__null", {}).get("subsets", {})
        for subset, values in entry["subsets"].items():
            cells = []
            for mode in MODES:
                m = values["modes"].get(mode)
                cells.append(f"{m['seed_mean']:.3f} ± {m['seed_sd']:.3f} (v {m['vote']:.3f})" if m else "—")
            n = null.get(subset, {}).get("modes", {})
            null_cell = f"{n['zero']['seed_mean']:.3f} / {n['cal_tau1']['seed_mean']:.3f}" if n else "—"
            lines.append(f"| {subset} | {window} | {norm} | {entry['n_runs']} | " + " | ".join(cells) + f" | {null_cell} |")
    args.output_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
