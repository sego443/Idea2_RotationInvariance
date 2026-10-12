"""Summarize a v21 development grid: per-run macro-F1 by mode and seed-vote ensembles per configuration."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def config_key(arguments: dict) -> str:
    null = "" if arguments.get("null_seed") in (None, "None") else "__null"
    return (f"{arguments['protocol']}__{arguments['backbone']}__ep{arguments['episodic_weight']}"
            f"__{arguments.get('episode_support', 'same')}__aug{arguments['augment']}{null}")


def macro_f1(true, predicted) -> float:
    return float(f1_score(true, predicted, average="macro", labels=list(range(4))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+", help="Grid roots, e.g. an original root and its recovery roots")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = defaultdict(list)
    for marker in sorted(m for root in args.roots for m in root.glob("*/RUN_COMPLETE.json")):
        run = marker.parent
        manifest = json.loads((run / "run_manifest.json").read_text())
        runs[config_key(manifest["arguments"])].append(run)
    report = {}
    for key, members in sorted(runs.items()):
        per_seed = defaultdict(list)
        votes = defaultdict(list)
        for run in members:
            for name, modes in json.loads((run / "metrics.json").read_text()).items():
                for mode, metric in modes.items():
                    per_seed[(name, mode)].append(metric["macro_f1"])
            with (run / "predictions.csv").open() as file:
                rows = list(csv.DictReader(file))
            for row in rows:
                for mode in row:
                    if mode.startswith(("zero", "cal_", "calnb_", "unsup_")):
                        votes[(row["evaluation"], row["source"], row["sample_index"], mode)].append(int(row[mode]))
                votes[(row["evaluation"], row["source"], row["sample_index"], "true")] = [int(row["true"])]
        ensemble = defaultdict(lambda: ([], []))
        for (name, source, index, mode), predicted in votes.items():
            if mode == "true":
                continue
            counts = np.bincount(predicted, minlength=4)
            ensemble[(name, mode)][0].append(votes[(name, source, index, "true")][0])
            ensemble[(name, mode)][1].append(int(counts.argmax()))
        report[key] = {
            "n_runs": len(members),
            "modes": {f"{name}/{mode}": {"seed_mean": float(np.mean(values)), "seed_sd": float(np.std(values)),
                                         "seed_vote": macro_f1(*ensemble[(name, mode)])}
                      for (name, mode), values in sorted(per_seed.items())},
        }
    args.output.write_text(json.dumps({"roots": [str(r) for r in args.roots], "configs": report}, indent=2))
    for key, entry in report.items():
        print(f"\n{key}  (runs={entry['n_runs']})")
        for mode, values in entry["modes"].items():
            print(f"  {mode:38s} mean {values['seed_mean']:.3f} ±{values['seed_sd']:.3f}  vote {values['seed_vote']:.3f}")


if __name__ == "__main__":
    main()
