"""Write a v21 freeze manifest from completed all-data training runs, before any W02 test wear is collected.

The manifest fixes the real and matched-null checkpoints (with hashes), the inference rule for every endpoint, the
calibration strength tau, the gate, the scorer source hash and the bootstrap seed. Commit it to Git before
collecting the wears it will be evaluated on. The output file must not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCORER = PROJECT_ROOT / "scripts" / "analysis" / "evaluate_v21_frozen_w02.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_entry(run: Path) -> dict:
    if not (run / "RUN_COMPLETE.json").exists():
        raise FileNotFoundError(f"incomplete run: {run}")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if manifest["arguments"]["protocol"] not in ("all_current", "all_current_plus_w02"):
        raise ValueError(f"freeze candidates must be all-data runs: {run}")
    return {"run": str(run), "checkpoint_sha256": sha256(run / "checkpoint.pt"),
            "arguments": manifest["arguments"], "source_sha256": manifest["source_sha256"],
            "data_sha256": manifest["data_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("S1", "S2"), required=True)
    parser.add_argument("--real-run", type=Path, action="append", required=True)
    parser.add_argument("--null-run", type=Path, action="append", required=True)
    parser.add_argument("--e1-backbone", required=True)
    parser.add_argument("--e1-tau", type=float, required=True)
    parser.add_argument("--e1b-backbone", required=True)
    parser.add_argument("--e1b-tau", type=float, required=True)
    parser.add_argument("--e2-backbone", required=True)
    parser.add_argument("--e2-tau", type=float, required=True)
    parser.add_argument("--e3-backbone", required=True)
    parser.add_argument("--evaluate-parts", nargs="+", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    backbones = {json.loads((run / "run_manifest.json").read_text())["arguments"]["backbone"] for run in args.real_run}
    for needed in (args.e1_backbone, args.e1b_backbone, args.e2_backbone, args.e3_backbone):
        if needed not in backbones:
            raise ValueError(f"no frozen real run for backbone {needed}")
    manifest = {
        "stage": args.stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classes": ["UP", "DOWN", "LEFT", "RIGHT"],
        "endpoints": {
            "E1": {"backbone": args.e1_backbone, "tau": args.e1_tau,
                   "support": "labeled C prefix (8 strokes) of the same trial recording"},
            "E1b": {"backbone": args.e1b_backbone, "tau": args.e1b_tau,
                    "support": "labeled C prefixes of the other trial recordings of the same wear"},
            "E2": {"backbone": args.e2_backbone, "tau": args.e2_tau,
                   "support": "same-recording prefix with labels ignored plus earlier body strokes (causal)"},
            "E3": {"backbone": args.e3_backbone},
        },
        "scored_strokes": "body (S) strokes of every trial recording of the evaluated parts; prefix strokes are never scored",
        "evaluate_parts": args.evaluate_parts,
        "ensemble_rule": "per-seed argmax, majority vote across seeds, ties to lowest class ID",
        "gate": {"macro_f1_min": 0.80, "per_class_recall_min": 0.65, "applies_to": "E1"},
        "bootstrap": {"unit": "wear", "resamples": 10000, "seed": 20261012},
        "real_runs": [checkpoint_entry(run) for run in args.real_run],
        "null_runs": [checkpoint_entry(run) for run in args.null_run],
        "scorer_sha256": sha256(SCORER),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps({"output": str(args.output), "n_real": len(args.real_run), "n_null": len(args.null_run)}, indent=1))


if __name__ == "__main__":
    main()
