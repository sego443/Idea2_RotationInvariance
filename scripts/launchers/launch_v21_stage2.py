"""Midday-break launcher: fine-tune every frozen Stage-1 checkpoint on C170 + W01 + W02 Part A, then write S2's manifest.

The recipe is fixed in docs/designs/2026-10-12-v21-wear-calibrated-classifier.md:
- every real and null S1 run is fine-tuned for 30 epochs at lr 3e-4 (null runs keep their null seed);
- endpoint families and tau values are copied from S1;
- S2 is evaluated on Part B only.
--epochs exists only for the synthetic rehearsal; a formal S2 must use the default.
The output root must not exist.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINETUNE_EPOCHS = 30
FINETUNE_LR = 3e-4


def run_one(command: list[str]) -> int:
    log = Path(command[command.index("--output-dir") + 1]).with_suffix(".log")
    with log.open("w") as handle:
        return subprocess.run([sys.executable, "-m", "scripts.runners.run_v21_wear_calibrated", *command],
                              cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-freeze", type=Path, required=True)
    parser.add_argument("--w02-dataset", type=Path, required=True, help="Segmented W02 dataset containing Part A")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--s2-freeze-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS, help="Rehearsal only")
    args = parser.parse_args()
    s1 = json.loads(args.s1_freeze.read_text())
    if s1["stage"] != "S1":
        raise ValueError("--s1-freeze must be a Stage-1 manifest")
    args.output_root.mkdir(parents=True, exist_ok=False)

    jobs = {"real": [], "null": []}
    commands = []
    for kind in ("real", "null"):
        for entry in s1[f"{kind}_runs"]:
            arguments = entry["arguments"]
            name = f"{arguments['backbone']}__{kind}__s{arguments['seed']}"
            out = args.output_root / name
            command = ["--protocol", "all_current_plus_w02", "--w02-dataset", str(args.w02_dataset.resolve()),
                       "--w02-parts", "A", "--backbone", arguments["backbone"], "--seed", arguments["seed"],
                       "--episode-support", arguments.get("episode_support", "same"),
                       "--epochs", str(args.epochs), "--lr", str(FINETUNE_LR),
                       "--init-checkpoint", str(Path(entry["run"]).resolve() / "checkpoint.pt"),
                       "--cpu-threads", "2", "--output-dir", str(out)]
            if arguments.get("null_seed") not in (None, "None"):
                command += ["--null-seed", arguments["null_seed"]]
            commands.append(command)
            jobs[kind].append(out)
    (args.output_root / "commands.json").write_text(json.dumps(commands, indent=2))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run_one, commands))
    failed = [str(c[c.index("--output-dir") + 1]) for c, code in zip(commands, codes) if code != 0]
    if failed:
        raise SystemExit(f"S2 runs failed; do not freeze S2: {failed}")

    endpoints = s1["endpoints"]
    freeze = [sys.executable, "-m", "scripts.analysis.make_v21_freeze_manifest", "--stage", "S2",
              "--e1-backbone", endpoints["E1"]["backbone"], "--e1-tau", str(endpoints["E1"]["tau"]),
              "--e1b-backbone", endpoints["E1b"]["backbone"], "--e1b-tau", str(endpoints["E1b"]["tau"]),
              "--e2-backbone", endpoints["E2"]["backbone"], "--e2-tau", str(endpoints["E2"]["tau"]),
              "--e3-backbone", endpoints["E3"]["backbone"], "--evaluate-parts", "B",
              "--output", str(args.s2_freeze_output)]
    for out in jobs["real"]:
        freeze += ["--real-run", str(out)]
    for out in jobs["null"]:
        freeze += ["--null-run", str(out)]
    subprocess.run(freeze, cwd=PROJECT_ROOT, check=True)
    print(f"S2 frozen: {args.s2_freeze_output}. Commit it before starting Part B.")


if __name__ == "__main__":
    main()
