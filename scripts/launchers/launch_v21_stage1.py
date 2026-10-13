"""Train the v21 Stage-1 freeze candidates on C170 only (W01 excluded) and write the S1 freeze manifest.

The recipe is fixed in docs/designs/2026-10-12-v21-wear-calibrated-classifier.md: backbones B1 and B2 with seeds
17/29/43/61/89 (fixed 120 epochs, same-group episodes, augmentation on) and five B2 matched nulls (null seed = seed).
Endpoints: E1 B2 tau 1; E1b B1 tau 16; E2 B1 tau 1; E3 B1; evaluated on W02 Parts A and B.
--epochs exists only for the synthetic rehearsal; a formal S1 must use the default.
The output root and the manifest path must not exist. Commit the manifest before collecting W02.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (17, 29, 43, 61, 89)
BACKBONES = ("b1_complex_gram", "b2_signed_cnn")
NULL_BACKBONE = "b2_signed_cnn"


def run_one(command: list[str]) -> int:
    log = Path(command[command.index("--output-dir") + 1]).with_suffix(".log")
    with log.open("w") as handle:
        return subprocess.run([sys.executable, "-m", "scripts.runners.run_v21_wear_calibrated", *command],
                              cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--s1-freeze-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="Torch device for every run, e.g. cuda")
    parser.add_argument("--epochs", type=int, default=120, help="Rehearsal only")
    args = parser.parse_args()
    if args.s1_freeze_output.exists():
        raise FileExistsError(args.s1_freeze_output)
    args.output_root.mkdir(parents=True, exist_ok=False)

    jobs = {"real": [], "null": []}
    commands = []
    for kind in ("real", "null"):
        for backbone in BACKBONES if kind == "real" else (NULL_BACKBONE,):
            for seed in SEEDS:
                out = args.output_root / f"{backbone}__{kind}__s{seed}"
                command = ["--protocol", "c170_all", "--backbone", backbone, "--seed", str(seed),
                           "--epochs", str(args.epochs), "--cpu-threads", "2", "--device", args.device,
                           "--output-dir", str(out)]
                if kind == "null":
                    command += ["--null-seed", str(seed)]
                commands.append(command)
                jobs[kind].append(out)
    (args.output_root / "commands.json").write_text(json.dumps(commands, indent=2))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run_one, commands))
    failed = [str(c[c.index("--output-dir") + 1]) for c, code in zip(commands, codes) if code != 0]
    if failed:
        raise SystemExit(f"S1 runs failed; do not freeze S1: {failed}")

    freeze = [sys.executable, "-m", "scripts.analysis.make_v21_freeze_manifest", "--stage", "S1",
              "--e1-backbone", "b2_signed_cnn", "--e1-tau", "1", "--e1b-backbone", "b1_complex_gram", "--e1b-tau", "16",
              "--e2-backbone", "b1_complex_gram", "--e2-tau", "1", "--e3-backbone", "b1_complex_gram",
              "--evaluate-parts", "A", "B", "--output", str(args.s1_freeze_output)]
    for out in jobs["real"]:
        freeze += ["--real-run", str(out)]
    for out in jobs["null"]:
        freeze += ["--null-run", str(out)]
    subprocess.run(freeze, cwd=PROJECT_ROOT, check=True)
    print(f"S1 frozen: {args.s1_freeze_output}. Commit it before collecting W02.")


if __name__ == "__main__":
    main()
