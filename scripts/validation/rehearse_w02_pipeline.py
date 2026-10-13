"""Rehearse the W02 acquisition-day pipeline end to end on synthetic OpenBCI files.

Steps: plan generation (parts A and B) -> synthetic raw recordings with direction-dependent bursts -> segmentation
-> Stage-1 launcher (short C170-only runs, W01 excluded, S1 freeze) -> Stage-2 launcher (fine-tuning with synthetic part A, S2 freeze)
-> one-shot S1 (parts A+B) and S2 (part B) evaluations. Scores are meaningless; the rehearsal only checks that every step runs and that artifacts and
hash checks line up. Everything is written below a new --output-dir.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LETTER_CHANNEL = {"U": 0, "D": 1, "L": 2, "R": 3}


def write_recording(path: Path, sequence: str, rng: np.random.Generator) -> None:
    rows = 250 * (3 + 3 * len(sequence) + 3)
    exg = np.zeros((rows, 8))
    exg[:, :4] = 1000 + rng.normal(0, 5, (rows, 4))
    t = np.arange(250) / 250
    for k, letter in enumerate(sequence):
        start = 250 * (3 + 3 * k)
        burst = 400 * np.sin(2 * np.pi * 3 * t) * np.hanning(250)
        exg[start:start + 250, :4] += 0.3 * burst[:, None]
        exg[start:start + 250, LETTER_CHANNEL[letter]] += burst
    header = ["%OpenBCI Raw EXG Data", "%Number of channels = 8", "%Sample Rate = 250 Hz", "%Board = Cyton"]
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(header) + "\n")
        file.write(", ".join(["Sample Index"] + [f"EXG Channel {i}" for i in range(8)]) + "\n")
        for index, values in enumerate(exg):
            file.write(", ".join([str(index)] + [f"{v:.3f}" for v in values]) + "\n")


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    session = root / "OpenBCISession_synthetic"
    session.mkdir()
    rng = np.random.default_rng(0)
    plans = []
    for part, first in (("A", 1), ("B", 5)):  # 4 wears per part; the first two of each are synthesized
        run("-m", "scripts.data.make_w02_session_plan", "--part", part, "--day", "1", "--n-wears", "4",
            "--first-wear", str(first), "--seed", str(first), "--output-dir", str(root / f"plan_{part}"))
        plan = next((root / f"plan_{part}").glob("*_plan.csv"))
        with plan.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        for row in rows:
            if int(row["wear"]) < first + 2:
                write_recording(session / f"OpenBCI-RAW-synthetic_{row['filename_label']}.txt", row["sequence"], rng)
                row.update(accepted="yes", actual_tape_mm=row["scheduled_tape_mm"])
            else:
                row.update(accepted="no", notes="not synthesized")
        completed = root / f"completed_plan_{part}.csv"
        with completed.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plans.append(completed)
    run("-m", "scripts.data.prepare_w02_session", "--session-dir", str(session), "--plan", str(plans[0]),
        "--plan", str(plans[1]), "--output-dir", str(root / "w02_dataset"))
    dataset = root / "w02_dataset" / "dataset.npz"
    s1 = root / "v21_freeze_S1.json"
    run("-m", "scripts.launchers.launch_v21_stage1", "--output-root", str(root / "stage1_runs"),
        "--s1-freeze-output", str(s1), "--epochs", str(args.epochs))
    s2 = root / "v21_freeze_S2.json"
    run("-m", "scripts.launchers.launch_v21_stage2", "--s1-freeze", str(s1), "--w02-dataset", str(dataset),
        "--output-root", str(root / "stage2_runs"), "--s2-freeze-output", str(s2), "--epochs", str(args.epochs))
    for stage, manifest in (("S1", s1), ("S2", s2)):
        run("-m", "scripts.analysis.evaluate_v21_frozen_w02", "--freeze-manifest", str(manifest), "--w02-dataset",
            str(dataset), "--output-dir", str(root / f"evaluation_{stage}"))
    print("rehearsal complete:", root)


if __name__ == "__main__":
    main()
