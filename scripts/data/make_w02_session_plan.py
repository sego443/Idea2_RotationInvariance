"""Generate a randomized W02 rewear acquisition plan and OpenBCI Sequence Collection prompts.

Each wear is a full don/doff of the band at a scheduled angle and contains one REST recording and several TRIAL
recordings. Every TRIAL recording starts with a labeled calibration prefix (C, cal_per_direction strokes per
direction) followed by the scored body (S, trial_per_direction strokes per direction). Directions are interleaved
in a random, class-balanced order inside both parts so that direction is not confounded with recording or time. Angles are stratified over
360 degrees and wear order is randomized. The output directory must not exist.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DIRECTION_CODES = {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT"}
PARTS = ("A", "B")
ANGLE_RESOLUTION_DEG = 5
MAX_RUN_LENGTH = 2
REST_SECONDS = 20
STROKE_PERIOD_S = 3
RECORDING_MARGIN_S = 8
PLAN_COLUMNS = (
    "row", "participant", "part", "day", "wear", "scheduled_angle_deg", "recording", "kind", "prefix", "body",
    "sequence", "n_prefix", "n_strokes", "prompt", "expected_duration_s", "actual_angle_deg", "strap_hole", "actual_filename",
    "accepted", "retry_of", "notes",
)


def balanced_sequence(rng: np.random.Generator, per_direction: int) -> str:
    """Random order with each direction exactly per_direction times and no run longer than MAX_RUN_LENGTH."""
    if per_direction < 1:
        raise ValueError("per_direction must be positive")
    pool = list(DIRECTION_CODES) * per_direction
    for _ in range(10_000):
        order = [pool[index] for index in rng.permutation(len(pool))]
        if all(len(set(order[i:i + MAX_RUN_LENGTH + 1])) > 1 for i in range(len(order) - MAX_RUN_LENGTH)):
            return "".join(order)
    raise RuntimeError("could not satisfy the run-length constraint")


def stratified_angles(rng: np.random.Generator, n_wears: int) -> list[int]:
    """One angle per equal-width 360-degree stratum, jittered within the stratum, returned in random order."""
    if n_wears < 4:
        raise ValueError("at least four wears are required to cover the circle")
    width = 360 / n_wears
    angles = []
    for stratum in range(n_wears):
        low = stratum * width
        candidates = [a for a in range(0, 360, ANGLE_RESOLUTION_DEG) if low <= a < low + width]
        if not candidates:
            raise ValueError("stratum narrower than the angle resolution")
        angles.append(int(rng.choice(candidates)))
    return [angles[index] for index in rng.permutation(n_wears)]


def build_plan(participant: str, part: str, day: int, n_wears: int, n_trials: int, cal_per_direction: int,
               trial_per_direction: int, seed: int, first_wear: int = 1) -> list[dict]:
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    if not participant.isalnum():
        raise ValueError("participant must be alphanumeric")
    rng = np.random.default_rng(seed)
    rows = []
    for wear_index, angle in enumerate(stratified_angles(rng, n_wears), start=first_wear):
        recordings = [("REST", "", "")]
        recordings += [(f"T{trial}", balanced_sequence(rng, cal_per_direction), balanced_sequence(rng, trial_per_direction))
                       for trial in range(1, n_trials + 1)]
        for recording, prefix, body in recordings:
            stem = f"{participant}_{part}D{day}_W{wear_index:02d}_A{angle:03d}_{recording}"
            prompt = f"{stem}_C{prefix}_S{body}" if body else stem
            sequence = prefix + body
            rows.append({
                "row": len(rows) + 1, "participant": participant, "part": part, "day": day, "wear": wear_index,
                "scheduled_angle_deg": angle, "recording": recording, "kind": "rest" if not body else "trial",
                "prefix": prefix, "body": body, "sequence": sequence, "n_prefix": len(prefix),
                "n_strokes": len(sequence), "prompt": prompt,
                "expected_duration_s": REST_SECONDS if not body else len(sequence) * STROKE_PERIOD_S + RECORDING_MARGIN_S,
                "actual_angle_deg": "", "strap_hole": "", "actual_filename": "", "accepted": "", "retry_of": "", "notes": "",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--part", choices=PARTS, required=True, help="A: first half of the day; B: second half")
    parser.add_argument("--first-wear", type=int, default=1, help="Wear number of the first wear (continue numbering in part B)")
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--n-wears", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--cal-per-direction", type=int, default=2)
    parser.add_argument("--trial-per-direction", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; must not exist")
    args = parser.parse_args()

    rows = build_plan(args.participant, args.part, args.day, args.n_wears, args.n_trials,
                      args.cal_per_direction, args.trial_per_direction, args.seed, args.first_wear)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stem = f"w02_{args.participant}_{args.part}D{args.day}"
    plan_path = args.output_dir / f"{stem}_plan.csv"
    with plan_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    prompts_path = args.output_dir / f"{stem}_prompts.txt"
    prompts_path.write_text("".join(f"{row['prompt']}\n" for row in rows), encoding="utf-8")
    strokes = [row for row in rows if row["n_strokes"]]
    manifest = {
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "n_recordings": len(rows),
        "n_strokes": sum(row["n_strokes"] for row in strokes),
        "scheduled_angles_in_wear_order": [row["scheduled_angle_deg"] for row in rows if row["recording"] == "REST"],
        "estimated_recording_minutes": round(sum(row["expected_duration_s"] for row in rows) / 60, 1),
    }
    (args.output_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
