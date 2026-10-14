from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DIRECTION_CODES = {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT"}
PARTS = ("A", "B")
TAPE_CIRCUMFERENCE_MM = 170
TAPE_RESOLUTION_MM = 5
DORSAL_MIDLINE_MM = 10.0
MIDPOINT_TO_S0_MM = TAPE_CIRCUMFERENCE_MM / 8  # four equally spaced sensors; the S3/S0 midpoint is half a gap before S0
MAX_RUN_LENGTH = 2
REST_SECONDS = 20
STROKE_PERIOD_S = 3
PREPARE_SECONDS = 3
RECORDING_MARGIN_S = 6
PLAN_COLUMNS = (
    "row", "participant", "part", "day", "wear", "scheduled_tape_mm", "scheduled_angle_deg", "recording", "kind",
    "prefix", "body", "sequence", "n_prefix", "n_strokes", "prompt", "filename_label", "expected_duration_s", "actual_tape_mm",
    "strap_hole", "actual_filename",
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


def band_angle_deg(tape_mm: float) -> float:
    """Angle of S0 from the dorsal midline, in the tape direction, for a reading at the S3/S0 midpoint."""
    if not 0 <= tape_mm < TAPE_CIRCUMFERENCE_MM:
        raise ValueError(f"tape reading must be in [0, {TAPE_CIRCUMFERENCE_MM}) mm, got {tape_mm}")
    offset = (tape_mm + MIDPOINT_TO_S0_MM - DORSAL_MIDLINE_MM) % TAPE_CIRCUMFERENCE_MM
    return offset * 360 / TAPE_CIRCUMFERENCE_MM


def stratified_tape_positions(rng: np.random.Generator, n_wears: int) -> list[int]:
    """One tape reading per equal-width stratum of the circumference, jittered within it, returned in random order."""
    if n_wears < 4:
        raise ValueError("at least four wears are required to cover the circle")
    width = TAPE_CIRCUMFERENCE_MM / n_wears
    positions = []
    for stratum in range(n_wears):
        low = stratum * width
        candidates = [m for m in range(0, TAPE_CIRCUMFERENCE_MM, TAPE_RESOLUTION_MM) if low <= m < low + width]
        if not candidates:
            raise ValueError("stratum narrower than the tape resolution")
        positions.append(int(rng.choice(candidates)))
    return [positions[index] for index in rng.permutation(n_wears)]


def build_plan(participant: str, part: str, day: int, n_wears: int, n_trials: int, cal_per_direction: int,
               trial_per_direction: int, seed: int, first_wear: int = 1) -> list[dict]:
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}")
    if not participant.isalnum():
        raise ValueError("participant must be alphanumeric")
    rng = np.random.default_rng(seed)
    rows = []
    for wear_index, tape in enumerate(stratified_tape_positions(rng, n_wears), start=first_wear):
        recordings = [("REST", "", "")]
        recordings += [(f"T{trial}", balanced_sequence(rng, cal_per_direction), balanced_sequence(rng, trial_per_direction))
                       for trial in range(1, n_trials + 1)]
        for recording, prefix, body in recordings:
            stem = f"{participant}_{part}D{day}_W{wear_index:02d}_M{tape:03d}_{recording}"
            prompt = f"{stem}_C{prefix}_S{body}" if body else stem
            sequence = prefix + body
            rows.append({
                "row": len(rows) + 1, "participant": participant, "part": part, "day": day, "wear": wear_index,
                "scheduled_tape_mm": tape, "scheduled_angle_deg": round(band_angle_deg(tape)), "recording": recording, "kind": "rest" if not body else "trial",
                "prefix": prefix, "body": body, "sequence": sequence, "n_prefix": len(prefix),
                "n_strokes": len(sequence), "prompt": prompt, "filename_label": stem,
                "expected_duration_s": REST_SECONDS if not body else len(sequence) * STROKE_PERIOD_S + RECORDING_MARGIN_S,
                "actual_tape_mm": "", "strap_hole": "", "actual_filename": "", "accepted": "", "retry_of": "", "notes": "",
            })
    return rows


def build_widget_prompts(rows: list[dict]) -> dict:
    prompts = []
    for row in rows:
        sequence = row["sequence"]
        prompts.append({
            "filename_label": row["filename_label"],
            "bpm": 60 / STROKE_PERIOD_S if sequence else 60 / REST_SECONDS,
            "prepare_seconds": PREPARE_SECONDS if sequence else 0,
            "actions": [f"{DIRECTION_CODES[code]} - return to centre" for code in sequence]
                       if sequence else ["Hold still in centre"],
            "finish_seconds": 3 if sequence else 0,
        })
    return {"version": 1, "prompts": prompts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a randomized W02 rewear plan and timed JSON action prompts.")
    parser.add_argument("--participant", default="P01")
    parser.add_argument("--part", choices=PARTS, required=True, help="A: first half of the day; B: second half")
    parser.add_argument("--first-wear", type=int, default=1, help="Wear number of the first wear (continue numbering in part B)")
    parser.add_argument("--day", type=int, required=True)
    parser.add_argument("--n-wears", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=6)
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
    prompts_path = args.output_dir / f"{stem}_prompts.json"
    prompts_path.write_text(json.dumps(build_widget_prompts(rows), indent=2) + "\n", encoding="utf-8")
    strokes = [row for row in rows if row["n_strokes"]]
    manifest = {
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "n_recordings": len(rows),
        "n_strokes": sum(row["n_strokes"] for row in strokes),
        "scheduled_tape_mm_in_wear_order": [row["scheduled_tape_mm"] for row in rows if row["recording"] == "REST"],
        "scheduled_angles_in_wear_order": [row["scheduled_angle_deg"] for row in rows if row["recording"] == "REST"],
        "estimated_recording_minutes": round(sum(row["expected_duration_s"] for row in rows) / 60, 1),
    }
    (args.output_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
