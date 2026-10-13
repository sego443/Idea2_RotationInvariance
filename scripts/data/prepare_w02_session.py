"""Segment a W02 session into labeled stroke windows using the plan's prompt sequences.

Inputs: one or more OpenBCI session folders and the completed plan CSV(s) from make_w02_session_plan.py. Only plan
rows with accepted=yes are used; their files are found by prompt suffix. Stroke detection uses the C170 localizer
parameters (0.5-12 Hz, 75-sample envelope, distance 525, prominence 1.5) but requires exactly the planned number
of strokes; any other count flags the recording instead of repairing it. Labels come from the prompt letters in
order; strokes from the C prefix are marked is_prefix (calibration strokes, never scored). REST recordings provide non-overlapping 2 s windows for unsupervised gain estimates. Raw files are never
modified; the output directory must not exist. Wear positions come from the plan's actual_tape_mm (the wrist-tape
reading at the S3/S0 midpoint) and are converted to the S0 band angle with make_w02_session_plan.band_angle_deg.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import signal

from scripts.data.make_w02_session_plan import band_angle_deg
from scripts.data.prepare_c170_session import active_channels, load_recording

SAMPLE_RATE = 250
WINDOW_SAMPLES = 500
CLASSES = ("UP", "DOWN", "LEFT", "RIGHT")
LETTERS = {"U": 0, "D": 1, "L": 2, "R": 3}
PROMPT_RE = re.compile(
    r"(?P<participant>[A-Za-z0-9]+)_(?P<part>[AB])D(?P<day>\d+)_W(?P<wear>\d{2})_M(?P<tape>\d{3})_"
    r"(?P<recording>REST|T\d)(?:_C(?P<prefix>[UDLR]+)_S(?P<body>[UDLR]+))?")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stroke_centers(samples: np.ndarray, channels: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [0.5, 12], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfiltfilt(sos, samples[:, channels], axis=0)
    median = np.median(filtered, axis=0)
    scale = np.median(np.abs(filtered - median), axis=0) + 1e-9
    energy = np.sqrt(np.mean((filtered / scale) ** 2, axis=1))
    energy = np.convolve(energy, np.ones(75) / 75, mode="same")
    peaks, _ = signal.find_peaks(energy, distance=525, prominence=1.5)
    return peaks[(peaks >= WINDOW_SAMPLES // 2) & (peaks + WINDOW_SAMPLES // 2 <= len(samples))]


def find_file(session_dirs: list[Path], prompt: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", prompt):
        raise ValueError(f"invalid filename label: {prompt}")
    matches = [path for directory in session_dirs for path in directory.glob(f"*_{prompt}.txt")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one accepted file for {prompt}, found {len(matches)}")
    return matches[0]


def prepare(session_dirs: list[Path], plan_paths: list[Path], output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    rows = []
    for plan_path in plan_paths:
        with plan_path.open(encoding="utf-8") as file:
            rows += [row for row in csv.DictReader(file) if row["accepted"].strip().lower() == "yes"]
    prompts = [row["prompt"] for row in rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("accepted plan rows contain duplicate prompts")

    windows, rests, stroke_rows, rest_rows, recording_rows, flagged = [], [], [], [], [], []
    for row in rows:
        match = PROMPT_RE.fullmatch(row["prompt"])
        if match is None:
            raise ValueError(f"invalid prompt: {row['prompt']}")
        path = find_file(session_dirs, row.get("filename_label") or row["prompt"])
        samples = load_recording(path)
        channels = active_channels(samples, path)
        actual_tape = float(row["actual_tape_mm"]) if row["actual_tape_mm"].strip() else None
        meta = {"participant": match["participant"], "part": match["part"], "day": int(match["day"]),
                "wear": int(match["wear"]), "scheduled_tape_mm": int(match["tape"]),
                "scheduled_angle_deg": round(band_angle_deg(int(match["tape"]))), "actual_tape_mm": actual_tape,
                "actual_angle_deg": None if actual_tape is None else round(band_angle_deg(actual_tape)),
                "recording": match["recording"], "source_filename": path.name,
                "channel_map": ",".join(map(str, channels))}
        record = {**meta, "sha256": sha256(path), "raw_rows": len(samples)}
        if match["recording"] == "REST":
            for start in range(WINDOW_SAMPLES, len(samples) - 2 * WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                rests.append(samples[start:start + WINDOW_SAMPLES, channels].astype(np.float32))
                rest_rows.append({**meta, "start_row": start})
            recording_rows.append({**record, "status": "rest", "detected": ""})
            continue
        prefix, body = match["prefix"] or "", match["body"] or ""
        sequence = prefix + body
        centers = stroke_centers(samples, channels)
        if len(centers) != len(sequence):
            flagged.append({**record, "expected": len(sequence), "detected": len(centers)})
            recording_rows.append({**record, "status": "flagged_count_mismatch", "detected": len(centers)})
            continue
        for position, (letter, center) in enumerate(zip(sequence, centers), start=1):
            start = int(center) - WINDOW_SAMPLES // 2
            windows.append(samples[start:start + WINDOW_SAMPLES, channels].astype(np.float32))
            stroke_rows.append({**meta, "position_in_recording": position, "is_prefix": position <= len(prefix),
                                "label": LETTERS[letter],
                                "direction": CLASSES[LETTERS[letter]], "center_row": int(center)})
        recording_rows.append({**record, "status": "segmented", "detected": len(centers)})

    output_dir.mkdir(parents=True)
    keys = ("participant", "part", "day", "wear", "scheduled_tape_mm", "recording", "source_filename")
    np.savez_compressed(
        output_dir / "dataset.npz",
        X_raw=np.stack(windows) if windows else np.zeros((0, WINDOW_SAMPLES, 4), np.float32),
        label=np.array([r["label"] for r in stroke_rows], dtype=np.int64),
        wear=np.array([r["wear"] for r in stroke_rows], dtype=np.int64),
        part=np.array([r["part"] for r in stroke_rows]),
        recording=np.array([r["recording"] for r in stroke_rows]),
        position_in_recording=np.array([r["position_in_recording"] for r in stroke_rows], dtype=np.int64),
        is_prefix=np.array([r["is_prefix"] for r in stroke_rows], dtype=bool),
        scheduled_tape_mm=np.array([r["scheduled_tape_mm"] for r in stroke_rows], dtype=np.int64),
        actual_tape_mm=np.array([np.nan if r["actual_tape_mm"] is None else r["actual_tape_mm"] for r in stroke_rows]),
        scheduled_angle_deg=np.array([r["scheduled_angle_deg"] for r in stroke_rows], dtype=np.int64),
        actual_angle_deg=np.array([-1 if r["actual_angle_deg"] is None else r["actual_angle_deg"] for r in stroke_rows],
                                  dtype=np.int64),
        participant=np.array([r["participant"] for r in stroke_rows]),
        rest_X_raw=np.stack(rests) if rests else np.zeros((0, WINDOW_SAMPLES, 4), np.float32),
        rest_wear=np.array([r["wear"] for r in rest_rows], dtype=np.int64),
        rest_part=np.array([r["part"] for r in rest_rows]),
    )
    for name, table in (("strokes.csv", stroke_rows), ("recordings.csv", recording_rows), ("flagged.csv", flagged)):
        if table:
            with (output_dir / name).open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    manifest = {
        "plans": {str(p): sha256(p) for p in plan_paths},
        "session_dirs": [str(d) for d in session_dirs],
        "source_sha256": sha256(Path(__file__)),
        "n_strokes": len(windows), "n_rest_windows": len(rests), "n_flagged_recordings": len(flagged),
        "counts_by_class": dict(zip(CLASSES, np.bincount([r["label"] for r in stroke_rows], minlength=4).tolist())),
        "dataset_sha256": sha256(output_dir / "dataset.npz"),
        "group_keys": keys,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.session_dir, args.plan, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
