from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import signal


PROMPT_RE = re.compile(r"B(?P<block>\d{3})_D(?P<distance>\d{3})MM_(?P<direction>UP|DOWN|LEFT|RIGHT)_R20")
FILE_RE = re.compile(r"_(?P<prompt>B\d{3}_D\d{3}MM_(?:UP|DOWN|LEFT|RIGHT)_R20)\.txt$")
EXG_COLUMNS = [f"EXG Channel {index}" for index in range(8)]
SAMPLE_RATE = 250
WINDOW_SAMPLES = 500
MIN_COMPLETE_ROWS = 12500
MAX_SHORT_ROWS = 2500


def source_digest(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_recording(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as file:
        metadata = [file.readline().strip() for _ in range(4)]
    if metadata[1] != "%Number of channels = 8" or metadata[2] != "%Sample Rate = 250 Hz":
        raise ValueError(f"Unexpected acquisition metadata: {path}")
    frame = pd.read_csv(path, skiprows=4, skipinitialspace=True, usecols=EXG_COLUMNS)
    samples = frame[EXG_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(samples).all():
        raise ValueError(f"EXG data contains non-finite values: {path}")
    return samples


def active_channels(samples: np.ndarray, path: Path) -> np.ndarray:
    channels = np.flatnonzero(np.mean(np.abs(samples) > 1, axis=0) > 0.95)
    if channels.tolist() not in ([0, 2, 4, 6], [0, 1, 2, 3]):
        raise ValueError(f"Unexpected column indices for the four active channels: {path}, {channels.tolist()}")
    return channels


def action_centers(samples: np.ndarray, channels: np.ndarray, path: Path) -> np.ndarray:
    sos = signal.butter(3, [0.5, 12], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfiltfilt(sos, samples[:, channels], axis=0)
    median = np.median(filtered, axis=0)
    scale = np.median(np.abs(filtered - median), axis=0) + 1e-9
    energy = np.sqrt(np.mean((filtered / scale) ** 2, axis=1))
    energy = np.convolve(energy, np.ones(75) / 75, mode="same")
    peaks, _ = signal.find_peaks(energy, distance=525, prominence=1.5)
    centers = peaks[peaks > 375]
    if not 19 <= len(centers) <= 22:
        raise ValueError(f"Movement count is outside the acquisition range: {path}; detected {len(centers)} movements")
    if centers[0] < WINDOW_SAMPLES // 2 or centers[min(len(centers), 20) - 1] + WINDOW_SAMPLES // 2 > len(samples):
        raise ValueError(f"Movement window extends beyond the recording: {path}")
    if np.max(np.diff(centers)) > 5 * SAMPLE_RATE:
        raise ValueError(f"Unexpected movement intervals: {path}")
    return centers


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(session_dir: Path, prompts_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not session_dir.is_dir():
        raise FileNotFoundError(session_dir)
    prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) != len(set(prompts)) or any(PROMPT_RE.fullmatch(label) is None for label in prompts):
        raise ValueError("Acquisition prompts contain duplicate or invalid labels")
    files: dict[str, Path] = {}
    for path in session_dir.glob("*.txt"):
        match = FILE_RE.search(path.name)
        if match is None:
            continue
        label = match.group("prompt")
        if label in files:
            raise ValueError(f"Multiple accepted files have the same label: {label}")
        files[label] = path
    if set(files) != set(prompts):
        raise ValueError(f"Accepted files differ from the acquisition prompts; missing {sorted(set(prompts) - set(files))}; extra {sorted(set(files) - set(prompts))}")

    windows: list[np.ndarray] = []
    segment_rows: list[dict] = []
    recording_rows: list[dict] = []
    excluded_rows: list[dict] = []
    channel_map_counts: dict[str, int] = {}
    for label in prompts:
        match = PROMPT_RE.fullmatch(label)
        assert match is not None
        block = int(match.group("block"))
        distance = int(match.group("distance"))
        direction = match.group("direction")
        path = files[label]
        samples = load_recording(path)
        count = len(samples)
        if count <= MAX_SHORT_ROWS:
            excluded_rows.append({
                "block": block, "distance_mm": distance, "direction": direction,
                "source_filename": path.name, "raw_rows": count,
                "duration_s": round(count / SAMPLE_RATE, 3),
                "sha256": source_digest(path), "reason": "short_recording",
            })
            continue
        if count < MIN_COMPLETE_ROWS:
            raise ValueError(f"Cannot classify recording length: {path}, {count} rows")

        channels = active_channels(samples, path)
        channel_map = ",".join(str(channel) for channel in channels)
        channel_map_counts[channel_map] = channel_map_counts.get(channel_map, 0) + 1
        centers = action_centers(samples, channels, path)
        source_filename = path.name
        retained = centers[:20]
        block_windows: list[np.ndarray] = []
        block_rows: list[dict] = []
        for repetition, center in enumerate(retained, start=1):
            start = int(center) - WINDOW_SAMPLES // 2
            end = start + WINDOW_SAMPLES
            block_windows.append(samples[start:end, channels].copy())
            block_rows.append({
                "dataset_index": len(windows) + len(block_windows) - 1,
                "block": block, "distance_mm": distance, "direction": direction,
                "repetition": repetition, "source_repetition": repetition,
                "is_duplicate": False, "source_filename": source_filename,
                "channel_map": channel_map, "start_row": start, "end_row": end,
                "center_row": int(center), "center_s": round(int(center) / SAMPLE_RATE, 3),
            })
        if len(block_windows) == 19:
            copied = block_rows[9].copy()
            copied["dataset_index"] = len(windows) + 19
            copied["repetition"] = 20
            copied["is_duplicate"] = True
            block_windows.append(block_windows[9].copy())
            block_rows.append(copied)
        if len(block_windows) != 20:
            raise ValueError(f"Unexpected movement count after processing: {label}")
        windows.extend(block_windows)
        segment_rows.extend(block_rows)
        gaps = np.diff(centers) / SAMPLE_RATE
        recording_rows.append({
            "block": block, "distance_mm": distance, "direction": direction,
            "source_filename": source_filename, "sha256": source_digest(path),
            "raw_rows": count, "duration_s": round(count / SAMPLE_RATE, 3),
            "channel_map": channel_map, "detected_count": len(centers),
            "retained_count": 20, "trimmed_count": max(len(centers) - 20, 0),
            "duplicate_count": max(20 - len(centers), 0),
            "first_center_s": round(int(centers[0]) / SAMPLE_RATE, 3),
            "last_center_s": round(int(centers[-1]) / SAMPLE_RATE, 3),
            "min_gap_s": round(float(gaps.min()), 3),
            "median_gap_s": round(float(np.median(gaps)), 3),
            "max_gap_s": round(float(gaps.max()), 3),
            "all_centers_s": ",".join(f"{center / SAMPLE_RATE:.3f}" for center in centers),
        })

    distances = sorted({row["distance_mm"] for row in recording_rows})
    if len(recording_rows) != 80 or len(excluded_rows) != 56 or len(windows) != 1600:
        raise ValueError("Acquisition file count does not match expectations")
    if any(sum(row["distance_mm"] == distance for row in recording_rows) != 4 for distance in distances):
        raise ValueError("A recorded position does not contain all four directions")

    dataset = {
        "X_raw": np.stack(windows),
        "block": np.asarray([row["block"] for row in segment_rows], dtype=np.uint16),
        "distance_mm": np.asarray([row["distance_mm"] for row in segment_rows], dtype=np.uint16),
        "direction": np.asarray([row["direction"] for row in segment_rows]),
        "repetition": np.asarray([row["repetition"] for row in segment_rows], dtype=np.uint8),
        "source_repetition": np.asarray([row["source_repetition"] for row in segment_rows], dtype=np.uint8),
        "is_duplicate": np.asarray([row["is_duplicate"] for row in segment_rows], dtype=np.bool_),
        "channel_map": np.asarray([[int(channel) for channel in row["channel_map"].split(",")] for row in segment_rows], dtype=np.uint8),
    }
    manifest = {
        "input_session": str(session_dir.resolve()),
        "prompts_file": str(prompts_path.resolve()),
        "sample_rate_hz": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "window_duration_s": WINDOW_SAMPLES / SAMPLE_RATE,
        "signal_values": "Raw EXG values; movement localization uses a 0.5–12 Hz bandpass and a 75-sample smoothed envelope",
        "peak_detection": {"minimum_distance_samples": 525, "minimum_prominence": 1.5, "first_usable_center_row": 376},
        "extra_action_policy": "Keep the earliest 20 movements",
        "missing_action_policy": "If 19 movements are detected, copy movement 10 as movement 20 and record its source",
        "prompt_count": len(prompts),
        "recording_count": len(recording_rows),
        "excluded_short_count": len(excluded_rows),
        "segment_count": len(segment_rows),
        "original_segment_count": sum(not row["is_duplicate"] for row in segment_rows),
        "duplicate_segment_count": sum(row["is_duplicate"] for row in segment_rows),
        "recorded_distances_mm": distances,
        "channel_map_recording_counts": channel_map_counts,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "dataset.npz", **dataset)
    write_csv(output_dir / "segments.csv", segment_rows, list(segment_rows[0]))
    write_csv(output_dir / "recordings.csv", recording_rows, list(recording_rows[0]))
    write_csv(output_dir / "excluded.csv", excluded_rows, list(excluded_rows[0]))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment four-direction recordings from the 170 mm circumference wristband")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.session_dir, args.prompts, args.output_dir)
    print(json.dumps({key: manifest[key] for key in ["recording_count", "excluded_short_count", "segment_count", "original_segment_count", "duplicate_segment_count", "channel_map_recording_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
