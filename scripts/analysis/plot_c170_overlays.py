from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
import numpy as np
from scipy import signal


SAMPLE_RATE = 250
SEED = 20260927
FONT_PATH = Path("/Library/Fonts/Arial Unicode.ttf")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_traces(record: dict[str, str], segments: list[dict[str, str]], dataset: dict[str, np.ndarray]) -> np.ndarray:
    block = int(record["block"])
    rows = sorted(
        (row for row in segments if int(row["block"]) == block),
        key=lambda row: int(row["repetition"]),
    )
    if len(rows) != 20 or [int(row["repetition"]) for row in rows] != list(range(1, 21)):
        raise ValueError(f"B{block:03d} does not contain 20 ordered movement windows")
    channels = [int(value) for value in record["channel_map"].split(",")]
    if any(row["source_filename"] != record["source_filename"] or row["channel_map"] != record["channel_map"] for row in rows):
        raise ValueError(f"B{block:03d} has inconsistent movement source recordings")
    indices = [int(row["dataset_index"]) for row in rows]
    if not np.all(dataset["block"][indices] == block) or not np.all(dataset["channel_map"][indices] == channels):
        raise ValueError(f"B{block:03d} has inconsistent data labels")
    raw = dataset["X_raw"][indices]
    if raw.shape != (20, 500, 4) or not np.isfinite(raw).all():
        raise ValueError(f"B{block:03d} has an invalid movement tensor shape: {raw.shape}")
    sos = signal.butter(3, [0.5, 40], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    filtered = signal.sosfiltfilt(sos, raw, axis=1)
    baseline = np.median(filtered[:, :50, :], axis=1, keepdims=True)
    return filtered - baseline


def save_plot(record: dict[str, str], traces: np.ndarray, output: Path) -> None:
    block = int(record["block"])
    distance = int(record["distance_mm"])
    direction = record["direction"]
    channels = [int(value) for value in record["channel_map"].split(",")]
    time = np.arange(500) / SAMPLE_RATE - 1.0
    palette = plt.get_cmap("plasma")
    normalization = colors.Normalize(vmin=1, vmax=20)
    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    figure.subplots_adjust(left=0.12, right=0.86, top=0.91, bottom=0.13, hspace=0.18)
    for channel_index, axis in enumerate(axes):
        for repetition in range(20):
            axis.plot(
                time,
                traces[repetition, :, channel_index],
                color=palette(normalization(repetition + 1)),
                alpha=0.58,
                linewidth=0.85,
            )
        axis.axvline(0, color="#303030", linestyle="--", linewidth=0.8, alpha=0.65)
        axis.set_ylabel(f"EXG Channel {channels[channel_index]}")
        axis.grid(alpha=0.16)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[-1].set_xlim(-1, 1)
    axes[-1].set_xlabel("Time relative to movement center (s)")
    figure.suptitle(f"B{block:03d} · {distance} mm · {direction} · 20 overlaid movements", fontsize=17)
    colorbar_axis = figure.add_axes([0.9, 0.22, 0.018, 0.61])
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=palette), cax=colorbar_axis)
    colorbar.set_label("Movement index")
    colorbar.set_ticks([1, 5, 10, 15, 20])
    figure.text(0.12, 0.075, record["source_filename"], fontsize=9)
    figure.text(0.12, 0.047, "Each window is bandpass filtered at 0.5–40 Hz and centered by its first 0.2 s median; dashed lines mark detected movement centers.", fontsize=9)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomly select five recordings and plot overlaid movements")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["sample_rate_hz"] != SAMPLE_RATE or manifest["window_samples"] != 500:
        raise ValueError("Sample rate or movement-window length does not match the plot settings")
    recordings = read_csv(args.dataset_dir / "recordings.csv")
    segments = read_csv(args.dataset_dir / "segments.csv")
    if len(recordings) != 80:
        raise ValueError(f"Expected 80 complete recordings; found {len(recordings)}")
    selected_indices = np.random.default_rng(SEED).choice(len(recordings), size=5, replace=False)
    with np.load(args.dataset_dir / "dataset.npz", allow_pickle=False) as loaded:
        dataset = {key: loaded[key] for key in ["X_raw", "block", "channel_map"]}
    selected = [(recordings[index], build_traces(recordings[index], segments, dataset)) for index in selected_indices]

    font_manager.fontManager.addfont(str(FONT_PATH))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selection: list[dict[str, str | int]] = []
    for record, traces in selected:
        name = (
            f"B{int(record['block']):03d}_D{int(record['distance_mm']):03d}MM_"
            f"{record['direction']}_R20_overlay.png"
        )
        save_plot(record, traces, args.output_dir / name)
        selection.append({
            "block": int(record["block"]),
            "distance_mm": int(record["distance_mm"]),
            "direction": record["direction"],
            "source_filename": record["source_filename"],
            "channel_map": record["channel_map"],
            "image_filename": name,
        })
    (args.output_dir / "selection.json").write_text(
        json.dumps({"random_seed": SEED, "recordings": selection}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
