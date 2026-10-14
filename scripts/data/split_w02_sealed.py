"""Seal three W02 Part B wears as an untouched holdout and write separate development and sealed datasets.

One Part B wear is drawn from each 120-degree sector of the measured band angle ([0,120), [120,240), [240,360)) with
numpy default_rng(SEED). Every stroke and REST window of a sealed wear goes to the sealed dataset only; all other
wears go to the development dataset. Both are written with the same keys as prepare_w02_session.py. The output
directories must not exist. Decision of 2026-10-15, made before any W02 stroke was scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 20261015
SECTORS = ((0, 120), (120, 240), (240, 360))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_sealed(wears: dict[int, int]) -> list[int]:
    """wears: Part B wear -> measured angle in degrees."""
    rng = np.random.default_rng(SEED)
    sealed = []
    for low, high in SECTORS:
        candidates = sorted(w for w, angle in wears.items() if low <= angle % 360 < high)
        sealed.append(int(rng.choice(candidates)))
    return sorted(sealed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dev-dir", type=Path, required=True)
    parser.add_argument("--sealed-dir", type=Path, required=True)
    args = parser.parse_args()
    for directory in (args.dev_dir, args.sealed_dir):
        if directory.exists():
            raise FileExistsError(directory)
    with np.load(args.dataset) as archive:
        arrays = {key: archive[key] for key in archive.files}
    part_b = arrays["part"] == "B"
    wears = {int(w): int(a) for w, a in zip(arrays["wear"][part_b], arrays["actual_angle_deg"][part_b])}
    sealed = draw_sealed(wears)
    stroke_sealed = np.isin(arrays["wear"], sealed)
    rest_sealed = np.isin(arrays["rest_wear"], sealed)
    outputs = {}
    for directory, keep_stroke, keep_rest in ((args.dev_dir, ~stroke_sealed, ~rest_sealed),
                                              (args.sealed_dir, stroke_sealed, rest_sealed)):
        directory.mkdir(parents=True)
        subset = {key: value[keep_rest] if key.startswith("rest_") else value[keep_stroke]
                  for key, value in arrays.items()}
        np.savez_compressed(directory / "dataset.npz", **subset)
        outputs[str(directory)] = {"dataset_sha256": sha256(directory / "dataset.npz"),
                                   "n_strokes": int(keep_stroke.sum()), "n_rest_windows": int(keep_rest.sum()),
                                   "wears": sorted(set(arrays["wear"][keep_stroke].tolist()))}
    manifest = {"source_dataset": str(args.dataset), "source_dataset_sha256": sha256(args.dataset), "seed": SEED,
                "sectors_deg": SECTORS, "part_b_wear_angles": wears, "sealed_wears": sealed,
                "source_sha256": sha256(Path(__file__)), "outputs": outputs}
    for directory in (args.dev_dir, args.sealed_dir):
        (directory / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
