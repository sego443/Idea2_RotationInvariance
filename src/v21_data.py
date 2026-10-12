"""Unified loaders for v21: C170 positions and W01 angles as wear groups with block/repetition metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
C170_PATH = PROJECT_ROOT / "data" / "processed_c170_20260926" / "dataset.npz"
W01_PATH = PROJECT_ROOT / "outputs" / "round1_rotation_poc_2026-08-27" / "segmented_dataset.npz"
CLASSES = ("UP", "DOWN", "LEFT", "RIGHT")
C170_F2_TRAIN_MM = (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)
C170_OLD_VALIDATION_MM = (5, 40, 90, 130)
C170_EXPOSED_TEST_MM = (15, 25, 110, 150)
EXISTING_CAL_SHOTS = 2  # C170/W01: the first repetitions of each direction recording act as calibration strokes
C170_CIRCUMFERENCE_MM = 170


def load_c170(path: Path = C170_PATH) -> dict[str, np.ndarray]:
    """All C170 windows except the marked duplicate. Group = position; block = one direction recording."""
    with np.load(path) as archive:
        keep = ~archive["is_duplicate"]
        positions = archive["distance_mm"][keep].astype(int)
        return {
            "X": archive["X_raw"][keep].astype(np.float32),
            "y": np.array([CLASSES.index(value) for value in archive["direction"][keep]]),
            "group": np.array([f"C170_P{p:03d}" for p in positions]),
            "position": positions,
            "block": archive["block"][keep].astype(int),
            "repetition": archive["repetition"][keep].astype(int),
            "sample_index": np.flatnonzero(keep),
            "source": np.array(["C170"] * int(keep.sum())),
            "is_calibration": archive["repetition"][keep].astype(int) <= EXISTING_CAL_SHOTS,
            "part": np.array([""] * int(keep.sum())),
            "wear_group": np.array([f"C170_P{p:03d}" for p in positions]),
        }


def load_w01(path: Path = W01_PATH) -> dict[str, np.ndarray]:
    """All 240 W01 strokes. Group = nominal angle; block = one direction recording (labels share C170 order)."""
    with np.load(path) as archive:
        angles = archive["angles"].astype(int)
        return {
            "X": archive["strokes"].astype(np.float32),
            "y": archive["labels"].astype(int),
            "group": np.array([f"W01_A{a:03d}" for a in angles]),
            "position": angles,
            "block": archive["blocks"].astype(int) + 10_000,
            "repetition": archive["repetitions"].astype(int),
            "sample_index": np.arange(len(angles)),
            "source": np.array(["W01"] * len(angles)),
            "is_calibration": archive["repetitions"].astype(int) <= EXISTING_CAL_SHOTS,
            "part": np.array([""] * len(angles)),
            "wear_group": np.array([f"W01_A{a:03d}" for a in angles]),
        }


def concatenate(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in data.items()}


def block_permuted_labels(data: dict[str, np.ndarray], seed: int) -> np.ndarray:
    """Matched null: within every group, permute the direction labels of complete recording blocks."""
    rng = np.random.default_rng(seed)
    labels = data["y"].copy()
    for group in np.unique(data["group"]):
        mask = data["group"] == group
        labels[mask] = rng.permutation(len(CLASSES))[data["y"][mask]]
    return labels


def load_w02(path: Path) -> dict[str, np.ndarray]:
    """W02 strokes from prepare_w02_session.py.

    Group = one trial recording (its labeled C prefix is the calibration support); wear_group = the wear.
    repetition = order within the wear.
    """
    with np.load(path) as archive:
        wear = archive["wear"].astype(int)
        recording = archive["recording"].astype(str)
        participant = archive["participant"].astype(str)
        part = archive["part"].astype(str)
        trial = np.array([int(r[1:]) for r in recording])
        return {
            "X": archive["X_raw"].astype(np.float32),
            "y": archive["label"].astype(int),
            "group": np.array([f"W02_{p}_{q}_W{w:02d}_{r}" for p, q, w, r in zip(participant, part, wear, recording)]),
            "wear_group": np.array([f"W02_{p}_{q}_W{w:02d}" for p, q, w in zip(participant, part, wear)]),
            "position": archive["actual_angle_deg"].astype(int),
            "block": wear * 10 + trial + 20_000,
            "repetition": (trial * 1000 + archive["position_in_recording"]).astype(int),
            "sample_index": np.arange(len(wear)),
            "source": np.array(["W02"] * len(wear)),
            "is_calibration": archive["is_prefix"].astype(bool),
            "part": part,
        }


def circular_distance(source: str, a: int, b: int) -> int:
    period = C170_CIRCUMFERENCE_MM if source == "C170" else 360
    distance = abs(int(a) - int(b)) % period
    return min(distance, period - distance)


def nearest_groups(data: dict[str, np.ndarray]) -> dict[str, list[str]]:
    """For C170/W01 groups, other groups of the same source ordered by circular position distance."""
    table = {}
    groups = {g: (data["source"][i], data["position"][i]) for g, i in zip(*np.unique(data["group"], return_index=True))}
    for group, (source, position) in groups.items():
        others = [g for g, (s, _) in groups.items() if s == source and g != group]
        table[group] = sorted(others, key=lambda g: (circular_distance(source, position, groups[g][1]), g))
    return table
