"""V9 adds C4-equivariant cross-sensor lag products to the v8 KRR features."""

from __future__ import annotations

from typing import Any

import numpy as np

from src import c170_c4_kernel_ridge_v7 as kernel_core
from src import c170_c4_kernel_ridge_v8 as v8

PREPROCESSORS = v8.PREPROCESSORS
SPECTRAL_BANDS_HZ = v8.SPECTRAL_BANDS_HZ
KERNEL_MODES = v8.KERNEL_MODES
MIN_BASE_SIGMA = v8.MIN_BASE_SIGMA
EPSILON = v8.EPSILON
CHANNEL_COUNT = v8.CHANNEL_COUNT
WINDOW_LENGTH = v8.WINDOW_LENGTH
FAMILY_DIMS = {"q": 252, "e": 62, "f": 62, "c": 5, "d": 1}
VIEW_FAMILIES = {
    "qcd": ("q", "c", "d"),
    "qecd": ("q", "e", "c", "d"),
    "qefcd": ("q", "e", "f", "c", "d"),
}
VIEWS = tuple(VIEW_FAMILIES)
OFFSETS = (1, 2, 3)
LAGS = (0, 4, 16, 64)
SEGMENT_COUNTS = (1, 4, 16)
MIN_FEATURE_STD = v8.MIN_FEATURE_STD

preprocess_windows = v8.preprocess_windows
estimate_sigma_base = kernel_core.estimate_sigma_base
weighted_krr_fit_many = kernel_core.weighted_krr_fit_many
safe_npz_save = kernel_core.safe_npz_save
safe_npz_load = kernel_core.safe_npz_load


def build_feature_families(windows: np.ndarray, epsilon: float = EPSILON) -> dict[str, np.ndarray]:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (WINDOW_LENGTH, CHANNEL_COUNT):
        raise ValueError(f"expected windows shaped (N,{WINDOW_LENGTH},{CHANNEL_COUNT}), got {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("windows must be finite and epsilon must be finite positive")

    inherited = v8.build_feature_families(values, epsilon=epsilon)
    rms = np.sqrt(np.mean(np.square(values), axis=1))
    safe_rms = np.maximum(rms, epsilon)
    q_parts: list[np.ndarray] = []
    for offset in OFFSETS:
        neighbor = np.roll(values, shift=-offset, axis=2)
        neighbor_rms = np.roll(safe_rms, shift=-offset, axis=1)
        denominator = safe_rms * neighbor_rms
        for lag in LAGS:
            products = np.zeros_like(values)
            if lag == 0:
                products[:] = values * neighbor
            else:
                products[:, lag:, :] = values[:, lag:, :] * neighbor[:, :-lag, :]
            normalized = np.divide(products, denominator[:, None, :], out=np.zeros_like(products), where=denominator[:, None, :] > 0)
            for segment_count in SEGMENT_COUNTS:
                chunks = np.array_split(normalized, segment_count, axis=1)
                q_parts.append(np.stack([np.mean(chunk, axis=1) for chunk in chunks], axis=-1))
    q = np.arcsinh(np.concatenate(q_parts, axis=-1))
    result = {"q": q, "e": inherited["e"], "f": inherited["f"], "c": inherited["c"], "d": inherited["d"]}
    for name, family in result.items():
        if family.shape != (len(values), CHANNEL_COUNT, FAMILY_DIMS[name]) or not np.isfinite(family).all():
            raise ValueError(f"feature family {name} has invalid shape or non-finite values: {family.shape}")
    return result


def assemble_view(families: dict[str, np.ndarray], view: str) -> np.ndarray:
    if view not in VIEW_FAMILIES:
        raise ValueError(f"unknown v9 view {view!r}")
    selected = VIEW_FAMILIES[view]
    arrays = [np.asarray(families[name], dtype=np.float64) for name in selected]
    n = arrays[0].shape[0]
    if any(array.ndim != 3 or array.shape[:2] != (n, CHANNEL_COUNT) for array in arrays):
        raise ValueError("v9 feature families must share [N,4] leading dimensions")
    return np.concatenate(arrays, axis=-1)


def fit_fold_scaler(sensor_features: np.ndarray, view: str) -> dict[str, Any]:
    features = np.asarray(sensor_features, dtype=np.float64)
    if view not in VIEW_FAMILIES or features.ndim != 3 or features.shape[1] != CHANNEL_COUNT or not len(features):
        raise ValueError("sensor_features must be non-empty [N,4,D] for a known view")
    if not np.isfinite(features).all():
        raise ValueError("fold-training features contain non-finite values")
    family_names = VIEW_FAMILIES[view]
    expected_dim = sum(FAMILY_DIMS[name] for name in family_names)
    if features.shape[-1] != expected_dim:
        raise ValueError(f"view {view} expects {expected_dim} features, got {features.shape[-1]}")
    shared = features.reshape(-1, expected_dim)
    mean = shared.mean(axis=0)
    raw_std = shared.std(axis=0, ddof=0)
    std = np.maximum(raw_std, MIN_FEATURE_STD)
    slices: dict[str, tuple[int, int]] = {}
    scales: dict[str, float] = {}
    offset = 0
    for name in family_names:
        end = offset + FAMILY_DIMS[name]
        slices[name] = (offset, end)
        scales[name] = 1.0 / np.sqrt(FAMILY_DIMS[name] * len(family_names))
        offset = end
    return {"view": view, "mean": mean, "std": std, "raw_std": raw_std,
            "family_slices": slices, "family_scales": scales,
            "training_sensor_rows": int(shared.shape[0]), "std_floor": MIN_FEATURE_STD}


def apply_fold_scaler(sensor_features: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    features = np.asarray(sensor_features, dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    if features.ndim != 3 or features.shape[1] != CHANNEL_COUNT or features.shape[-1] != len(mean):
        raise ValueError("v9 features do not match fitted shared scaler")
    if not np.isfinite(features).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("non-finite values passed to v9 scaler")
    normalized = (features - mean[None, None, :]) / std[None, None, :]
    for name, (start, end) in scaler["family_slices"].items():
        normalized[:, :, start:end] *= float(scaler["family_scales"][name])
    if not np.isfinite(normalized).all():
        raise ValueError("v9 shared scaler produced non-finite descriptors")
    return normalized.reshape(len(features), -1)
