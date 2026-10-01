"""V8 multiscale RMS/mean and non-wrapping lag features; v7 C4/KRR core reused."""

from __future__ import annotations

from typing import Any

import numpy as np

from src import c170_c4_kernel_ridge_v7 as v7

PREPROCESSORS = v7.PREPROCESSORS
SPECTRAL_BANDS_HZ = v7.SPECTRAL_BANDS_HZ
KERNEL_MODES = v7.KERNEL_MODES
MIN_BASE_SIGMA = v7.MIN_BASE_SIGMA
EPSILON = v7.EPSILON
CHANNEL_COUNT = v7.CHANNEL_COUNT
WINDOW_LENGTH = v7.WINDOW_LENGTH
FAMILY_DIMS = {"e": 62, "f": 62, "h": 84, "c": 5, "d": 1}
VIEW_FAMILIES = {
    "ecd": ("e", "c", "d"),
    "efcd": ("e", "f", "c", "d"),
    "ehcd": ("e", "h", "c", "d"),
}
VIEWS = tuple(VIEW_FAMILIES)
LEVELS = (2, 4, 8, 16, 32)
LAGS = (1, 4, 16, 64)
H_LEVELS = (1, 4, 16)
MIN_FEATURE_STD = v7.MIN_FEATURE_STD

preprocess_windows = v7.preprocess_windows
estimate_sigma_base = v7.estimate_sigma_base
weighted_krr_fit_many = v7.weighted_krr_fit_many
safe_npz_save = v7.safe_npz_save
safe_npz_load = v7.safe_npz_load


def build_feature_families(windows: np.ndarray, epsilon: float = EPSILON) -> dict[str, np.ndarray]:
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (WINDOW_LENGTH, CHANNEL_COUNT):
        raise ValueError(f"expected windows shaped (N,{WINDOW_LENGTH},{CHANNEL_COUNT}), got {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("windows must be finite and epsilon must be finite positive")
    n = len(values)
    rms = np.sqrt(np.mean(np.square(values), axis=1))
    safe_rms = np.maximum(rms, epsilon)

    e_parts: list[np.ndarray] = []
    f_parts: list[np.ndarray] = []
    for count in LEVELS:
        chunks = np.array_split(values, count, axis=1)
        block_rms = np.stack([np.sqrt(np.mean(np.square(chunk), axis=1)) for chunk in chunks], axis=-1)
        block_means = np.stack([np.mean(chunk, axis=1) for chunk in chunks], axis=-1)
        ratios = np.divide(block_rms, safe_rms[..., None], out=np.zeros_like(block_rms), where=rms[..., None] > epsilon)
        e_parts.append(np.log1p(ratios))
        f_parts.append(np.divide(block_means, safe_rms[..., None], out=np.zeros_like(block_means), where=rms[..., None] > epsilon))
    e = np.concatenate(e_parts, axis=-1)
    f = np.concatenate(f_parts, axis=-1)

    h_parts: list[np.ndarray] = []
    denominator = np.square(rms) + epsilon
    for lag in LAGS:
        products = np.zeros_like(values)
        products[:, lag:, :] = values[:, lag:, :] * values[:, :-lag, :]
        normalized = np.divide(products, denominator[:, None, :], out=np.zeros_like(products), where=denominator[:, None, :] > 0)
        lag_parts = []
        for count in H_LEVELS:
            chunks = np.array_split(normalized, count, axis=1)
            lag_parts.append(np.stack([np.mean(chunk, axis=1) for chunk in chunks], axis=-1))
        h_parts.append(np.concatenate(lag_parts, axis=-1))
    h = np.arcsinh(np.concatenate(h_parts, axis=-1))

    base = v7.build_feature_families(values, epsilon=epsilon)
    result = {"e": e, "f": f, "h": h, "c": base["c"], "d": base["d"]}
    for name, family in result.items():
        if family.shape != (n, CHANNEL_COUNT, FAMILY_DIMS[name]) or not np.isfinite(family).all():
            raise ValueError(f"feature family {name} has invalid shape or non-finite values: {family.shape}")
    return result


def assemble_view(families: dict[str, np.ndarray], view: str) -> np.ndarray:
    if view not in VIEW_FAMILIES:
        raise ValueError(f"unknown v8 view {view!r}")
    selected = VIEW_FAMILIES[view]
    arrays = [np.asarray(families[name], dtype=np.float64) for name in selected]
    n = arrays[0].shape[0]
    if any(a.ndim != 3 or a.shape[:2] != (n, CHANNEL_COUNT) for a in arrays):
        raise ValueError("v8 feature families must share [N,4] leading dimensions")
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
        raise ValueError("v8 features do not match fitted shared scaler")
    if not np.isfinite(features).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("non-finite values passed to v8 scaler")
    normalized = (features - mean[None, None, :]) / std[None, None, :]
    for name, (start, end) in scaler["family_slices"].items():
        normalized[:, :, start:end] *= float(scaler["family_scales"][name])
    if not np.isfinite(normalized).all():
        raise ValueError("v8 shared scaler produced non-finite descriptors")
    return normalized.reshape(len(features), -1)
