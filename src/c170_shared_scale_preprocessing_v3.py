"""C170 v3 preprocessing with one train-derived scale shared by all channels."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import run_c170_experiment as c170


SAMPLE_RATE_HZ = 250
CHANNEL_COUNT = 4


def calibrate_windows_shared_scale_v3(
    samples: np.ndarray,
    scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """Demean each window/channel and normalize with one scalar.

    With ``scale=None``, fit the scalar over every sample, time point, and
    channel in the passed training array. Validation callers must pass that
    train-derived scalar explicitly.
    """
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != CHANNEL_COUNT:
        raise ValueError(f"expected (batch, time, {CHANNEL_COUNT}), got {values.shape}")
    if not values.shape[0] or not values.shape[1]:
        raise ValueError("calibration requires non-empty batch and time dimensions")
    if not np.isfinite(values).all():
        raise ValueError("calibration input contains non-finite values")

    centered = values - np.mean(values, axis=1, keepdims=True)
    if scale is None:
        fitted_scale = max(
            float(np.median(np.abs(centered))),
            float(np.finfo(np.float32).eps),
        )
    else:
        candidate = np.asarray(scale)
        if candidate.ndim != 0:
            raise ValueError("shared scale must be one scalar for all four channels")
        fitted_scale = float(candidate)
        if not np.isfinite(fitted_scale) or fitted_scale <= 0.0:
            raise ValueError("shared scale must be finite and positive")

    normalized = (centered / np.float32(fitted_scale)).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized calibration output contains non-finite values")
    return normalized, fitted_scale


def prepare_model_inputs_shared_scale_v3(
    windows: np.ndarray,
    splits: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], float]:
    """Filter and normalize train/validation windows using train-only scale.

    Only train and validation indices are indexed into ``windows``. This
    helper does not preprocess the registered test split.
    """
    values = np.asarray(windows)
    if values.ndim != 3 or values.shape[1] != 500 or values.shape[2] != CHANNEL_COUNT:
        raise ValueError(f"expected (N, 500, {CHANNEL_COUNT}) C170 windows, got {values.shape}")
    if "train" not in splits or "val" not in splits:
        raise ValueError("splits must contain train and val indices")

    train_indices = np.asarray(splits["train"])
    val_indices = np.asarray(splits["val"])
    if train_indices.ndim != 1 or val_indices.ndim != 1:
        raise ValueError("train and validation indices must be one-dimensional")
    if train_indices.dtype.kind not in "iu" or val_indices.dtype.kind not in "iu":
        raise ValueError("train and validation indices must be integers")
    train_indices = train_indices.astype(np.int64, copy=False)
    val_indices = val_indices.astype(np.int64, copy=False)
    if not len(train_indices) or not len(val_indices):
        raise ValueError("train and validation splits must be non-empty")
    if (
        np.any(train_indices < 0)
        or np.any(val_indices < 0)
        or np.any(train_indices >= len(values))
        or np.any(val_indices >= len(values))
    ):
        raise ValueError("train or validation index is outside the dataset")
    if len(np.unique(train_indices)) != len(train_indices) or len(np.unique(val_indices)) != len(val_indices):
        raise ValueError("train and validation indices must not contain duplicates")
    if np.intersect1d(train_indices, val_indices).size:
        raise ValueError("train and validation indices must be disjoint")

    indices = np.concatenate((train_indices, val_indices))
    filtered = c170.filter_windows(values[indices], sample_rate=SAMPLE_RATE_HZ)
    train_filtered = filtered[: len(train_indices)]
    val_filtered = filtered[len(train_indices) :]
    train_normalized, shared_scale = calibrate_windows_shared_scale_v3(train_filtered)
    val_normalized, _ = calibrate_windows_shared_scale_v3(val_filtered, scale=shared_scale)
    return {"train": train_normalized, "val": val_normalized}, shared_scale
