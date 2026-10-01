"""Fixed and learned per-sensor scale controls for the C170 v14 experiment."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import run_c170_experiment as c170


NORMALIZATIONS = ("sensor_median", "sensor_rms", "window_sensor_rms", "bounded_sensor_gain")
SCALE_FLOOR = 1e-6
GAIN_BOUND = 0.1
REGULARIZATION_WEIGHT = 1e-3
LEGACY_SHARED_SCALE_SENTINEL = 11123.037109375


def prepare_train_val(
    kind: str, windows: np.ndarray, splits: Mapping[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Filter only train and validation windows, then fit/apply the fixed protocol."""
    values = np.asarray(windows)
    train_idx = np.asarray(splits["train"], dtype=np.int64)
    val_idx = np.asarray(splits["val"], dtype=np.int64)
    if kind not in NORMALIZATIONS or kind == "bounded_sensor_gain":
        raise ValueError(f"preprocessing is not defined for {kind}")
    if values.ndim != 3 or values.shape[1:] != (500, 4):
        raise ValueError(f"expected [N,500,4], got {values.shape}")
    if not len(train_idx) or not len(val_idx) or np.intersect1d(train_idx, val_idx).size:
        raise ValueError("train/validation indices are empty or overlap")

    selected = np.concatenate((train_idx, val_idx))
    filtered = np.asarray(c170.filter_windows(values[selected], sample_rate=250), dtype=np.float32)
    centered = filtered - np.mean(filtered, axis=1, keepdims=True, dtype=np.float32)
    train = centered[: len(train_idx)]
    val = centered[len(train_idx) :]
    if kind == "sensor_median":
        scales = np.maximum(np.median(np.abs(train), axis=(0, 1)), SCALE_FLOOR).astype(np.float32)
        train_out = train / scales.reshape(1, 1, 4)
        val_out = val / scales.reshape(1, 1, 4)
        fit = "train 879 x time pooled per physical sensor; median absolute value"
    elif kind == "sensor_rms":
        scales = np.maximum(np.sqrt(np.mean(np.square(train, dtype=np.float32), axis=(0, 1))), SCALE_FLOOR).astype(np.float32)
        train_out = train / scales.reshape(1, 1, 4)
        val_out = val / scales.reshape(1, 1, 4)
        fit = "train 879 x time pooled per physical sensor; root mean square"
    else:
        scales_train = np.maximum(np.sqrt(np.mean(np.square(train, dtype=np.float32), axis=1, keepdims=True)), SCALE_FLOOR)
        scales_val = np.maximum(np.sqrt(np.mean(np.square(val, dtype=np.float32), axis=1, keepdims=True)), SCALE_FLOOR)
        train_out = train / scales_train
        val_out = val / scales_val
        scales = None
        fit = "per window and physical sensor temporal RMS; sample-adaptive amplitude normalization"

    train_out = np.asarray(train_out, dtype=np.float32)
    val_out = np.asarray(val_out, dtype=np.float32)
    if not np.isfinite(train_out).all() or not np.isfinite(val_out).all():
        raise ValueError("sensor-normalized train/validation inputs are non-finite")
    info: dict[str, object] = {
        "sensor_normalization": kind,
        "sensor_order": ["S0", "S1", "S2", "S3"],
        "fit_scope": fit,
        "scale_floor": SCALE_FLOOR,
        "fitted_sensor_scales": None if scales is None else [float(value) for value in scales],
        "normalization_fit_splits": ["train"] if scales is not None else [],
        "processed_splits": ["train", "validation"],
        "test_preprocessed": False,
        "interpretation": "normalization statistics may reflect both hardware and label/position signal differences; they are not isolated hardware calibration parameters",
    }
    return {"train": train_out, "val": val_out}, info
