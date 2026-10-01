"""Small deterministic feature and C4/RBF kernel-ridge primitives for C170 v7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

import run_c170_experiment as c170


SAMPLE_RATE_HZ = 250.0
CHANNEL_COUNT = 4
WINDOW_LENGTH = 500
PREPROCESSORS = ("bandpass_0p5_40", "demean_only", "lowpass_40")
VIEWS = ("ab", "bcd", "abcd")
FAMILIES = ("a", "b", "c", "d")
FAMILY_DIMS = {"a": 64, "b": 8, "c": 5, "d": 1}
VIEW_FAMILIES = {"ab": ("a", "b"), "bcd": ("b", "c", "d"), "abcd": FAMILIES}
SPECTRAL_BANDS_HZ = ((0.5, 4.0), (4.0, 8.0), (8.0, 15.0), (15.0, 40.0), (40.0, 125.0))
EPSILON = 1e-12
MIN_SIGMA = 1e-12
MIN_BASE_SIGMA = 1e-6
MIN_FEATURE_STD = 1e-12
KERNEL_MODES = ("c4_group", "plain_rbf")


def preprocess_windows(windows: np.ndarray, mode: str) -> np.ndarray:
    """Filter with the input dtype where supported, then demean each window/channel."""
    values = np.asarray(windows)
    if values.ndim != 3 or values.shape[1:] != (WINDOW_LENGTH, CHANNEL_COUNT):
        raise ValueError(f"expected windows shaped (N,{WINDOW_LENGTH},{CHANNEL_COUNT}), got {values.shape}")
    if values.dtype.kind != "f":
        raise TypeError("raw windows must be floating point")
    if not np.isfinite(values).all():
        raise ValueError("raw windows contain non-finite values")
    if mode not in PREPROCESSORS:
        raise ValueError(f"unknown preprocessor {mode!r}; expected one of {PREPROCESSORS}")

    if mode == "bandpass_0p5_40":
        filtered = c170.filter_windows(values.copy(), sample_rate=SAMPLE_RATE_HZ)
    elif mode == "lowpass_40":
        sos = signal.butter(3, 40.0, btype="lowpass", fs=SAMPLE_RATE_HZ, output="sos")
        filtered = signal.sosfiltfilt(sos, values.copy(), axis=1)
    else:
        filtered = values.copy()
    centered = np.asarray(filtered) - np.mean(filtered, axis=1, keepdims=True)
    if not np.isfinite(centered).all():
        raise ValueError(f"{mode} preprocessing produced non-finite values")
    return centered


def build_feature_families(windows: np.ndarray, epsilon: float = EPSILON) -> dict[str, np.ndarray]:
    """Extract a/b/c/d features as arrays with shape [N,4,family_dimension]."""
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (WINDOW_LENGTH, CHANNEL_COUNT):
        raise ValueError(f"expected windows shaped (N,{WINDOW_LENGTH},{CHANNEL_COUNT}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("preprocessed windows contain non-finite values")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")

    rms = np.sqrt(np.mean(np.square(values), axis=1))  # [N,4]
    safe_rms = np.maximum(rms, epsilon)

    # Family a: RMS-normalized waveform resampled from 500 to 64 points.
    normalized = np.divide(
        values,
        safe_rms[:, None, :],
        out=np.zeros_like(values),
        where=(rms[:, None, :] > epsilon),
    )
    a_time_major = signal.resample_poly(normalized, up=16, down=125, axis=1)
    a = np.transpose(a_time_major, (0, 2, 1)).copy()
    if a.shape[-1] != FAMILY_DIMS["a"]:
        raise RuntimeError(f"resample_poly returned {a.shape[-1]} points; expected 64")

    # Family b: eight contiguous blocks; linspace boundaries handle 500 % 8 != 0.
    edges = np.linspace(0, WINDOW_LENGTH, 9, dtype=np.int64)
    block_values = []
    for block_index in range(8):
        start, end = int(edges[block_index]), int(edges[block_index + 1])
        block_rms = np.sqrt(np.mean(np.square(values[:, start:end, :]), axis=1))
        block_values.append(np.divide(block_rms, safe_rms, out=np.zeros_like(block_rms), where=rms > epsilon))
    b = np.stack(block_values, axis=-1)

    # Family c: Hann periodogram relative band powers. The last band includes 125 Hz.
    frequencies, density = signal.periodogram(
        values,
        fs=SAMPLE_RATE_HZ,
        window="hann",
        detrend=False,
        return_onesided=True,
        scaling="density",
        axis=1,
    )
    frequency_step = SAMPLE_RATE_HZ / WINDOW_LENGTH
    # The five declared bands begin at 0.5 Hz, so the relative-power
    # denominator uses the same 0.5–125 Hz range and excludes the DC bin.
    denominator_mask = frequencies >= 0.5
    total_power = np.sum(density[:, denominator_mask, :], axis=1) * frequency_step
    safe_total_power = np.maximum(total_power, epsilon)
    relative_powers = []
    for band_index, (low_hz, high_hz) in enumerate(SPECTRAL_BANDS_HZ):
        include_upper = band_index == len(SPECTRAL_BANDS_HZ) - 1
        mask = (frequencies >= low_hz) & (
            (frequencies <= high_hz) if include_upper else (frequencies < high_hz)
        )
        if not np.any(mask):
            raise RuntimeError(f"spectral band {low_hz:g}-{high_hz:g} Hz has no periodogram bins")
        band_power = np.sum(density[:, mask, :], axis=1) * frequency_step
        relative = np.divide(
            band_power,
            safe_total_power,
            out=np.zeros_like(band_power),
            where=total_power > epsilon,
        )
        relative_powers.append(np.log(np.maximum(relative, epsilon)))
    c = np.stack(relative_powers, axis=-1)

    # Family d: log full-window RMS.
    d = np.log(safe_rms)[..., None]
    result = {"a": a, "b": b, "c": c, "d": d}
    for name, family in result.items():
        if family.shape != (len(values), CHANNEL_COUNT, FAMILY_DIMS[name]):
            raise RuntimeError(f"family {name} has unexpected shape {family.shape}")
        if not np.isfinite(family).all():
            raise ValueError(f"family {name} contains non-finite values")
    return result


def assemble_view(families: dict[str, np.ndarray], view: str) -> np.ndarray:
    if view not in VIEW_FAMILIES:
        raise ValueError(f"unknown feature view {view!r}; expected one of {tuple(VIEW_FAMILIES)}")
    selected = VIEW_FAMILIES[view]
    if any(name not in families for name in selected):
        raise ValueError(f"missing a feature family for view {view}")
    arrays = [np.asarray(families[name], dtype=np.float64) for name in selected]
    n = arrays[0].shape[0]
    if any(array.ndim != 3 or array.shape[:2] != (n, CHANNEL_COUNT) for array in arrays):
        raise ValueError("feature families must share [N,4] leading dimensions")
    return np.concatenate(arrays, axis=-1)


def fit_fold_scaler(sensor_features: np.ndarray, view: str) -> dict[str, Any]:
    """Fit coordinate statistics on fold-training rows × all four sensors."""
    features = np.asarray(sensor_features, dtype=np.float64)
    if view not in VIEW_FAMILIES:
        raise ValueError(f"unknown view {view!r}")
    if features.ndim != 3 or features.shape[1] != CHANNEL_COUNT or not len(features):
        raise ValueError("sensor_features must be non-empty [N,4,D]")
    if not np.isfinite(features).all():
        raise ValueError("fold-training features contain non-finite values")
    dimensions = [FAMILY_DIMS[name] for name in VIEW_FAMILIES[view]]
    if sum(dimensions) != features.shape[-1]:
        raise ValueError("view feature dimension does not match its family dimensions")
    shared = features.reshape(-1, features.shape[-1])
    mean = shared.mean(axis=0)
    raw_std = shared.std(axis=0, ddof=0)
    std = np.maximum(raw_std, MIN_FEATURE_STD)
    slices: dict[str, tuple[int, int]] = {}
    family_scales: dict[str, float] = {}
    offset = 0
    family_count = len(VIEW_FAMILIES[view])
    for name in VIEW_FAMILIES[view]:
        end = offset + FAMILY_DIMS[name]
        slices[name] = (offset, end)
        family_scales[name] = 1.0 / np.sqrt(FAMILY_DIMS[name] * family_count)
        offset = end
    return {
        "view": view,
        "mean": mean,
        "std": std,
        "raw_std": raw_std,
        "family_slices": slices,
        "family_scales": family_scales,
        "training_sensor_rows": int(shared.shape[0]),
        "std_floor": MIN_FEATURE_STD,
    }


def apply_fold_scaler(sensor_features: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    features = np.asarray(sensor_features, dtype=np.float64)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    if features.ndim != 3 or features.shape[1] != CHANNEL_COUNT or features.shape[-1] != len(mean):
        raise ValueError("features do not match the fitted shared scaler")
    if not np.isfinite(features).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("non-finite values passed to fold scaler")
    transformed = (features - mean[None, None, :]) / std[None, None, :]
    for family, (start, end) in scaler["family_slices"].items():
        transformed[:, :, start:end] *= float(scaler["family_scales"][family])
    if not np.isfinite(transformed).all():
        raise ValueError("fold normalization produced non-finite descriptors")
    return transformed.reshape(len(features), -1)


def _as_sensor_descriptors(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    if value.ndim == 3 and value.shape[1] == CHANNEL_COUNT:
        result = value
    elif value.ndim == 2 and value.shape[1] % CHANNEL_COUNT == 0:
        result = value.reshape(value.shape[0], CHANNEL_COUNT, value.shape[1] // CHANNEL_COUNT)
    else:
        raise ValueError("descriptors must be [N,4,D] or flattened [N,4D]")
    if not result.is_floating_point() or not torch.isfinite(result).all():
        raise ValueError("kernel descriptors must be finite floating point")
    return result.to(dtype=torch.float64)


def estimate_sigma_base(descriptors: torch.Tensor) -> dict[str, float | bool]:
    """Use sqrt(median upper-triangle squared distance), with a recorded floor."""
    values = _as_sensor_descriptors(descriptors).reshape(len(descriptors), -1)
    if len(values) < 2:
        raise ValueError("at least two training descriptors are required for sigma estimation")
    squared = torch.cdist(values, values, p=2).square()
    upper = torch.triu(torch.ones_like(squared, dtype=torch.bool), diagonal=1)
    pairwise = squared[upper]
    if not torch.isfinite(pairwise).all() or not len(pairwise):
        raise ValueError("pairwise squared distances are empty or non-finite")
    median_squared = float(torch.median(pairwise).detach().cpu())
    if not np.isfinite(median_squared) or median_squared < 0:
        raise ValueError("median pairwise squared distance must be finite and non-negative")
    sigma_unfloored = float(np.sqrt(median_squared))
    floor_applied = sigma_unfloored < MIN_BASE_SIGMA
    return {
        "median_squared_distance": median_squared,
        "sigma_unfloored": sigma_unfloored,
        "sigma_base": max(sigma_unfloored, MIN_BASE_SIGMA),
        "sigma_floor": MIN_BASE_SIGMA,
        "sigma_floor_applied": bool(floor_applied),
    }


def _squared_distances(x: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "plain_rbf":
        left = x.reshape(len(x), -1)
        right = y.reshape(len(y), -1)
        left_sq = left.square().sum(dim=1, keepdim=True)
        right_sq = right.square().sum(dim=1).unsqueeze(0)
        return (left_sq + right_sq - 2.0 * (left @ right.T)).clamp_min(0.0)
    if mode != "c4_group":
        raise ValueError(f"unknown kernel mode {mode!r}; expected one of {KERNEL_MODES}")
    distances = []
    for shift in range(CHANNEL_COUNT):
        shifted = torch.roll(y, shifts=shift, dims=1).reshape(len(y), -1)
        left = x.reshape(len(x), -1)
        left_sq = left.square().sum(dim=1, keepdim=True)
        right_sq = shifted.square().sum(dim=1).unsqueeze(0)
        distances.append((left_sq + right_sq - 2.0 * (left @ shifted.T)).clamp_min(0.0))
    return torch.stack(distances, dim=0)


def _raw_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float, mode: str) -> torch.Tensor:
    if sigma <= MIN_SIGMA or not np.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    squared = _squared_distances(x, y, mode)
    exponent = -squared / (2.0 * float(sigma) * float(sigma))
    values = torch.exp(exponent)
    return values.mean(dim=0) if mode == "c4_group" else values


def kernel_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
    mode: str,
    *,
    x_self_diagonal: torch.Tensor | None = None,
    y_self_diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the diagonally normalized group-average or plain RBF kernel."""
    left = _as_sensor_descriptors(x)
    right = _as_sensor_descriptors(y)
    raw = _raw_kernel(left, right, sigma, mode)
    same = left.shape == right.shape and left.data_ptr() == right.data_ptr()
    if mode == "plain_rbf":
        if same:
            return 0.5 * (raw + raw.T)
        return raw
    if x_self_diagonal is None:
        x_raw = raw if same else _raw_kernel(left, left, sigma, mode)
        x_diag = torch.diagonal(x_raw) if same else torch.diagonal(x_raw)
    else:
        x_diag = torch.as_tensor(x_self_diagonal, dtype=raw.dtype, device=raw.device)
    if y_self_diagonal is None:
        y_raw = raw if same else _raw_kernel(right, right, sigma, mode)
        y_diag = torch.diagonal(y_raw)
    else:
        y_diag = torch.as_tensor(y_self_diagonal, dtype=raw.dtype, device=raw.device)
    denom = torch.sqrt(torch.clamp(x_diag, min=MIN_SIGMA)[:, None] * torch.clamp(y_diag, min=MIN_SIGMA)[None, :])
    normalized = raw / denom
    if same:
        normalized = 0.5 * (normalized + normalized.T)
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized kernel contains non-finite values")
    return normalized


def group_self_diagonal(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute diag(K_G(x,x)) without materializing the full self-kernel."""
    values = _as_sensor_descriptors(x)
    raw_diagonal = torch.zeros(len(values), dtype=values.dtype, device=values.device)
    flattened = values.reshape(len(values), -1)
    norm_sq = flattened.square().sum(dim=1)
    for shift in range(CHANNEL_COUNT):
        rolled = torch.roll(values, shifts=shift, dims=1).reshape(len(values), -1)
        shifted_norm_sq = rolled.square().sum(dim=1)
        dot = torch.sum(flattened * rolled, dim=1)
        squared = (norm_sq + shifted_norm_sq - 2.0 * dot).clamp_min(0.0)
        raw_diagonal += torch.exp(-squared / (2.0 * float(sigma) * float(sigma))) / CHANNEL_COUNT
    return raw_diagonal


def group_kernel_matrix_cached(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
    mode: str,
    *,
    x_self_diagonal: torch.Tensor | None = None,
    y_self_diagonal: torch.Tensor | None = None,
) -> torch.Tensor:
    """Kernel matrix variant using cached group self diagonals for CV batching."""
    if mode == "plain_rbf":
        return kernel_matrix(x, y, sigma, mode)
    left = _as_sensor_descriptors(x)
    right = _as_sensor_descriptors(y)
    raw = _raw_kernel(left, right, sigma, "c4_group")
    same = left.shape == right.shape and left.data_ptr() == right.data_ptr()
    if x_self_diagonal is None:
        x_diag = group_self_diagonal(left, sigma) if not same else group_self_diagonal(left, sigma)
    else:
        x_diag = torch.as_tensor(x_self_diagonal, dtype=raw.dtype, device=raw.device)
    if y_self_diagonal is None:
        y_diag = group_self_diagonal(right, sigma)
    else:
        y_diag = torch.as_tensor(y_self_diagonal, dtype=raw.dtype, device=raw.device)
    normalized = raw / torch.sqrt(torch.clamp(x_diag, min=MIN_SIGMA)[:, None] * torch.clamp(y_diag, min=MIN_SIGMA)[None, :])
    if same:
        normalized = 0.5 * (normalized + normalized.T)
    if not torch.isfinite(normalized).all():
        raise ValueError("normalized kernel contains non-finite values")
    return normalized


def weighted_krr_fit(
    kernel_train: torch.Tensor,
    one_hot_targets: torch.Tensor,
    weights: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, float]:
    """Solve one weighted centered-label KRR system in float64 via Cholesky."""
    kernel = torch.as_tensor(kernel_train, dtype=torch.float64)
    targets = torch.as_tensor(one_hot_targets, dtype=torch.float64, device=kernel.device)
    sample_weights = torch.as_tensor(weights, dtype=torch.float64, device=kernel.device).reshape(-1)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel_train must be square")
    n = kernel.shape[0]
    if targets.ndim != 2 or targets.shape[0] != n or sample_weights.shape != (n,):
        raise ValueError("target/weight dimensions do not match kernel_train")
    if not torch.isfinite(kernel).all() or not torch.isfinite(targets).all() or not torch.isfinite(sample_weights).all():
        raise ValueError("KRR inputs must be finite")
    if torch.any(sample_weights <= 0) or alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("KRR weights and alpha must be positive")
    sqrt_w = torch.sqrt(sample_weights)
    system = sqrt_w[:, None] * kernel * sqrt_w[None, :]
    system = system + float(alpha) * torch.eye(n, dtype=kernel.dtype, device=kernel.device)
    rhs = sqrt_w[:, None] * (targets - 0.25)
    chol = torch.linalg.cholesky(system)
    solution = torch.cholesky_solve(rhs, chol)
    residual = system @ solution - rhs
    residual_ratio = torch.linalg.vector_norm(residual) / torch.clamp(torch.linalg.vector_norm(rhs), min=EPSILON)
    coefficients = sqrt_w[:, None] * solution
    if not torch.isfinite(coefficients).all() or not torch.isfinite(residual_ratio):
        raise ValueError("weighted KRR solve produced non-finite output")
    return coefficients, float(residual_ratio.detach().cpu())


def weighted_krr_fit_many(
    kernel_train: torch.Tensor,
    one_hot_targets: torch.Tensor,
    weights: torch.Tensor,
    alphas: tuple[float, ...] | list[float],
    *,
    max_residual: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch Cholesky solves over alpha values, reusing one weighted kernel."""
    kernel = torch.as_tensor(kernel_train, dtype=torch.float64)
    targets = torch.as_tensor(one_hot_targets, dtype=torch.float64, device=kernel.device)
    sample_weights = torch.as_tensor(weights, dtype=torch.float64, device=kernel.device).reshape(-1)
    alpha_values = torch.as_tensor(alphas, dtype=kernel.dtype, device=kernel.device)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel_train must be square")
    n = kernel.shape[0]
    if targets.ndim != 2 or targets.shape[0] != n or sample_weights.shape != (n,):
        raise ValueError("target/weight dimensions do not match kernel_train")
    if (
        not torch.isfinite(sample_weights).all()
        or not torch.isfinite(alpha_values).all()
        or not np.isfinite(max_residual)
        or max_residual <= 0
    ):
        raise ValueError("KRR weights, alphas, and residual tolerance must be finite")
    if not len(alpha_values) or torch.any(sample_weights <= 0) or torch.any(alpha_values <= 0):
        raise ValueError("KRR weights and alphas must be positive")
    if not torch.isfinite(kernel).all() or not torch.isfinite(targets).all():
        raise ValueError("KRR inputs must be finite")
    sqrt_w = torch.sqrt(sample_weights)
    weighted_kernel = sqrt_w[:, None] * kernel * sqrt_w[None, :]
    eye = torch.eye(n, dtype=kernel.dtype, device=kernel.device)
    systems = weighted_kernel.unsqueeze(0) + alpha_values[:, None, None] * eye.unsqueeze(0)
    rhs = (sqrt_w[:, None] * (targets - 0.25)).unsqueeze(0).expand(len(alpha_values), -1, -1)
    factors = torch.linalg.cholesky(systems)
    solutions = torch.cholesky_solve(rhs, factors)
    residuals = systems @ solutions - rhs
    denominators = torch.linalg.vector_norm(rhs, dim=(1, 2)).clamp_min(EPSILON)
    residual_ratios = torch.linalg.vector_norm(residuals, dim=(1, 2)) / denominators
    coefficients = sqrt_w[None, :, None] * solutions
    if not torch.isfinite(coefficients).all() or not torch.isfinite(residual_ratios).all():
        raise ValueError("batched weighted KRR solve produced non-finite output")
    if float(residual_ratios.max().detach().cpu()) > max_residual:
        raise ValueError(f"Cholesky residual exceeds {max_residual}: {float(residual_ratios.max())}")
    return coefficients, residual_ratios


def safe_npz_save(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    normalized: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        value = np.asarray(array)
        if not isinstance(name, str) or not name:
            raise ValueError("safe NPZ array names must be non-empty strings")
        if value.dtype.hasobject or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"safe NPZ array {name} must be finite numeric data")
        normalized[name] = value
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **normalized)
        stream.flush()
    temporary.replace(destination)


def safe_npz_load(path: str | Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(Path(path), allow_pickle=False) as archive:
        for name in archive.files:
            value = np.array(archive[name], copy=True)
            if value.dtype.hasobject or value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError(f"unsafe or non-finite array in NPZ: {name}")
            arrays[name] = value
    if not arrays:
        raise ValueError("safe NPZ archive is empty")
    return arrays
