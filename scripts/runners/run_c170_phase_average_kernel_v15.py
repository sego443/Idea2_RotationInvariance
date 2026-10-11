from __future__ import annotations

"""C170 analytic continuous-phase average Gaussian KRR experiment (v15).

Runs three positive-definite group-integrated kernels on train/old-validation
only: ideal U(1) phase average, U(1) x Nyquist-sign average, and exact C4
orbit-average control. Candidate selection is nested in 11-position train LOPO.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import signal

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from scripts.runners import run_c170_experiment as c170
from scripts.runners import run_c170_c4_kernel_ridge_v7 as v7
from src import c170_c4_kernel_ridge_v7 as core


TRAIN_MM = v7.TRAIN_MM
VAL_MM = v7.VAL_MM
TEST_MM = v7.TEST_MM
SEEDS = v7.SEEDS
SIGMA_FACTORS = (0.5, 1.0, 2.0)
ALPHAS = (0.01, 0.1, 1.0, 10.0)
FILTERS = ("bandpass_0p5_40", "bandpass_0p1_40", "lowpass_40")
NORMALIZATIONS = ("train_median_abs", "window_global_rms")
KERNELS = ("u1_phase", "u1_phase_nyquist_c2", "c4_discrete")
ARMS = ("group_true", *(f"group_shuffle_{seed}" for seed in SEEDS))
LABELS = ("UP", "DOWN", "LEFT", "RIGHT")
DATASET_DEFAULT = c170.DEFAULT_DATASET_PATH
OUTPUT_DEFAULT = PROJECT_DIR / "outputs" / "c170_phase_average_kernel_v15_formal_20260930"
EPS = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _metric(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    return v7._macro_metrics(truth, prediction)


def _filter_center(raw: np.ndarray, kind: str) -> np.ndarray:
    """Filter in source dtype; cast only after filtering, then temporal demean."""
    values = np.asarray(raw)
    if kind == "bandpass_0p5_40":
        filtered = c170.filter_windows(values.copy(), sample_rate=250.0)
    elif kind == "bandpass_0p1_40":
        sos = signal.butter(3, (0.1, 40.0), btype="bandpass", fs=250.0, output="sos")
        filtered = signal.sosfiltfilt(sos, values.copy(), axis=1)
    elif kind == "lowpass_40":
        sos = signal.butter(3, 40.0, btype="lowpass", fs=250.0, output="sos")
        filtered = signal.sosfiltfilt(sos, values.copy(), axis=1)
    else:
        raise ValueError(f"unknown filter {kind}")
    result = np.asarray(filtered, dtype=np.float32)
    result -= result.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"filter {kind} produced non-finite values")
    return result


def _load_train_val(dataset_path: Path) -> dict[str, Any]:
    before = _sha256(dataset_path)
    dataset = c170.load_dataset(dataset_path)
    if _sha256(dataset_path) != before:
        raise RuntimeError("dataset changed while decoding")
    split = c170.split_by_distance(dataset["distance_mm"], {"train": TRAIN_MM, "val": VAL_MM, "test": TEST_MM})
    ti, vi = split["train"], split["val"]
    # Immediately isolate training/old-validation arrays. Test is only present
    # in the NPZ integrity decode and is never filtered or scored here.
    train = {"raw": dataset["X_raw"][ti].copy(), "y": dataset["labels"][ti].copy(),
             "pos": dataset["distance_mm"][ti].astype(np.int64, copy=True),
             "block": dataset["block"][ti].astype(np.int64, copy=True),
             "sample_id": dataset["sample_index"][ti].astype(np.int64, copy=True),
             "channel_map": dataset["channel_map"][ti].copy()}
    val = {"raw": dataset["X_raw"][vi].copy(), "y": dataset["labels"][vi].copy(),
           "pos": dataset["distance_mm"][vi].astype(np.int64, copy=True),
           "block": dataset["block"][vi].astype(np.int64, copy=True),
           "sample_id": dataset["sample_index"][vi].astype(np.int64, copy=True),
           "channel_map": dataset["channel_map"][vi].copy()}
    del dataset, split
    if len(train["y"]) != 879 or len(val["y"]) != 320:
        raise ValueError(f"expected 879/320 train/validation, found {len(train['y'])}/{len(val['y'])}")
    c170.validate_channel_maps(np.concatenate((train["channel_map"], val["channel_map"])), confirm_physical_order=True)
    v7._validate_trainval_groups(train["pos"], train["block"], train["y"])
    v7._validate_trainval_groups(val["pos"], val["block"], val["y"])
    return {"train": train, "val": val, "dataset_sha256": before}


def _normal_resample(
    filtered_train: np.ndarray,
    filtered_query: np.ndarray,
    train_idx: np.ndarray,
    query_idx: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_raw = filtered_train[train_idx]
    query_raw = filtered_query[query_idx]
    if normalization == "train_median_abs":
        scale = max(float(np.median(np.abs(train_raw.astype(np.float64)))), 1e-6)
        train_norm = train_raw / scale
        query_norm = query_raw / scale
        metadata = {"train_median_abs_scale": scale}
    elif normalization == "window_global_rms":
        train_rms = np.sqrt(np.mean(np.square(train_raw.astype(np.float64)), axis=(1, 2), keepdims=True))
        query_rms = np.sqrt(np.mean(np.square(query_raw.astype(np.float64)), axis=(1, 2), keepdims=True))
        train_norm = train_raw / np.maximum(train_rms, 1e-6).astype(np.float32)
        query_norm = query_raw / np.maximum(query_rms, 1e-6).astype(np.float32)
        metadata = {"train_global_rms_median": float(np.median(train_rms)),
                    "query_global_rms_median": float(np.median(query_rms))}
    else:
        raise ValueError(f"unknown normalization {normalization}")
    train64 = signal.resample_poly(train_norm, up=16, down=125, axis=1).astype(np.float64)
    query64 = signal.resample_poly(query_norm, up=16, down=125, axis=1).astype(np.float64)
    if train64.shape[1:] != (64, 4) or query64.shape[1:] != (64, 4):
        raise RuntimeError("polyphase resampling did not produce [64,4]")
    return train64, query64, metadata


def _fit_shared_time_scaler(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Identical time-coordinate statistics are shared by all four sensors;
    # therefore the affine transform commutes with every channel roll.
    mean = train.mean(axis=(0, 2), dtype=np.float64)
    std = train.std(axis=(0, 2), dtype=np.float64)
    return mean, np.maximum(std, 1e-12)


def _apply_shared_time_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = (values - mean[None, :, None]) / std[None, :, None]
    if not np.isfinite(out).all():
        raise ValueError("shared time scaler produced non-finite values")
    return out


def _components(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    x = torch.as_tensor(values, dtype=torch.float64, device="cuda")
    if x.ndim != 3 or x.shape[1:] != (64, 4):
        raise ValueError("waveforms must be [N,64,4]")
    s = x.sum(dim=2) * 0.5
    a = (x[:, :, 0] - x[:, :, 2]) / math.sqrt(2.0)
    b = (x[:, :, 1] - x[:, :, 3]) / math.sqrt(2.0)
    n = (x[:, :, 0] - x[:, :, 1] + x[:, :, 2] - x[:, :, 3]) * 0.5
    return s, a, b, n


def _rotate_components(parts: tuple[torch.Tensor, ...], theta: float, sign_n: float = 1.0) -> tuple[torch.Tensor, ...]:
    s, a, b, n = parts
    c, si = math.cos(theta), math.sin(theta)
    return s, c * a - si * b, si * a + c * b, sign_n * n


def _log_i0_nonnegative(x: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch.special, "i0e"):
        raise RuntimeError("this PyTorch build lacks torch.special.i0e required for stable phase integration")
    positive = torch.clamp(x, min=0.0)
    return torch.log(torch.special.i0e(positive).clamp_min(torch.finfo(positive.dtype).tiny)) + positive


def _log_cosh(x: torch.Tensor) -> torch.Tensor:
    a = torch.abs(x)
    return a + torch.log1p(torch.exp(-2.0 * a)) - math.log(2.0)


def _raw_kernel(x: tuple[torch.Tensor, ...], y: tuple[torch.Tensor, ...], sigma: float, kind: str) -> torch.Tensor:
    if sigma <= 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and positive")
    sx, ax, bx, nx = x
    sy, ay, by, ny = y
    sig2 = float(sigma) ** 2
    sx2, sy2 = sx.square().sum(1), sy.square().sum(1)
    ds = (sx2[:, None] + sy2[None, :] - 2.0 * sx @ sy.T).clamp_min(0.0)
    zxn = ax.square().sum(1) + bx.square().sum(1)
    zyn = ay.square().sum(1) + by.square().sum(1)
    cross_r = ax @ ay.T + bx @ by.T
    cross_i = ax @ by.T - bx @ ay.T
    zn = torch.sqrt((cross_r.square() + cross_i.square()).clamp_min(0.0))
    if kind == "u1_phase":
        log_kernel = -ds / (2.0 * sig2) - (zxn[:, None] + zyn[None, :]) / (2.0 * sig2)
        log_kernel = log_kernel + _log_i0_nonnegative(zn / sig2)
        # The exact group average is <= 1; clamp only floating roundoff above 1.
        return torch.exp(torch.clamp(log_kernel, max=0.0))
    if kind == "u1_phase_nyquist_c2":
        nxn, nyn = nx.square().sum(1), ny.square().sum(1)
        ndot = nx @ ny.T
        log_kernel = -ds / (2.0 * sig2) - (zxn[:, None] + zyn[None, :]) / (2.0 * sig2)
        log_kernel = log_kernel + _log_i0_nonnegative(zn / sig2)
        log_kernel = log_kernel - (nxn[:, None] + nyn[None, :]) / (2.0 * sig2)
        log_kernel = log_kernel + _log_cosh(ndot / sig2)
        return torch.exp(torch.clamp(log_kernel, max=0.0))
    if kind == "c4_discrete":
        nxn, nyn = nx.square().sum(1), ny.square().sum(1)
        ndot = nx @ ny.T
        real_correlations = (cross_r, -cross_i, -cross_r, cross_i)
        total = torch.zeros_like(ds)
        for rotation, correlation in enumerate(real_correlations):
            sign = 1.0 if rotation % 2 == 0 else -1.0
            distance = ds + zxn[:, None] + zyn[None, :] - 2.0 * correlation
            distance = distance + nxn[:, None] + nyn[None, :] - 2.0 * sign * ndot
            total = total + torch.exp(-distance.clamp_min(0.0) / (2.0 * sig2))
        return total * 0.25
    raise ValueError(f"unknown kernel {kind}")


def _normalized_kernels(
    train: tuple[torch.Tensor, ...], query: tuple[torch.Tensor, ...], sigma: float, kind: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_train = _raw_kernel(train, train, sigma, kind)
    raw_train = 0.5 * (raw_train + raw_train.T)
    raw_query = _raw_kernel(query, train, sigma, kind)
    raw_query_self = _raw_kernel(query, query, sigma, kind)
    td = torch.diagonal(raw_train).clamp_min(1e-12)
    qd = torch.diagonal(raw_query_self).clamp_min(1e-12)
    train_kernel = raw_train / torch.sqrt(td[:, None] * td[None, :])
    query_kernel = raw_query / torch.sqrt(qd[:, None] * td[None, :])
    if not torch.isfinite(train_kernel).all() or not torch.isfinite(query_kernel).all():
        raise ValueError(f"{kind} normalized kernel has non-finite values")
    return train_kernel, query_kernel, td


def _sigma_base(parts: tuple[torch.Tensor, ...], kind: str) -> dict[str, float]:
    s, a, b, n = parts
    if kind == "u1_phase":
        descriptor = torch.cat((s, a, b), dim=1)
    else:
        descriptor = torch.cat((s, a, b, n), dim=1)
    d2 = torch.cdist(descriptor, descriptor, p=2).square()
    upper = torch.triu(torch.ones_like(d2, dtype=torch.bool), diagonal=1)
    values = d2[upper]
    med2 = float(torch.median(values).detach().cpu())
    sig = math.sqrt(max(med2, 0.0))
    return {"median_pairwise_squared_distance": med2, "sigma_base": max(sig, 1e-6),
            "sigma_floor_applied": float(sig < 1e-6)}


def _weights(positions: np.ndarray, labels_original: np.ndarray) -> torch.Tensor:
    return v7._weights(positions, labels_original)


def _one_hot(labels: np.ndarray) -> np.ndarray:
    return v7._one_hot(labels)


def _run_kernel_sanity() -> dict[str, float]:
    # One compact runtime check (no separate test suite) before formal scoring.
    gen = torch.Generator(device="cuda").manual_seed(150930)
    x = torch.randn((9, 64, 4), generator=gen, dtype=torch.float64, device="cuda")
    y = torch.randn((7, 64, 4), generator=gen, dtype=torch.float64, device="cuda")
    xp, yp = _components(x), _components(y)
    diagnostics: dict[str, float] = {}
    for kind in KERNELS:
        k = _raw_kernel(xp, xp, 1.7, kind)
        symmetry = torch.max(torch.abs(k - k.T))
        eigen_min = torch.linalg.eigvalsh(0.5 * (k + k.T)).min()
        if not torch.isfinite(k).all() or float(symmetry) > 1e-10 or float(eigen_min) < -1e-8:
            raise RuntimeError(f"{kind} kernel failed finite/symmetry/PSD sanity")
        if kind == "u1_phase":
            transformed = _rotate_components(yp, 0.713)
        elif kind == "u1_phase_nyquist_c2":
            transformed = _rotate_components(yp, 0.713, -1.0)
        else:
            transformed = _components(torch.roll(y, shifts=1, dims=2))
        invariant_delta = torch.max(torch.abs(_raw_kernel(xp, yp, 1.7, kind) - _raw_kernel(xp, transformed, 1.7, kind)))
        if not torch.isfinite(invariant_delta) or float(invariant_delta) > 1e-9:
            raise RuntimeError(f"{kind} group invariance sanity failed: {float(invariant_delta)}")
        diagnostics[kind] = {"min_eigenvalue": float(eigen_min), "max_symmetry_error": float(symmetry),
                             "max_group_invariance_error": float(invariant_delta)}
    torch.cuda.synchronize()
    return diagnostics  # type: ignore[return-value]


def _fold_features(
    filtered_train: np.ndarray, filtered_val: np.ndarray,
    train_indices: np.ndarray, query_windows: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    xtr, xq, norm_meta = _normal_resample(filtered_train, filtered_val, train_indices, query_windows, normalization)
    mean, std = _fit_shared_time_scaler(xtr)
    return _apply_shared_time_scaler(xtr, mean, std), _apply_shared_time_scaler(xq, mean, std), mean, std, norm_meta


def _load_filtered(data: dict[str, Any], filter_name: str) -> tuple[np.ndarray, np.ndarray]:
    train = _filter_center(data["train"]["raw"], filter_name)
    val = _filter_center(data["val"]["raw"], filter_name)
    return train, val


def _empty_collect() -> dict[str, dict[str, dict[int, list[float]]]]:
    return {kind: {arm: {} for arm in ARMS} for kind in KERNELS}


def _add_fold_scores(
    collect: dict[str, dict[str, dict[int, list[float]]]],
    config_order: int,
    kind: str,
    sigma_index: int,
    alpha_index: int,
    values: np.ndarray,
    y_true: np.ndarray,
    shuffled: dict[int, np.ndarray],
    heldout_indices: np.ndarray,
) -> None:
    candidate_order = config_order * len(SIGMA_FACTORS) * len(ALPHAS) + sigma_index * len(ALPHAS) + alpha_index
    for mode_index, arm in enumerate(ARMS):
        key = candidate_order
        prediction = values[:, mode_index * 4:(mode_index + 1) * 4].argmax(axis=1)
        if arm == "group_true":
            truth = y_true[heldout_indices]
        else:
            seed = int(arm.rsplit("_", 1)[1])
            truth = shuffled[seed][heldout_indices]
        score = _metric(truth, prediction)["macro_f1"]
        collect[kind][arm].setdefault(key, []).append(float(score))


def _candidate_tables(
    collect: dict[str, dict[str, dict[int, list[float]]]],
    config_meta: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, dict[str, dict[str, Any]]]]:
    tables: dict[str, dict[str, list[dict[str, Any]]]] = {}
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in KERNELS:
        tables[kind], selected[kind] = {}, {}
        for arm in ARMS:
            rows = []
            for order, scores in sorted(collect[kind][arm].items()):
                config_order, within = divmod(order, len(SIGMA_FACTORS) * len(ALPHAS))
                sigma_index, alpha_index = divmod(within, len(ALPHAS))
                meta = config_meta[config_order]
                rows.append({
                    "kernel": kind, "arm": arm, "filter": meta["filter"], "normalization": meta["normalization"],
                    "sigma_factor": SIGMA_FACTORS[sigma_index], "alpha": ALPHAS[alpha_index],
                    "candidate_order": order, "fold_macro_f1": scores,
                    "mean_macro_f1": float(np.mean(scores)), "fold_count": len(scores),
                })
            if not rows:
                raise RuntimeError(f"no candidates for {kind}/{arm}")
            best = max(row["mean_macro_f1"] for row in rows)
            tied = [row for row in rows if best - row["mean_macro_f1"] <= 1e-8]
            tied.sort(key=lambda row: (-float(row["alpha"]), int(row["candidate_order"])))
            tables[kind][arm] = rows
            selected[kind][arm] = tied[0]
    return tables, selected


def _manifest_config(dataset_path: Path) -> dict[str, Any]:
    return {
        "format": "c170-phase-average-kernel-v15", "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path), "train_positions_mm": list(TRAIN_MM),
        "validation_positions_mm": list(VAL_MM), "registered_test_positions_mm": list(TEST_MM),
        "train_count": 879, "validation_count": 320,
        "test_integrity_decode_only": True, "test_preprocessed": False, "test_predictions_computed": False,
        "test_metrics_computed": False, "filters": list(FILTERS), "normalizations": list(NORMALIZATIONS),
        "kernels": list(KERNELS), "sigma_factors": list(SIGMA_FACTORS), "ridge_alphas": list(ALPHAS),
        "candidate_count_per_kernel_arm": len(FILTERS) * len(NORMALIZATIONS) * len(SIGMA_FACTORS) * len(ALPHAS),
        "selection": "11-fold leave-one-training-position-out, equal-weight macro-F1; all settings selected on train only",
        "shared_time_scaler": "fit per fold over fold-training samples x four sensors at each resampled time; shared across sensors",
        "sigma_rule": "sqrt median non-diagonal Euclidean squared distance; U1 excludes Nyquist component; other kernels use full orthonormal DFT representation",
        "kernel_definitions": {
            "u1_phase": "analytic uniform SO(2) phase integral over complex first spatial harmonic; excludes Nyquist coordinate",
            "u1_phase_nyquist_c2": "analytic U(1) phase integral and independent Nyquist sign average",
            "c4_discrete": "exact arithmetic average of four raw-waveform C4 Gaussian kernels",
        },
        "sample_weights": "v7 position Voronoi/(4 x original true-label group size), fold normalized to mean one",
        "nulls": "five full-train per-position whole-block shuffles, seeds 10017/10029/10043/10061/10089; reused across folds",
        "tie_break": "within 1e-8, larger alpha then earlier fixed grid order",
        "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
        "python_version": platform.python_version(),
    }


def _snapshot(root: Path) -> None:
    target = root / "source_snapshot"
    target.mkdir()
    sources = {
        'scripts/runners/run_c170_phase_average_kernel_v15.py': Path(__file__).resolve(),
        'scripts/runners/run_c170_c4_kernel_ridge_v7.py': PROJECT_DIR / 'scripts/runners/run_c170_c4_kernel_ridge_v7.py',
        "src/c170_c4_kernel_ridge_v7.py": PROJECT_DIR / "src" / "c170_c4_kernel_ridge_v7.py",
        'scripts/runners/run_c170_experiment.py': PROJECT_DIR / 'scripts/runners/run_c170_experiment.py',
        'scripts/runners/run_c170_ring_cuda_experiment_v2.py': PROJECT_DIR / 'scripts/runners/run_c170_ring_cuda_experiment_v2.py',
        "src/rotation_poc.py": PROJECT_DIR / "src" / "rotation_poc.py",
    }
    hashes = {}
    for rel, source in sources.items():
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, out)
        hashes[rel.replace("/", "\\")] = _sha256(out)
    _atomic_json(target / "snapshot_manifest.json", {
        "format": "c170-phase-average-kernel-v15-source-snapshot", "source_sha256": hashes,
        "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
    })


def _run(args: argparse.Namespace) -> int:
    root = args.output_dir.resolve()
    dataset_path = args.dataset.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    if not torch.cuda.is_available():
        raise RuntimeError("v15 requires CUDA")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root.mkdir(parents=True, exist_ok=False)
    (root / "models").mkdir()
    started = time.perf_counter()
    config = _manifest_config(dataset_path)
    _snapshot(root)
    manifest: dict[str, Any] = {
        "format": config["format"], "format_version": 1, "status": "running", "config": config,
        "source_data_access": {"container_integrity_decode": True, "train_preprocessed": True,
                               "validation_preprocessed": True, "test_preprocessed": False,
                               "test_predictions_computed": False, "test_metrics_computed": False},
        "timing": {"started_unix": time.time()},
    }
    _atomic_json(root / "run_manifest.json", manifest)
    try:
        print(f"cuda_device={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
        sanity = _run_kernel_sanity()
        print(f"kernel_sanity={sanity}", flush=True)
        data = _load_train_val(dataset_path)
        y, pos, blocks = data["train"]["y"], data["train"]["pos"], data["train"]["block"]
        shuffled, mappings = v7._make_null_labels(y, pos, blocks)
        manifest["split_metadata"] = {
            "train_positions_mm": sorted(np.unique(pos).astype(int).tolist()),
            "validation_positions_mm": sorted(np.unique(data["val"]["pos"]).astype(int).tolist()),
            "shuffle_seed_by_arm": {str(seed): seed + 10000 for seed in SEEDS}, "shuffle_mapping": mappings,
        }
        manifest["kernel_sanity"] = sanity
        _atomic_json(root / "run_manifest.json", manifest)

        collect = _empty_collect()
        config_meta: list[dict[str, Any]] = []
        all_idx = np.arange(len(y), dtype=np.int64)
        fold_positions = list(TRAIN_MM)
        cv_started = time.perf_counter()
        config_order = 0
        raw_train_windows, raw_val_windows = data["train"]["raw"], data["val"]["raw"]
        for filter_name in FILTERS:
            filter_started = time.perf_counter()
            ftrain, fval = _load_filtered(data, filter_name)
            print(f"filtered={filter_name} elapsed_sec={time.perf_counter()-filter_started:.2f}", flush=True)
            for normalization in NORMALIZATIONS:
                meta = {"filter": filter_name, "normalization": normalization}
                config_meta.append(meta)
                config_started = time.perf_counter()
                for fold_no, heldout_position in enumerate(fold_positions, start=1):
                    heldout_mask = pos == heldout_position
                    tri = all_idx[~heldout_mask]
                    qri = all_idx[heldout_mask]
                    # LOPO's held-out block is still drawn from the original
                    # training array; old validation is only used at final refit.
                    tr_np, q_np, _mean, _std, _norm_info = _fold_features(ftrain, ftrain, tri, qri, normalization)
                    tr_parts, q_parts = _components(tr_np), _components(q_np)
                    weights = _weights(pos[tri], y[tri])
                    rhs_blocks = [_one_hot(y[tri])]
                    rhs_blocks.extend(_one_hot(shuffled[seed][tri]) for seed in SEEDS)
                    rhs = np.concatenate(rhs_blocks, axis=1)
                    target_cuda = torch.as_tensor(rhs, dtype=torch.float64, device="cuda")
                    held_truths = [y[qri], *(shuffled[seed][qri] for seed in SEEDS)]
                    fold_sigma: dict[str, dict[str, float]] = {}
                    for kernel in KERNELS:
                        sigma_meta = _sigma_base(tr_parts, kernel)
                        fold_sigma[kernel] = sigma_meta
                        base = sigma_meta["sigma_base"]
                        for sigma_index, factor in enumerate(SIGMA_FACTORS):
                            sigma = max(base * factor, 1e-6)
                            kt, kq, _train_diag = _normalized_kernels(tr_parts, q_parts, sigma, kernel)
                            coeffs, residuals = core.weighted_krr_fit_many(kt, target_cuda, weights, ALPHAS)
                            scores = 0.25 + torch.matmul(kq.unsqueeze(0), coeffs)
                            score_np = scores.detach().cpu().numpy()
                            if not np.isfinite(score_np).all():
                                raise RuntimeError(f"non-finite scores {filter_name}/{normalization}/{kernel}")
                            for alpha_index in range(len(ALPHAS)):
                                _add_fold_scores(collect, config_order, kernel, sigma_index, alpha_index,
                                                 score_np[alpha_index], y, shuffled, qri)
                            max_residual = float(residuals.max().detach().cpu())
                            if max_residual > 1e-8:
                                raise RuntimeError(f"KRR solve residual {max_residual:.3e} exceeds tolerance")
                    torch.cuda.synchronize()
                    print(f"cv={filter_name}/{normalization} fold={fold_no}/11 position={heldout_position} sec={time.perf_counter()-config_started:.1f}", flush=True)
                config_order += 1
                print(f"completed_config={filter_name}/{normalization} elapsed_sec={time.perf_counter()-config_started:.1f}", flush=True)
            del ftrain, fval
        cv_seconds = time.perf_counter() - cv_started
        tables, selected = _candidate_tables(collect, config_meta)

        candidate_rows: list[dict[str, Any]] = []
        for kernel in KERNELS:
            for arm in ARMS:
                candidate_rows.extend(tables[kernel][arm])
        _atomic_csv(root / "cv_candidates.csv", candidate_rows,
                    ["kernel", "arm", "filter", "normalization", "sigma_factor", "alpha", "candidate_order", "mean_macro_f1", "fold_macro_f1", "fold_count"])
        _atomic_json(root / "cv_selection.json", {"selected": selected, "test_metrics_computed": False})
        print("train_LOPO_selection=" + json.dumps({k: {a: {"filter": r["filter"], "normalization": r["normalization"],
               "sigma_factor": r["sigma_factor"], "alpha": r["alpha"], "cv_f1": r["mean_macro_f1"]}
               for a, r in v.items()} for k, v in selected.items()}, ensure_ascii=False), flush=True)

        val = data["val"]
        result_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        final_started = time.perf_counter()
        # One full-train refit per selected kernel/config/sigma, batching all
        # selected null/true arms and alphas where possible.
        for kernel in KERNELS:
            grouped: dict[tuple[str, str, float], list[str]] = {}
            for arm in ARMS:
                row = selected[kernel][arm]
                key = (str(row["filter"]), str(row["normalization"]), float(row["sigma_factor"]))
                grouped.setdefault(key, []).append(arm)
            filter_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for (filter_name, normalization, sigma_factor), arms in grouped.items():
                if filter_name not in filter_cache:
                    filter_cache[filter_name] = _load_filtered(data, filter_name)
                ftrain, fval = filter_cache[filter_name]
                alltri = np.arange(len(y), dtype=np.int64)
                tr_np, q_np, mean, std, norm_info = _fold_features(ftrain, fval, alltri,
                                                                   np.arange(len(val["y"]), dtype=np.int64), normalization)
                tr_parts, q_parts = _components(tr_np), _components(q_np)
                sigma_meta = _sigma_base(tr_parts, kernel)
                sigma = max(sigma_meta["sigma_base"] * sigma_factor, 1e-6)
                kt, kq, train_diag = _normalized_kernels(tr_parts, q_parts, sigma, kernel)
                # Selected arms in this group can use different ridge values;
                # solve all four alpha settings once and retain the chosen rows.
                target_by_arm = {arm: (y if arm == "group_true" else shuffled[int(arm.rsplit("_", 1)[1])]) for arm in arms}
                rhs = np.concatenate([_one_hot(target_by_arm[arm]) for arm in arms], axis=1)
                targets = torch.as_tensor(rhs, dtype=torch.float64, device="cuda")
                weights = _weights(pos, y)
                coeffs_all, residuals = core.weighted_krr_fit_many(kt, targets, weights, ALPHAS)
                all_scores = 0.25 + torch.matmul(kq.unsqueeze(0), coeffs_all)
                scores_np = all_scores.detach().cpu().numpy()
                for arm_index, arm in enumerate(arms):
                    selection = selected[kernel][arm]
                    alpha_index = ALPHAS.index(float(selection["alpha"]))
                    score_block = scores_np[alpha_index, :, arm_index * 4:(arm_index + 1) * 4]
                    pred = score_block.argmax(axis=1)
                    metrics = _metric(val["y"], pred)
                    name = f"{kernel}__{filter_name}__{normalization}__{arm}"
                    model_path = root / "models" / f"{name}.npz"
                    arm_targets = target_by_arm[arm]
                    core.safe_npz_save(model_path, {
                        "train_wave_standardized": tr_np,
                        "time_mean": mean,
                        "time_std": std,
                        "normalization_scale": np.asarray(norm_info.get("train_median_abs_scale", 1.0), dtype=np.float64),
                        "train_kernel_raw_diagonal": train_diag.detach().cpu().numpy(),
                        "coefficients": coeffs_all[alpha_index, :, arm_index * 4:(arm_index + 1) * 4].detach().cpu().numpy(),
                        "train_labels_used": arm_targets,
                        "sample_weights": weights.detach().cpu().numpy(),
                        "train_sample_id": data["train"]["sample_id"],
                    })
                    result_rows.append({
                        "kernel": kernel, "arm": arm, "filter": filter_name, "normalization": normalization,
                        "sigma_factor": sigma_factor, "sigma_base": sigma_meta["sigma_base"], "sigma": sigma,
                        "sigma_median_pairwise_squared_distance": sigma_meta["median_pairwise_squared_distance"],
                        "alpha": float(selection["alpha"]), "selected_cv_macro_f1": float(selection["mean_macro_f1"]),
                        "selected_cv_fold_scores": json.dumps(selection["fold_macro_f1"]),
                        "validation_macro_f1": metrics["macro_f1"],
                        "validation_recall_by_class": json.dumps(metrics["recall_by_class"]),
                        "validation_precision_by_class": json.dumps(metrics["precision_by_class"]),
                        "validation_confusion": json.dumps(metrics["confusion"]),
                        "validation_unique_predictions": metrics["unique_predictions"],
                        "max_linear_residual": float(residuals[alpha_index].detach().cpu()),
                        "normalization_metadata": json.dumps(norm_info), "state_npz": str(model_path.relative_to(root)),
                    })
                    for i in range(len(val["y"])):
                        prediction_rows.append({
                            "kernel": kernel, "arm": arm, "sample_id": int(val["sample_id"][i]),
                            "position_mm": int(val["pos"][i]), "block": int(val["block"][i]),
                            "y_true": int(val["y"][i]), "y_pred": int(pred[i]),
                            "score_UP": float(score_block[i, 0]), "score_DOWN": float(score_block[i, 1]),
                            "score_LEFT": float(score_block[i, 2]), "score_RIGHT": float(score_block[i, 3]),
                        })
                print(f"final={kernel}/{filter_name}/{normalization} arms={len(arms)}", flush=True)
            del filter_cache
        final_seconds = time.perf_counter() - final_started
        _atomic_csv(root / "selected_validation_summary.csv", result_rows, list(result_rows[0].keys()))
        _atomic_csv(root / "validation_predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
        manifest["timing"].update({"cv_seconds": cv_seconds, "final_refit_validation_seconds": final_seconds,
                                   "total_seconds": time.perf_counter() - started})
        manifest["selection"] = selected
        manifest["validation_summary"] = result_rows
        manifest["artifacts"] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.name not in ("run_manifest.json", "RUN_COMPLETE.json") and not path.name.endswith(".tmp"):
                manifest["artifacts"][str(path.relative_to(root))] = _sha256(path)
        manifest["status"] = "completed"
        _atomic_json(root / "run_manifest.json", manifest)
        _atomic_json(root / "RUN_COMPLETE.json", {
            "format": "c170-phase-average-kernel-v15-complete", "status": "completed",
            "manifest_sha256": _sha256(root / "run_manifest.json"),
            "artifact_sha256": manifest["artifacts"], "test_metrics_computed": False,
        })
        print(f"RUN_COMPLETE root={root} elapsed_sec={manifest['timing']['total_seconds']:.2f}", flush=True)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        manifest["timing"]["total_seconds"] = time.perf_counter() - started
        _atomic_json(root / "run_manifest.json", manifest)
        print(f"FAILED {type(error).__name__}: {error}", flush=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(_run(_parse_args()))
