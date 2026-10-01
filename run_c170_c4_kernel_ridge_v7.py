from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_experiment as c170
import run_c170_ring_cuda_experiment_v2 as v2
from src import c170_c4_kernel_ridge_v7 as core


TRAIN_MM = (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)
VAL_MM = (5, 40, 90, 130)
TEST_MM = (15, 25, 110, 150)
SEEDS = (17, 29, 43, 61, 89)
SIGMA_FACTORS = (0.5, 1.0, 2.0)
ALPHAS = (0.01, 0.1, 1.0, 10.0)
LABEL_MODES = ("group_true", *(f"group_shuffle_{seed}" for seed in SEEDS), "plain_true")
N_CLASSES = 4
DATASET_DEFAULT = c170.DEFAULT_DATASET_PATH
OUTPUT_DEFAULT = PROJECT_DIR / "outputs" / "c170_c4_kernel_ridge_v7_formal_20260929"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or prediction.shape != truth.shape or not len(truth):
        raise ValueError("metrics need matching non-empty label vectors")
    if np.any((truth < 0) | (truth >= N_CLASSES)) or np.any((prediction < 0) | (prediction >= N_CLASSES)):
        raise ValueError("class id outside the fixed four-class range")
    confusion = np.bincount(truth * N_CLASSES + prediction, minlength=N_CLASSES**2).reshape(N_CLASSES, N_CLASSES)
    true_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    recall = np.divide(np.diag(confusion), true_count, out=np.zeros(N_CLASSES, dtype=np.float64), where=true_count > 0)
    precision = np.divide(np.diag(confusion), pred_count, out=np.zeros(N_CLASSES, dtype=np.float64), where=pred_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(N_CLASSES, dtype=np.float64), where=(precision + recall) > 0)
    return {
        "macro_f1": float(f1.mean()),
        "recall_by_class": recall.tolist(),
        "precision_by_class": precision.tolist(),
        "f1_by_class": f1.tolist(),
        "confusion": confusion.tolist(),
        "unique_predictions": int(np.unique(prediction).size),
    }


def _one_hot(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or np.any((values < 0) | (values >= N_CLASSES)):
        raise ValueError("labels must be a vector of four-class ids")
    return np.eye(N_CLASSES, dtype=np.float64)[values]


def _weights(positions: np.ndarray, original_labels: np.ndarray) -> torch.Tensor:
    positions = np.asarray(positions, dtype=np.int64)
    labels = np.asarray(original_labels, dtype=np.int64)
    if positions.shape != labels.shape or positions.ndim != 1 or not len(positions):
        raise ValueError("positions and original labels must be non-empty matching vectors")
    position_weights = v2._position_weights(sorted(np.unique(positions).astype(int).tolist()))
    counts: dict[tuple[int, int], int] = {}
    for position, label in zip(positions.tolist(), labels.tolist(), strict=True):
        counts[(int(position), int(label))] = counts.get((int(position), int(label)), 0) + 1
    raw = np.asarray(
        [position_weights[int(position)] / (N_CLASSES * counts[(int(position), int(label))])
         for position, label in zip(positions.tolist(), labels.tolist(), strict=True)],
        dtype=np.float64,
    )
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("sample weights must be finite and positive")
    raw /= raw.mean()
    if not np.isfinite(raw).all() or not np.isclose(raw.mean(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("normalized fold weights must have finite mean one")
    return torch.as_tensor(raw, dtype=torch.float64, device="cuda")


def _distance_stack(x: torch.Tensor, y: torch.Tensor, *, group: bool) -> torch.Tensor:
    left = x.reshape(len(x), -1)
    left_sq = left.square().sum(dim=1, keepdim=True)
    right_values = []
    shifts = range(4) if group else (0,)
    for shift in shifts:
        rolled = torch.roll(y, shifts=shift, dims=1) if group else y
        right = rolled.reshape(len(rolled), -1)
        right_sq = right.square().sum(dim=1).unsqueeze(0)
        right_values.append((left_sq + right_sq - 2.0 * (left @ right.T)).clamp_min_(0.0))
    return torch.stack(right_values, dim=0) if group else right_values[0]


def _group_kernel_from_distances(
    train_distances: torch.Tensor,
    query_distances: torch.Tensor,
    query_self_distances: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    divisor = 2.0 * float(sigma) * float(sigma)
    raw_train = torch.exp(-train_distances / divisor).mean(dim=0)
    raw_query = torch.exp(-query_distances / divisor).mean(dim=0)
    raw_query_self = torch.exp(-query_self_distances / divisor).mean(dim=0)
    train_diag = torch.diagonal(raw_train).clamp_min(1e-12)
    query_diag = torch.diagonal(raw_query_self).clamp_min(1e-12)
    train_kernel = raw_train / torch.sqrt(train_diag[:, None] * train_diag[None, :])
    query_kernel = raw_query / torch.sqrt(query_diag[:, None] * train_diag[None, :])
    train_kernel = 0.5 * (train_kernel + train_kernel.T)
    if not torch.isfinite(train_kernel).all() or not torch.isfinite(query_kernel).all():
        raise ValueError("group kernel contains non-finite values")
    return train_kernel, query_kernel


def _plain_kernel_from_distances(
    train_distances: torch.Tensor,
    query_distances: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    divisor = 2.0 * float(sigma) * float(sigma)
    train_kernel = torch.exp(-train_distances / divisor)
    query_kernel = torch.exp(-query_distances / divisor)
    train_kernel = 0.5 * (train_kernel + train_kernel.T)
    if not torch.isfinite(train_kernel).all() or not torch.isfinite(query_kernel).all():
        raise ValueError("plain RBF kernel contains non-finite values")
    return train_kernel, query_kernel


def _kernel_predictions(
    train_kernel: torch.Tensor,
    query_kernel: torch.Tensor,
    rhs: np.ndarray,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.as_tensor(rhs, dtype=torch.float64, device="cuda")
    coefficients, residuals = core.weighted_krr_fit_many(train_kernel, targets, weights, ALPHAS)
    scores = 0.25 + torch.matmul(query_kernel.unsqueeze(0), coefficients)
    if not torch.isfinite(scores).all():
        raise ValueError("KRR scores contain non-finite values")
    return scores, residuals


def _label_targets(
    labels_true: np.ndarray,
    shuffled_targets: dict[int, np.ndarray],
    indices: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    vectors = [np.asarray(labels_true[indices], dtype=np.int64)]
    names = ["group_true"]
    for seed in SEEDS:
        vectors.append(np.asarray(shuffled_targets[seed][indices], dtype=np.int64))
        names.append(f"group_shuffle_{seed}")
    return np.concatenate([_one_hot(vector) for vector in vectors], axis=1), names


def _candidate_key(index_sigma: int, index_alpha: int) -> tuple[int, int]:
    return index_sigma, index_alpha


def _select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("candidate table is empty")
    best_score = max(float(row["mean_macro_f1"]) for row in rows)
    tied = [row for row in rows if best_score - float(row["mean_macro_f1"]) <= 1e-8]
    tied.sort(key=lambda row: (-float(row["alpha"]), int(row["candidate_order"])))
    return tied[0]


def _validate_trainval_groups(positions: np.ndarray, blocks: np.ndarray, labels: np.ndarray) -> None:
    for position in np.unique(positions):
        indices = np.flatnonzero(positions == position)
        if set(np.unique(labels[indices]).tolist()) != set(range(N_CLASSES)):
            raise ValueError(f"train/validation position {int(position)} does not contain four classes")
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        if len(np.unique(positions[indices])) != 1 or len(np.unique(labels[indices])) != 1:
            raise ValueError(f"train/validation recording block {int(block)} crosses position or label")


def _make_null_labels(labels: np.ndarray, positions: np.ndarray, blocks: np.ndarray) -> tuple[dict[int, np.ndarray], dict[str, list[dict[str, int]]]]:
    local_indices = np.arange(len(labels), dtype=np.int64)
    targets: dict[int, np.ndarray] = {}
    mappings: dict[str, list[dict[str, int]]] = {}
    for seed in SEEDS:
        shuffle_seed = seed + 10_000
        pseudo = c170.shuffle_training_labels_by_recording(labels, positions, blocks, local_indices, shuffle_seed)
        rows = []
        for position in TRAIN_MM:
            for block in sorted(np.unique(blocks[positions == position]).tolist()):
                group = np.flatnonzero((positions == position) & (blocks == block))
                if len(np.unique(pseudo[group])) != 1:
                    raise ValueError("whole-block shuffle split one recording block")
                rows.append({"position_mm": int(position), "block": int(block), "pseudo_label": int(pseudo[group[0]])})
        targets[seed] = pseudo
        mappings[str(seed)] = rows
    return targets, mappings


def _fold_candidate_scores(
    train_sensor_features: np.ndarray,
    heldout_sensor_features: np.ndarray,
    view: str,
    y_true: np.ndarray,
    shuffled_targets: dict[int, np.ndarray],
    train_positions: np.ndarray,
    heldout_positions: np.ndarray,
    train_labels: np.ndarray,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
    *,
    collect: dict[str, dict[tuple[int, int], list[float]]],
) -> dict[str, Any]:
    scaler = core.fit_fold_scaler(train_sensor_features[train_indices], view)
    train_descriptors = core.apply_fold_scaler(train_sensor_features[train_indices], scaler).reshape(-1, 4, train_sensor_features.shape[-1])
    heldout_descriptors = core.apply_fold_scaler(heldout_sensor_features[heldout_indices], scaler).reshape(-1, 4, heldout_sensor_features.shape[-1])
    x_train = torch.as_tensor(train_descriptors, dtype=torch.float64, device="cuda")
    x_heldout = torch.as_tensor(heldout_descriptors, dtype=torch.float64, device="cuda")
    sigma_meta = core.estimate_sigma_base(x_train)
    base_sigma = float(sigma_meta["sigma_base"])
    group_train_d = _distance_stack(x_train, x_train, group=True)
    group_query_d = _distance_stack(x_heldout, x_train, group=True)
    group_query_self_d = _distance_stack(x_heldout, x_heldout, group=True)
    plain_train_d = _distance_stack(x_train, x_train, group=False)
    plain_query_d = _distance_stack(x_heldout, x_train, group=False)
    weights = _weights(train_positions[train_indices], train_labels[train_indices])
    rhs_group, group_names = _label_targets(y_true, shuffled_targets, train_indices)
    heldout_truth_by_mode = [
        y_true[heldout_indices],
        *(shuffled_targets[seed][heldout_indices] for seed in SEEDS),
    ]
    max_residual = 0.0
    for sigma_index, factor in enumerate(SIGMA_FACTORS):
        sigma = max(base_sigma * factor, core.MIN_BASE_SIGMA)
        group_k, group_q = _group_kernel_from_distances(group_train_d, group_query_d, group_query_self_d, sigma)
        group_scores, group_residuals = _kernel_predictions(group_k, group_q, rhs_group, weights)
        max_residual = max(max_residual, float(group_residuals.max().detach().cpu()))
        group_scores_np = group_scores.detach().cpu().numpy()
        for alpha_index, alpha in enumerate(ALPHAS):
            values = group_scores_np[alpha_index].reshape(len(heldout_indices), len(group_names), N_CLASSES)
            for mode_index, mode in enumerate(group_names):
                predicted = values[:, mode_index, :].argmax(axis=1)
                score = _macro_metrics(heldout_truth_by_mode[mode_index], predicted)["macro_f1"]
                collect[mode][(sigma_index, alpha_index)].append(float(score))
        plain_k, plain_q = _plain_kernel_from_distances(plain_train_d, plain_query_d, sigma)
        plain_scores, plain_residuals = _kernel_predictions(plain_k, plain_q, _one_hot(y_true[train_indices]), weights)
        max_residual = max(max_residual, float(plain_residuals.max().detach().cpu()))
        plain_values = plain_scores.detach().cpu().numpy()
        for alpha_index, _alpha in enumerate(ALPHAS):
            prediction = plain_values[alpha_index].argmax(axis=1)
            score = _macro_metrics(y_true[heldout_indices], prediction)["macro_f1"]
            collect["plain_true"][(sigma_index, alpha_index)].append(float(score))
    return {
        "fold_train_count": int(len(train_indices)),
        "fold_heldout_count": int(len(heldout_indices)),
        "heldout_positions_mm": sorted(np.unique(heldout_positions[heldout_indices]).astype(int).tolist()),
        "sigma": sigma_meta,
        "max_linear_residual": max_residual,
        "feature_scaler_train_sensor_rows": int(scaler["training_sensor_rows"]),
    }


def _candidate_table(collect: dict[str, dict[tuple[int, int], list[float]]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    selected: dict[str, dict[str, Any]] = {}
    for mode in LABEL_MODES:
        rows = []
        for sigma_index, factor in enumerate(SIGMA_FACTORS):
            for alpha_index, alpha in enumerate(ALPHAS):
                fold_scores = collect[mode][(sigma_index, alpha_index)]
                rows.append({
                    "mode": mode,
                    "sigma_factor": factor,
                    "alpha": alpha,
                    "candidate_order": sigma_index * len(ALPHAS) + alpha_index,
                    "fold_macro_f1": fold_scores,
                    "mean_macro_f1": float(np.mean(fold_scores)),
                    "fold_count": len(fold_scores),
                })
        tables[mode] = rows
        selected[mode] = _select_candidate(rows)
    return tables, selected


def _actual_fit_and_validate(
    preprocessor: str,
    view: str,
    train_sensor_features: np.ndarray,
    val_sensor_features: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    train_positions: np.ndarray,
    train_blocks: np.ndarray,
    shuffled_targets: dict[int, np.ndarray],
    selected: dict[str, dict[str, Any]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    scaler = core.fit_fold_scaler(train_sensor_features, view)
    train_desc_np = core.apply_fold_scaler(train_sensor_features, scaler).reshape(len(train_labels), 4, -1)
    val_desc_np = core.apply_fold_scaler(val_sensor_features, scaler).reshape(len(val_labels), 4, -1)
    x_train = torch.as_tensor(train_desc_np, dtype=torch.float64, device="cuda")
    x_val = torch.as_tensor(val_desc_np, dtype=torch.float64, device="cuda")
    sigma_meta = core.estimate_sigma_base(x_train)
    base_sigma = float(sigma_meta["sigma_base"])
    group_train_d = _distance_stack(x_train, x_train, group=True)
    group_val_d = _distance_stack(x_val, x_train, group=True)
    group_train_self_d = group_train_d
    group_val_self_d = _distance_stack(x_val, x_val, group=True)
    plain_train_d = _distance_stack(x_train, x_train, group=False)
    plain_val_d = _distance_stack(x_val, x_train, group=False)
    weights = _weights(train_positions, train_labels)
    predictions: list[dict[str, Any]] = []
    arm_rows: list[dict[str, Any]] = []
    c4_errors: dict[str, float] = {}
    group_modes = [mode for mode in selected if mode != "plain_true"]
    group_by_candidate: dict[tuple[int, int], list[str]] = {}
    for mode in group_modes:
        row = selected[mode]
        key = (int(round(math.log(float(row["sigma_factor"]), 2))), ALPHAS.index(float(row["alpha"])))
        # factors are powers of two except 1, so explicit lookup is more robust than log arithmetic.
        sigma_index = SIGMA_FACTORS.index(float(row["sigma_factor"]))
        group_by_candidate.setdefault((sigma_index, ALPHAS.index(float(row["alpha"]))), []).append(mode)
    for (sigma_index, alpha_index), modes in group_by_candidate.items():
        sigma = max(base_sigma * SIGMA_FACTORS[sigma_index], core.MIN_BASE_SIGMA)
        alpha = ALPHAS[alpha_index]
        group_k, group_q = _group_kernel_from_distances(group_train_d, group_val_d, group_val_self_d, sigma)
        targets = np.concatenate([
            _one_hot(train_labels if mode == "group_true" else shuffled_targets[int(mode.rsplit("_", 1)[1])])
            for mode in modes
        ], axis=1)
        coeff_batch, residuals = core.weighted_krr_fit_many(group_k, targets, weights, [alpha])
        coeff = coeff_batch[0]
        scores = 0.25 + group_q @ coeff
        score_array = scores.detach().cpu().numpy().reshape(len(val_labels), len(modes), N_CLASSES)
        coeff_array = coeff.detach().cpu().numpy()
        for mode_index, mode in enumerate(modes):
            arm_coeff = coeff_array[:, mode_index * N_CLASSES:(mode_index + 1) * N_CLASSES]
            metrics = _macro_metrics(val_labels, score_array[:, mode_index, :].argmax(axis=1))
            name = f"{preprocessor}__{view}__{mode}"
            npz_path = output_root / "models" / f"{name}.npz"
            core.safe_npz_save(npz_path, {
                "train_descriptors": train_desc_np,
                "feature_mean": scaler["mean"],
                "feature_std": scaler["std"],
                "feature_raw_std": scaler["raw_std"],
                "coefficients": arm_coeff,
                "train_labels_used": train_labels if mode == "group_true" else shuffled_targets[int(mode.rsplit("_", 1)[1])],
                "sample_weights": weights.detach().cpu().numpy(),
            })
            arm_rows.append({
                "preprocessor": preprocessor, "view": view, "mode": mode, "kernel_mode": "c4_group",
                "selected_sigma_factor": SIGMA_FACTORS[sigma_index], "selected_alpha": alpha,
                "selected_cv_mean_macro_f1": selected[mode]["mean_macro_f1"],
                "selected_cv_fold_scores": json.dumps(selected[mode]["fold_macro_f1"]),
                "sigma_base": base_sigma, "sigma_median_squared_distance": sigma_meta["median_squared_distance"],
                "sigma_floor": sigma_meta["sigma_floor"], "sigma_floor_applied": sigma_meta["sigma_floor_applied"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_recall_by_class": json.dumps(metrics["recall_by_class"]),
                "validation_precision_by_class": json.dumps(metrics["precision_by_class"]),
                "validation_unique_predictions": metrics["unique_predictions"],
                "max_linear_residual": float(residuals.max().detach().cpu()),
                "state_npz": str(npz_path.relative_to(output_root)),
            })
            for index, (truth, prediction) in enumerate(zip(val_labels.tolist(), score_array[:, mode_index, :].argmax(axis=1).tolist(), strict=True)):
                predictions.append({"preprocessor": preprocessor, "view": view, "mode": mode, "row": index,
                                    "y_true": truth, "y_pred": prediction})

    if "plain_true" in selected:
        plain_choice = selected["plain_true"]
        plain_sigma_index = SIGMA_FACTORS.index(float(plain_choice["sigma_factor"]))
        plain_alpha_index = ALPHAS.index(float(plain_choice["alpha"]))
        plain_sigma = max(base_sigma * SIGMA_FACTORS[plain_sigma_index], core.MIN_BASE_SIGMA)
        plain_k, plain_q = _plain_kernel_from_distances(plain_train_d, plain_val_d, plain_sigma)
        plain_coeffs, plain_residual = core.weighted_krr_fit_many(plain_k, _one_hot(train_labels), weights, [ALPHAS[plain_alpha_index]])
        plain_coeff = plain_coeffs[0]
        plain_scores = (0.25 + plain_q @ plain_coeff).detach().cpu().numpy()
        plain_metrics = _macro_metrics(val_labels, plain_scores.argmax(axis=1))
        plain_name = f"{preprocessor}__{view}__plain_true"
        plain_npz = output_root / "models" / f"{plain_name}.npz"
        core.safe_npz_save(plain_npz, {
            "train_descriptors": train_desc_np,
            "feature_mean": scaler["mean"],
            "feature_std": scaler["std"],
            "feature_raw_std": scaler["raw_std"],
            "coefficients": plain_coeff.detach().cpu().numpy(),
            "train_labels_used": train_labels,
            "sample_weights": weights.detach().cpu().numpy(),
        })
        arm_rows.append({
            "preprocessor": preprocessor, "view": view, "mode": "plain_true", "kernel_mode": "plain_rbf",
            "selected_sigma_factor": SIGMA_FACTORS[plain_sigma_index], "selected_alpha": ALPHAS[plain_alpha_index],
            "selected_cv_mean_macro_f1": plain_choice["mean_macro_f1"],
            "selected_cv_fold_scores": json.dumps(plain_choice["fold_macro_f1"]),
            "sigma_base": base_sigma, "sigma_median_squared_distance": sigma_meta["median_squared_distance"],
            "sigma_floor": sigma_meta["sigma_floor"], "sigma_floor_applied": sigma_meta["sigma_floor_applied"],
            "validation_macro_f1": plain_metrics["macro_f1"],
            "validation_recall_by_class": json.dumps(plain_metrics["recall_by_class"]),
            "validation_precision_by_class": json.dumps(plain_metrics["precision_by_class"]),
            "validation_unique_predictions": plain_metrics["unique_predictions"],
            "max_linear_residual": float(plain_residual.max().detach().cpu()),
            "state_npz": str(plain_npz.relative_to(output_root)),
        })
        for index, (truth, prediction) in enumerate(zip(val_labels.tolist(), plain_scores.argmax(axis=1).tolist(), strict=True)):
            predictions.append({"preprocessor": preprocessor, "view": view, "mode": "plain_true", "row": index,
                                "y_true": truth, "y_pred": prediction})

    # Measure C4 score stability from this fitted true-label group model, on train and validation rows.
    if "group_true" not in selected:
        return arm_rows, predictions, c4_errors
    true_selection = selected["group_true"]
    true_sigma_index = SIGMA_FACTORS.index(float(true_selection["sigma_factor"]))
    true_alpha_index = ALPHAS.index(float(true_selection["alpha"]))
    sigma = max(base_sigma * SIGMA_FACTORS[true_sigma_index], core.MIN_BASE_SIGMA)
    true_k, true_val_k = _group_kernel_from_distances(group_train_d, group_val_d, group_val_self_d, sigma)
    true_target = _one_hot(train_labels)
    true_coeffs, _ = core.weighted_krr_fit_many(true_k, true_target, weights, [ALPHAS[true_alpha_index]])
    coeff = true_coeffs[0]
    train_base = 0.25 + true_k @ coeff
    val_base = 0.25 + true_val_k @ coeff
    divisor = 2.0 * sigma * sigma
    train_diag = torch.diagonal(torch.exp(-group_train_self_d / divisor).mean(dim=0)).clamp_min(1e-12)
    val_diag = torch.diagonal(torch.exp(-group_val_self_d / divisor).mean(dim=0)).clamp_min(1e-12)
    train_errors = []
    val_errors = []
    for roll in range(4):
        rolled_train_raw = torch.exp(-torch.roll(group_train_d, shifts=roll, dims=0) / divisor).mean(dim=0)
        rolled_train_k = rolled_train_raw / torch.sqrt(train_diag[:, None] * train_diag[None, :])
        rolled_val_raw = torch.exp(-torch.roll(group_val_d, shifts=roll, dims=0) / divisor).mean(dim=0)
        rolled_val_k = rolled_val_raw / torch.sqrt(val_diag[:, None] * train_diag[None, :])
        train_errors.append(float(torch.max(torch.abs(0.25 + rolled_train_k @ coeff - train_base)).detach().cpu()))
        val_errors.append(float(torch.max(torch.abs(0.25 + rolled_val_k @ coeff - val_base)).detach().cpu()))
    c4_errors = {"train_max_abs_score_delta": max(train_errors), "validation_max_abs_score_delta": max(val_errors)}
    return arm_rows, predictions, c4_errors


def _config(dataset_path: Path, phase: str) -> dict[str, Any]:
    source_paths = {
        "runner": Path(__file__).resolve(),
        "kernel_core": PROJECT_DIR / "src" / "c170_c4_kernel_ridge_v7.py",
        "dataset_helpers": PROJECT_DIR / "run_c170_experiment.py",
        "weight_and_shuffle_helpers": PROJECT_DIR / "run_c170_ring_cuda_experiment_v2.py",
        "signal_processing": PROJECT_DIR / "src" / "rotation_poc.py",
    }
    return {
        "format": "c170-c4-kernel-ridge-v7-run",
        "phase": phase,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": _sha256(dataset_path),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "train_distance_mm": list(TRAIN_MM),
        "validation_distance_mm": list(VAL_MM),
        "registered_test_distance_mm": list(TEST_MM),
        "excluded_distance_mm": [70],
        "train_count": 879,
        "validation_count": 320,
        "test_preprocessed": False,
        "test_metrics_computed": False,
        "preprocessors": list(core.PREPROCESSORS),
        "feature_views": list(core.VIEWS),
        "spectral_bands_hz": [list(item) for item in core.SPECTRAL_BANDS_HZ],
        "spectral_denominator": "Hann periodogram power sum for frequencies >=0.5 Hz and <=Nyquist (125 Hz); excludes DC",
        "sigma_rule": "sqrt(median of non-diagonal fold-train squared Euclidean descriptor distances)",
        "sigma_floor": core.MIN_BASE_SIGMA,
        "sigma_factor_grid": list(SIGMA_FACTORS),
        "alpha_grid": list(ALPHAS),
        "candidate_count_per_arm": len(core.PREPROCESSORS) * len(core.VIEWS) * len(SIGMA_FACTORS) * len(ALPHAS),
        "smoke_candidate_count": len(SIGMA_FACTORS) * len(ALPHAS),
        "cv": "11-fold leave-one-training-position-out; equal-weight macro-F1",
        "kernel_modes": list(core.KERNEL_MODES),
        "group_kernel": "mean of all four C4-rolled RBF kernels, normalized by sqrt(Kxx*Kyy)",
        "sample_weight_rule": "existing position Voronoi weight/(4*original true-label group size), renormalized to fold mean 1",
        "null_seed_rule": "one per-position whole-block permutation on full train using seed+10000; fixed over folds",
        "tie_break": "within 1e-8: higher alpha, then sigma-factor/alpha enumeration order",
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "python_version": platform.python_version(),
    }


def _prepare_train_val(dataset_path: Path) -> dict[str, Any]:
    before_hash = _sha256(dataset_path)
    dataset = c170.load_dataset(dataset_path)
    if _sha256(dataset_path) != before_hash:
        raise RuntimeError("dataset changed while loading")
    splits = c170.split_by_distance(dataset["distance_mm"], {
        "train": TRAIN_MM, "val": VAL_MM, "test": TEST_MM,
    })
    train_idx = splits["train"]
    val_idx = splits["val"]
    # Immediately isolate train and validation; no test sample is passed to preprocessing/features.
    train_windows = dataset["X_raw"][train_idx].copy()
    val_windows = dataset["X_raw"][val_idx].copy()
    train = {
        "labels": dataset["labels"][train_idx].copy(),
        "positions": dataset["distance_mm"][train_idx].astype(np.int64, copy=True),
        "blocks": dataset["block"][train_idx].astype(np.int64, copy=True),
        "channel_map": dataset["channel_map"][train_idx].copy(),
    }
    val = {
        "labels": dataset["labels"][val_idx].copy(),
        "positions": dataset["distance_mm"][val_idx].astype(np.int64, copy=True),
        "blocks": dataset["block"][val_idx].astype(np.int64, copy=True),
        "channel_map": dataset["channel_map"][val_idx].copy(),
    }
    del dataset, splits
    if len(train_windows) != 879 or len(val_windows) != 320:
        raise ValueError(f"expected train/val counts 879/320, got {len(train_windows)}/{len(val_windows)}")
    c170.validate_channel_maps(np.concatenate([train["channel_map"], val["channel_map"]]), confirm_physical_order=True)
    _validate_trainval_groups(train["positions"], train["blocks"], train["labels"])
    _validate_trainval_groups(val["positions"], val["blocks"], val["labels"])
    return {"train_windows": train_windows, "val_windows": val_windows, "train": train, "val": val, "source_dataset_sha256": before_hash}


def _extract_views(raw_train: np.ndarray, raw_val: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for mode in core.PREPROCESSORS:
        started = time.perf_counter()
        joined = np.concatenate([raw_train, raw_val], axis=0)
        processed = core.preprocess_windows(joined, mode)
        families = core.build_feature_families(processed)
        train_n = len(raw_train)
        output[mode] = {}
        for view in core.VIEWS:
            combined = core.assemble_view(families, view)
            output[mode][view] = {
                "train": combined[:train_n],
                "val": combined[train_n:],
            }
        print(f"features mode={mode} elapsed_sec={time.perf_counter()-started:.3f}", flush=True)
    return output


def _empty_collect(modes: tuple[str, ...] = LABEL_MODES) -> dict[str, dict[tuple[int, int], list[float]]]:
    return {
        mode: {(si, ai): [] for si in range(len(SIGMA_FACTORS)) for ai in range(len(ALPHAS))}
        for mode in modes
    }


def _run_config_cv(
    preprocessor: str,
    view: str,
    feature_data: dict[str, np.ndarray],
    labels: np.ndarray,
    positions: np.ndarray,
    shuffled_targets: dict[int, np.ndarray],
    *,
    fold_positions: list[int],
    log_prefix: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    collect = _empty_collect()
    folds = []
    all_indices = np.arange(len(labels), dtype=np.int64)
    for fold_index, heldout_position in enumerate(fold_positions, start=1):
        heldout_mask = positions == heldout_position
        train_indices = all_indices[~heldout_mask]
        heldout_indices = all_indices[heldout_mask]
        fold_started = time.perf_counter()
        fold = _fold_candidate_scores(
            feature_data["train"], feature_data["train"], view,
            labels, shuffled_targets, positions, positions, labels,
            train_indices, heldout_indices, collect=collect,
        )
        torch.cuda.synchronize()
        fold["heldout_position_mm"] = int(heldout_position)
        fold["elapsed_seconds"] = time.perf_counter() - fold_started
        folds.append(fold)
        print(f"{log_prefix} fold={fold_index}/{len(fold_positions)} position_mm={heldout_position} elapsed_sec={fold['elapsed_seconds']:.3f} max_residual={fold['max_linear_residual']:.3e}", flush=True)
    tables, selected = _candidate_table(collect)
    for mode in tables:
        for row in tables[mode]:
            row["preprocessor"] = preprocessor
            row["view"] = view
    return tables, selected, folds


def _rows_for_candidates(
    preprocessor: str,
    view: str,
    tables: dict[str, list[dict[str, Any]]],
    folds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for mode, table in tables.items():
        for candidate in table:
            rows.append({
                "preprocessor": preprocessor,
                "view": view,
                "mode": mode,
                "sigma_factor": candidate["sigma_factor"],
                "alpha": candidate["alpha"],
                "candidate_order": candidate["candidate_order"],
                "mean_macro_f1": candidate["mean_macro_f1"],
                "fold_macro_f1": json.dumps(candidate["fold_macro_f1"]),
                "fold_count": candidate["fold_count"],
            })
    return rows


def _source_snapshot(root: Path, dataset_path: Path, config: dict[str, Any]) -> None:
    snapshot = root / "source_snapshot"
    code_root = snapshot / "code"
    code_root.mkdir(parents=True, exist_ok=False)
    sources = {
        "run_c170_c4_kernel_ridge_v7.py": PROJECT_DIR / "run_c170_c4_kernel_ridge_v7.py",
        "src/c170_c4_kernel_ridge_v7.py": PROJECT_DIR / "src" / "c170_c4_kernel_ridge_v7.py",
        "run_c170_experiment.py": PROJECT_DIR / "run_c170_experiment.py",
        "run_c170_ring_cuda_experiment_v2.py": PROJECT_DIR / "run_c170_ring_cuda_experiment_v2.py",
        "src/rotation_poc.py": PROJECT_DIR / "src" / "rotation_poc.py",
        "docs/designs/2026-09-29-c170-c4-kernel-ridge-v7.md": PROJECT_DIR / "docs" / "designs" / "2026-09-29-c170-c4-kernel-ridge-v7.md",
        "docs/plans/2026-09-29-c170-c4-kernel-ridge-v7.md": PROJECT_DIR / "docs" / "plans" / "2026-09-29-c170-c4-kernel-ridge-v7.md",
    }
    hashes = {}
    for rel, source in sources.items():
        if not source.is_file():
            continue
        target = code_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[rel.replace("/", "\\")] = _sha256(target)
    _atomic_json(snapshot / "snapshot_manifest.json", {
        "format": "c170-c4-kernel-ridge-v7-source-snapshot",
        "source_sha256": hashes,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": config["dataset_sha256"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
        "test_metrics_computed": False,
    })


def run(args: argparse.Namespace) -> int:
    output_root: Path = args.output_dir.resolve()
    dataset_path: Path = args.dataset.resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v7 smoke/formal execution requires CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "models").mkdir()
    started = time.perf_counter()
    config = _config(dataset_path, args.phase)
    if args.phase == "formal":
        _source_snapshot(output_root, dataset_path, config)
    manifest: dict[str, Any] = {
        "format": config["format"], "format_version": 1, "status": "running", "config": config,
        "timing": {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "source_data_access": {
            "container_integrity_decode": True,
            "train_preprocessed": True,
            "validation_preprocessed": True,
            "test_preprocessed": False,
            "test_predictions_computed": False,
            "test_metrics_computed": False,
        },
    }
    _atomic_json(output_root / "run_manifest.json", manifest)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA disappeared before execution")
        print(f"cuda_device={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
        data = _prepare_train_val(dataset_path)
        null_targets, null_mappings = _make_null_labels(data["train"]["labels"], data["train"]["positions"], data["train"]["blocks"])
        feature_started = time.perf_counter()
        extracted = _extract_views(data["train_windows"], data["val_windows"])
        feature_seconds = time.perf_counter() - feature_started
        manifest["counts"] = {"train": len(data["train"]["labels"]), "validation": len(data["val"]["labels"])}
        manifest["split_metadata"] = {
            "train_positions_mm": sorted(np.unique(data["train"]["positions"]).astype(int).tolist()),
            "validation_positions_mm": sorted(np.unique(data["val"]["positions"]).astype(int).tolist()),
            "shuffle_seed_by_mode": {str(seed): seed + 10_000 for seed in SEEDS},
            "shuffle_mapping": null_mappings,
        }
        manifest["timing"]["feature_extraction_seconds"] = feature_seconds
        _atomic_json(output_root / "run_manifest.json", manifest)

        preprocessors = ("bandpass_0p5_40",) if args.phase == "smoke" else core.PREPROCESSORS
        views = ("ab",) if args.phase == "smoke" else core.VIEWS
        fold_positions = [TRAIN_MM[0]] if args.phase == "smoke" else list(TRAIN_MM)
        candidate_rows: list[dict[str, Any]] = []
        global_tables: dict[str, list[dict[str, Any]]] = {mode: [] for mode in LABEL_MODES}
        summary_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        cv_elapsed = 0.0
        final_elapsed = 0.0
        c4_diagnostics: dict[str, dict[str, float]] = {}
        all_selected: dict[str, Any] = {}
        config_index = 0
        for preprocessor in preprocessors:
            for view in views:
                conf_started = time.perf_counter()
                tables, selected, folds = _run_config_cv(
                    preprocessor, view, extracted[preprocessor][view], data["train"]["labels"],
                    data["train"]["positions"], null_targets, fold_positions=fold_positions,
                    log_prefix=f"config={preprocessor}/{view}",
                )
                conf_cv_seconds = time.perf_counter() - conf_started
                cv_elapsed += conf_cv_seconds
                for mode in LABEL_MODES:
                    for row in tables[mode]:
                        row["candidate_order"] += config_index * len(SIGMA_FACTORS) * len(ALPHAS)
                        global_tables[mode].append(row)
                # Persist each completed configuration; monitoring remains stdout-only.
                partial = output_root / "partial" / f"{preprocessor}__{view}.json"
                partial.parent.mkdir(exist_ok=True)
                _atomic_json(partial, {
                    "preprocessor": preprocessor, "view": view, "candidate_table": tables,
                    "local_selected_preview": selected, "folds": folds,
                    "cv_elapsed_seconds": conf_cv_seconds,
                    "test_metrics_computed": False,
                })
                print(f"completed_config={preprocessor}/{view} cv_elapsed_sec={conf_cv_seconds:.3f}", flush=True)
                config_index += 1

        expected_candidates_per_arm = len(preprocessors) * len(views) * len(SIGMA_FACTORS) * len(ALPHAS)
        for mode in LABEL_MODES:
            if len(global_tables[mode]) != expected_candidates_per_arm:
                raise RuntimeError(f"{mode} has {len(global_tables[mode])} candidates; expected {expected_candidates_per_arm}")
            all_selected[mode] = _select_candidate(global_tables[mode])
        for mode in LABEL_MODES:
            candidate_rows.extend(global_tables[mode])

        if args.phase == "formal":
            for preprocessor in preprocessors:
                for view in views:
                    selected_here = {
                        mode: row for mode, row in all_selected.items()
                        if row["preprocessor"] == preprocessor and row["view"] == view
                    }
                    if not selected_here:
                        continue
                    final_started = time.perf_counter()
                    rows, preds, c4_error = _actual_fit_and_validate(
                        preprocessor, view, extracted[preprocessor][view]["train"], extracted[preprocessor][view]["val"],
                        data["train"]["labels"], data["val"]["labels"], data["train"]["positions"],
                        data["train"]["blocks"], null_targets, selected_here, output_root,
                    )
                    torch.cuda.synchronize()
                    conf_final_seconds = time.perf_counter() - final_started
                    final_elapsed += conf_final_seconds
                    summary_rows.extend(rows)
                    prediction_rows.extend(preds)
                    if c4_error:
                        c4_diagnostics[f"{preprocessor}/{view}"] = c4_error
                    print(f"final={preprocessor}/{view} selected_arms={len(rows)} elapsed_sec={conf_final_seconds:.3f} c4={c4_error}", flush=True)

        if candidate_rows:
            _atomic_csv(output_root / "cv_candidates.csv", candidate_rows,
                        ["preprocessor", "view", "mode", "sigma_factor", "alpha", "candidate_order", "mean_macro_f1", "fold_macro_f1", "fold_count"])
        if summary_rows:
            _atomic_csv(output_root / "selected_validation_summary.csv", summary_rows,
                        list(summary_rows[0].keys()))
            _atomic_csv(output_root / "validation_predictions.csv", prediction_rows,
                        ["preprocessor", "view", "mode", "row", "y_true", "y_pred"])
        manifest["selection"] = all_selected
        manifest["c4_score_diagnostics"] = c4_diagnostics
        manifest["timing"].update({
            "feature_extraction_seconds": feature_seconds,
            "cv_seconds": cv_elapsed,
            "final_fit_and_validation_seconds": final_elapsed,
            "total_elapsed_seconds": time.perf_counter() - started,
        })
        manifest["artifacts"] = {}
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and path.name != "run_manifest.json" and not path.name.endswith(".tmp"):
                manifest["artifacts"][str(path.relative_to(output_root))] = _sha256(path)
        manifest["status"] = "completed"
        _atomic_json(output_root / "run_manifest.json", manifest)
        marker = {
            "format": "c170-c4-kernel-ridge-v7-complete",
            "status": "completed",
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
            "manifest_sha256": _sha256(output_root / "run_manifest.json"),
            "artifact_sha256": manifest["artifacts"],
            "test_metrics_computed": False,
        }
        _atomic_json(output_root / "RUN_COMPLETE.json", marker)
        print(f"RUN_COMPLETE root={output_root} elapsed_sec={manifest['timing']['total_elapsed_seconds']:.3f}", flush=True)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        manifest["timing"]["total_elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(output_root / "run_manifest.json", manifest)
        print(f"FAILED {type(error).__name__}: {error}", flush=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C170 train/validation-only C4 group-kernel KRR v7")
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
