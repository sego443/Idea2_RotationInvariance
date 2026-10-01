"""One-off validation experiment: C4-orbit DTW nearest-class templates.

This script reads only the fixed train and validation waveform slices after
the repository loader's NPZ integrity scan. It never filters or scores test
waveforms. All feature transforms and pairwise distances are label-independent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy import signal

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR))

import run_c170_experiment as c170
import run_c170_ring_cuda_experiment_v2 as v2
import src.c170_c4_kernel_ridge_v7 as v7

FILTER_ORDER = ("bandpass_0p5_40", "lowpass_40")
K_VALUES = (1, 3, 5, 9)
ALPHA_VALUES = (0.0, 0.1, 1.0)
SHUFFLE_SEEDS = (17, 29, 43, 61, 89)
RESAMPLE_UP = 16
RESAMPLE_DOWN = 125
RESAMPLED_LENGTH = 64
DTW_BAND = 8
RMS_FLOOR = 1e-6
SAMPLE_RATE_HZ = 250.0
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "c170_c4_dtw_template_validation_20260929"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_descriptor(windows: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Filter/de-mean, common per-window RMS normalize, and resample to 64×4."""
    filtered = v7.preprocess_windows(windows, mode)
    # v7's preprocessing removes the temporal mean of each channel.  This
    # second explicit line documents and enforces the DTW contract at entry.
    centered = np.asarray(filtered) - np.mean(filtered, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.square(centered, dtype=np.float64), axis=(1, 2)))
    safe_rms = np.maximum(rms, RMS_FLOOR)
    normalized = centered / safe_rms[:, None, None]
    resampled = signal.resample_poly(
        normalized, up=RESAMPLE_UP, down=RESAMPLE_DOWN, axis=1
    )
    if resampled.shape != (len(windows), RESAMPLED_LENGTH, 4):
        raise RuntimeError(f"unexpected resampled shape {resampled.shape}")
    result = np.asarray(resampled, dtype=np.float32)
    log_rms = np.log(safe_rms).astype(np.float32)
    if not np.isfinite(result).all() or not np.isfinite(log_rms).all():
        raise ValueError("descriptor preprocessing produced non-finite values")
    return result, log_rms


def _gather_diagonal(
    values: torch.Tensor,
    indices: list[int],
    valid: list[bool],
) -> torch.Tensor:
    """Gather one anti-diagonal predecessor and mask missing boundary cells."""
    q_count, rotations, template_count, _ = values.shape
    safe = [max(0, int(index)) for index in indices]
    index = torch.as_tensor(safe, device=values.device, dtype=torch.long)
    index = index.view(1, 1, 1, -1).expand(q_count, rotations, template_count, -1)
    gathered = torch.gather(values, dim=3, index=index)
    mask = torch.as_tensor(valid, device=values.device, dtype=torch.bool)
    mask = mask.view(1, 1, 1, -1)
    return torch.where(mask, gathered, torch.full_like(gathered, float("inf")))


@torch.no_grad()
def pairwise_c4_dtw(
    query: np.ndarray,
    templates: np.ndarray,
    *,
    device: torch.device,
    query_chunk_size: int = 24,
) -> np.ndarray:
    """Band-8 squared-Euclidean DTW, minimized over four template C4 rolls."""
    q_cpu = np.asarray(query, dtype=np.float32)
    t_cpu = np.asarray(templates, dtype=np.float32)
    if q_cpu.ndim != 3 or q_cpu.shape[1:] != (RESAMPLED_LENGTH, 4):
        raise ValueError(f"query must have shape [N,64,4], got {q_cpu.shape}")
    if t_cpu.ndim != 3 or t_cpu.shape[1:] != (RESAMPLED_LENGTH, 4):
        raise ValueError(f"templates must have shape [N,64,4], got {t_cpu.shape}")
    if not np.isfinite(q_cpu).all() or not np.isfinite(t_cpu).all():
        raise ValueError("DTW inputs contain non-finite values")

    q_all = torch.as_tensor(q_cpu, device=device, dtype=torch.float32).contiguous()
    t_all = torch.as_tensor(t_cpu, device=device, dtype=torch.float32).contiguous()
    rolled_templates = torch.stack(
        [torch.roll(t_all, shifts=shift, dims=-1) for shift in range(4)], dim=0
    ).contiguous()
    output = np.empty((len(q_cpu), len(t_cpu)), dtype=np.float32)
    inf = float("inf")

    for chunk_start in range(0, len(q_cpu), query_chunk_size):
        q_chunk = q_all[chunk_start : chunk_start + query_chunk_size]
        q_count = q_chunk.shape[0]
        previous1: torch.Tensor | None = None
        previous2: torch.Tensor | None = None
        previous1_i: list[int] = []
        previous2_i: list[int] = []

        for diagonal in range(2 * RESAMPLED_LENGTH - 1):
            i_values = []
            j_values = []
            for i in range(max(0, diagonal - (RESAMPLED_LENGTH - 1)), min(RESAMPLED_LENGTH - 1, diagonal) + 1):
                j = diagonal - i
                if abs(j - i) <= DTW_BAND:
                    i_values.append(i)
                    j_values.append(j)
            if not i_values:
                raise RuntimeError(f"empty DTW anti-diagonal {diagonal}")

            i_index = torch.as_tensor(i_values, device=device, dtype=torch.long)
            j_index = torch.as_tensor(j_values, device=device, dtype=torch.long)
            q_cells = q_chunk[:, i_index, :]  # [Q,L,4]
            t_cells = rolled_templates[:, :, j_index, :]  # [4,N,L,4]
            delta = q_cells[:, None, None, :, :] - t_cells[None, :, :, :, :]
            local_cost = torch.sum(delta * delta, dim=-1)  # [Q,4,N,L]

            if diagonal == 0:
                current = local_cost
            else:
                up_indices = []
                up_valid = []
                left_indices = []
                left_valid = []
                diag_indices = []
                diag_valid = []
                previous1_lookup = {i: offset for offset, i in enumerate(previous1_i)}
                previous2_lookup = {i: offset for offset, i in enumerate(previous2_i)}
                for i, j in zip(i_values, j_values, strict=True):
                    up_coord = i - 1
                    up_ok = i > 0 and up_coord in previous1_lookup
                    up_indices.append(previous1_lookup.get(up_coord, 0))
                    up_valid.append(up_ok)

                    left_ok = j > 0 and i in previous1_lookup
                    left_indices.append(previous1_lookup.get(i, 0))
                    left_valid.append(left_ok)

                    diag_coord = i - 1
                    diag_ok = i > 0 and diag_coord in previous2_lookup
                    diag_indices.append(previous2_lookup.get(diag_coord, 0))
                    diag_valid.append(diag_ok)

                assert previous1 is not None
                up = _gather_diagonal(previous1, up_indices, up_valid)
                left = _gather_diagonal(previous1, left_indices, left_valid)
                if previous2 is None:
                    diagonal_prev = torch.full_like(local_cost, inf)
                else:
                    diagonal_prev = _gather_diagonal(previous2, diag_indices, diag_valid)
                current = local_cost + torch.minimum(torch.minimum(up, left), diagonal_prev)

            previous2, previous1 = previous1, current
            previous2_i, previous1_i = previous1_i, i_values

        if previous1 is None or previous1.shape[-1] != 1 or previous1_i != [RESAMPLED_LENGTH - 1]:
            raise RuntimeError("DTW endpoint was not reached")
        best_rotation = torch.min(previous1[..., 0], dim=1).values / RESAMPLED_LENGTH
        output[chunk_start : chunk_start + q_count] = best_rotation.cpu().numpy()

    if not np.isfinite(output).all():
        raise ValueError("DTW produced non-finite distances")
    return output


def metric_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("metric inputs must be same-length vectors")
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    recalls = []
    f1s = []
    for class_id in range(4):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id].sum()) - tp
        fp = int(confusion[:, class_id].sum()) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "macro_f1": float(np.mean(f1s)),
        "accuracy": float(np.mean(truth == predicted)),
        "recall_by_class": {name: recalls[i] for i, name in enumerate(c170.DIRECTIONS)},
        "f1_by_class": {name: f1s[i] for i, name in enumerate(c170.DIRECTIONS)},
        "prediction_counts": {
            name: int(np.count_nonzero(predicted == i)) for i, name in enumerate(c170.DIRECTIONS)
        },
        "confusion_matrix_true_rows_predicted_columns": confusion.tolist(),
    }


def predict_from_distances(
    dtw_distances: np.ndarray,
    query_log_rms: np.ndarray,
    template_log_rms: np.ndarray,
    template_labels: np.ndarray,
    *,
    k: int,
    alpha: float,
    log_rms_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    std = max(float(log_rms_std), 1e-6)
    amplitude = np.square(
        (np.asarray(query_log_rms)[:, None] - np.asarray(template_log_rms)[None, :]) / std
    )
    mixed = np.asarray(dtw_distances, dtype=np.float64) + float(alpha) * amplitude
    class_scores = np.empty((len(mixed), 4), dtype=np.float64)
    labels = np.asarray(template_labels, dtype=np.int64)
    for class_id in range(4):
        class_distances = mixed[:, labels == class_id]
        if class_distances.shape[1] < k:
            raise ValueError(f"class {class_id} has only {class_distances.shape[1]} templates for k={k}")
        nearest = np.partition(class_distances, kth=k - 1, axis=1)[:, :k]
        class_scores[:, class_id] = -np.mean(nearest, axis=1)
    predictions = np.argmax(class_scores, axis=1).astype(np.int64)
    return predictions, class_scores.astype(np.float32)


def _candidate_key(candidate: tuple[str, int, float], score: float) -> tuple[float, int, float, int]:
    filter_name, k, alpha = candidate
    return (-round(float(score), 12), -int(k), float(alpha), FILTER_ORDER.index(filter_name))


def select_candidate_by_lopo(
    train_distances: dict[str, np.ndarray],
    train_log_rms: dict[str, np.ndarray],
    y_train: np.ndarray,
    positions: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(y_train, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.int64)
    candidates = [(mode, k, alpha) for mode in FILTER_ORDER for k in K_VALUES for alpha in ALPHA_VALUES]
    fold_scores: dict[tuple[str, int, float], list[float]] = {candidate: [] for candidate in candidates}
    fold_reports: list[dict[str, Any]] = []

    for held_position in sorted(np.unique(pos).tolist()):
        held = np.flatnonzero(pos == held_position)
        fit = np.flatnonzero(pos != held_position)
        fold_scores_this_position: dict[tuple[str, int, float], float] = {}
        for filter_name in FILTER_ORDER:
            distances = train_distances[filter_name][np.ix_(held, fit)]
            log_rms = train_log_rms[filter_name]
            std = max(float(np.std(log_rms[fit], ddof=0)), 1e-6)
            amplitude = np.square((log_rms[held, None] - log_rms[None, fit]) / std)
            fit_labels = y[fit]
            for alpha in ALPHA_VALUES:
                mixed = distances.astype(np.float64) + alpha * amplitude
                per_class_sorted = []
                for class_id in range(4):
                    class_d = mixed[:, fit_labels == class_id]
                    if class_d.shape[1] < max(K_VALUES):
                        raise ValueError("LOPO fit fold has fewer than 9 templates for a class")
                    per_class_sorted.append(np.sort(class_d, axis=1)[:, : max(K_VALUES)])
                for k in K_VALUES:
                    scores = np.stack([values[:, :k].mean(axis=1) for values in per_class_sorted], axis=1)
                    pred = np.argmin(scores, axis=1)
                    candidate = (filter_name, k, alpha)
                    value = metric_report(y[held], pred)["macro_f1"]
                    fold_scores[candidate].append(float(value))
                    fold_scores_this_position[candidate] = float(value)
        best_fold_candidate = min(
            fold_scores_this_position,
            key=lambda candidate: _candidate_key(candidate, fold_scores_this_position[candidate]),
        )
        fold_reports.append(
            {
                "held_out_train_position_mm": int(held_position),
                "n_held_out": int(len(held)),
                "best_candidate_this_fold": {
                    "filter": best_fold_candidate[0],
                    "k": best_fold_candidate[1],
                    "alpha": best_fold_candidate[2],
                    "macro_f1": fold_scores_this_position[best_fold_candidate],
                },
            }
        )

    means = {candidate: float(np.mean(scores)) for candidate, scores in fold_scores.items()}
    selected = min(means, key=lambda candidate: _candidate_key(candidate, means[candidate]))
    candidate_rows = [
        {
            "filter": candidate[0],
            "k": candidate[1],
            "alpha": candidate[2],
            "mean_lopo_macro_f1": means[candidate],
            "fold_macro_f1": fold_scores[candidate],
        }
        for candidate in sorted(candidates, key=lambda c: _candidate_key(c, means[c]))
    ]
    return {
        "selected": {
            "filter": selected[0],
            "k": selected[1],
            "alpha": selected[2],
            "mean_lopo_macro_f1": means[selected],
        },
        "candidate_table": candidate_rows,
        "fold_reports": fold_reports,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=c170.DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--query-chunk-size", type=int, default=24)
    parser.add_argument("--confirm-physical-order", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse non-empty or existing output directory: {args.output_dir}")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if not torch.cuda.is_available():
        raise RuntimeError("this fixed experiment requires CUDA")
    if args.query_chunk_size < 1:
        raise ValueError("query-chunk-size must be positive")

    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()

    # load_dataset performs the previously approved compressed-NPZ decode and
    # whole-X finiteness integrity check.  Below, only train/validation indices
    # are used to slice waveform or label/metadata arrays; test indices are not
    # used for preprocessing, statistics, distance computation, or scoring.
    dataset = c170.load_dataset(args.dataset)
    assignments = {
        "train": v2.TRAIN_DISTANCE_MM,
        "val": v2.VAL_DISTANCE_MM,
        "test": v2.REGISTERED_TEST_DISTANCE_MM,
    }
    splits = c170.split_by_distance(dataset["distance_mm"], assignments)
    if len(splits["train"]) != 879 or len(splits["val"]) != 320 or len(splits["test"]) != 320:
        raise ValueError("fixed split count mismatch; expected train/val/test 879/320/320")
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    val_indices = np.asarray(splits["val"], dtype=np.int64)

    # Explicitly slice only train and validation.
    train_windows = dataset["X_raw"][train_indices]
    val_windows = dataset["X_raw"][val_indices]
    train_y = np.asarray(dataset["labels"][train_indices], dtype=np.int64)
    val_y = np.asarray(dataset["labels"][val_indices], dtype=np.int64)
    train_positions = np.asarray(dataset["distance_mm"][train_indices], dtype=np.int64)
    val_positions = np.asarray(dataset["distance_mm"][val_indices], dtype=np.int64)
    train_blocks = np.asarray(dataset["block"][train_indices], dtype=np.int64)
    train_metadata = {
        name: np.asarray(dataset[name])[np.concatenate((train_indices, val_indices))]
        for name in ("block", "distance_mm", "labels", "channel_map")
    }
    c170.assert_recording_groups_intact(train_metadata)
    channel_map_confirmation = c170.validate_channel_maps(
        train_metadata["channel_map"], confirm_physical_order=args.confirm_physical_order
    )

    print(
        f"loaded train={len(train_indices)} val={len(val_indices)}; test waveforms were not sliced; "
        "loader did decode/check full X_raw for integrity",
        flush=True,
    )
    preprocessing_started = time.monotonic()
    descriptors: dict[str, dict[str, np.ndarray]] = {}
    log_rms: dict[str, dict[str, np.ndarray]] = {}
    for mode in FILTER_ORDER:
        print(f"preprocessing {mode}", flush=True)
        descriptors[mode] = {}
        log_rms[mode] = {}
        descriptors[mode]["train"], log_rms[mode]["train"] = preprocess_descriptor(train_windows, mode)
        descriptors[mode]["val"], log_rms[mode]["val"] = preprocess_descriptor(val_windows, mode)
    preprocessing_seconds = time.monotonic() - preprocessing_started

    # A small, real-train-only implementation check before the formal matrix.
    small = descriptors[FILTER_ORDER[0]]["train"][:4]
    small_d = pairwise_c4_dtw(small, small, device=device, query_chunk_size=4)
    rolled_small = np.roll(small[:2], shift=1, axis=-1)
    rolled_d = pairwise_c4_dtw(rolled_small, small, device=device, query_chunk_size=2)
    if not np.isfinite(small_d).all() or not np.allclose(np.diag(small_d), 0.0, atol=1e-6, rtol=0.0):
        raise RuntimeError("real-train DTW self-distance sanity check failed")
    if not np.allclose(small_d, small_d.T, atol=2e-5, rtol=1e-5):
        raise RuntimeError("C4 DTW symmetry sanity check failed")
    if not np.allclose(small_d[:2], rolled_d, atol=2e-5, rtol=1e-5):
        raise RuntimeError("C4 roll-invariance sanity check failed")
    print(
        f"sanity passed: max_self={float(np.max(np.diag(small_d))):.3g}, "
        f"max_symmetry={float(np.max(np.abs(small_d-small_d.T))):.3g}, "
        f"max_roll={float(np.max(np.abs(small_d[:2]-rolled_d))):.3g}",
        flush=True,
    )

    train_distances: dict[str, np.ndarray] = {}
    distance_started = time.monotonic()
    for mode in FILTER_ORDER:
        print(f"computing train×train DTW matrix for {mode}", flush=True)
        t0 = time.monotonic()
        train_distances[mode] = pairwise_c4_dtw(
            descriptors[mode]["train"],
            descriptors[mode]["train"],
            device=device,
            query_chunk_size=args.query_chunk_size,
        )
        torch.cuda.synchronize(device)
        print(
            f"completed train×train {mode}: shape={train_distances[mode].shape} "
            f"seconds={time.monotonic()-t0:.1f}",
            flush=True,
        )

    arm_targets: dict[str, np.ndarray] = {"true": train_y.copy()}
    shuffle_metadata: dict[str, Any] = {}
    for seed in SHUFFLE_SEEDS:
        shuffled_full = c170.shuffle_training_labels_by_recording(
            dataset["labels"],
            dataset["distance_mm"],
            dataset["block"],
            train_indices,
            seed=seed + 10000,
        )
        target = np.asarray(shuffled_full[train_indices], dtype=np.int64)
        if np.array_equal(target, train_y):
            raise RuntimeError(f"shuffle seed {seed} left every training label unchanged")
        arm_name = f"shuffle_seed_{seed}"
        arm_targets[arm_name] = target
        shuffle_metadata[arm_name] = {
            "requested_seed": seed,
            "helper_seed": seed + 10000,
            "changed_training_rows": int(np.count_nonzero(target != train_y)),
            "val_targets_unchanged": True,
        }

    cv_started = time.monotonic()
    arm_selection: dict[str, Any] = {}
    train_log_rms = {mode: log_rms[mode]["train"] for mode in FILTER_ORDER}
    for arm_name, targets in arm_targets.items():
        print(f"11-position LOPO candidate selection for {arm_name}", flush=True)
        arm_selection[arm_name] = select_candidate_by_lopo(
            train_distances,
            train_log_rms,
            targets,
            train_positions,
        )
        chosen = arm_selection[arm_name]["selected"]
        print(
            f"selected {arm_name}: {chosen['filter']} k={chosen['k']} alpha={chosen['alpha']} "
            f"LOPO_macroF1={chosen['mean_lopo_macro_f1']:.6f}",
            flush=True,
        )

    selected_filters = sorted(
        {item["selected"]["filter"] for item in arm_selection.values()},
        key=FILTER_ORDER.index,
    )
    val_distances: dict[str, np.ndarray] = {}
    for mode in selected_filters:
        print(f"computing validation×train DTW matrix for selected filter {mode}", flush=True)
        t0 = time.monotonic()
        val_distances[mode] = pairwise_c4_dtw(
            descriptors[mode]["val"],
            descriptors[mode]["train"],
            device=device,
            query_chunk_size=args.query_chunk_size,
        )
        torch.cuda.synchronize(device)
        print(f"completed validation×train {mode}: seconds={time.monotonic()-t0:.1f}", flush=True)

    report_arms: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for arm_name, targets in arm_targets.items():
        chosen = arm_selection[arm_name]["selected"]
        mode = chosen["filter"]
        std = max(float(np.std(log_rms[mode]["train"], ddof=0)), 1e-6)
        pred, scores = predict_from_distances(
            val_distances[mode],
            log_rms[mode]["val"],
            log_rms[mode]["train"],
            targets,
            k=int(chosen["k"]),
            alpha=float(chosen["alpha"]),
            log_rms_std=std,
        )
        metrics = metric_report(val_y, pred)
        by_position = {}
        for position in sorted(np.unique(val_positions).tolist()):
            mask = val_positions == position
            by_position[str(int(position))] = {
                "n": int(np.count_nonzero(mask)),
                **metric_report(val_y[mask], pred[mask]),
            }
        report_arms[arm_name] = {
            "selected": chosen,
            "train_log_rms_std": std,
            "validation_metrics": metrics,
            "validation_by_position_mm": by_position,
            "candidate_table": arm_selection[arm_name]["candidate_table"],
            "fold_reports": arm_selection[arm_name]["fold_reports"],
            "shuffle": shuffle_metadata.get(arm_name),
        }
        for row, sample_index in enumerate(val_indices.tolist()):
            prediction_rows.append(
                {
                    "arm": arm_name,
                    "filter": mode,
                    "k": int(chosen["k"]),
                    "alpha": float(chosen["alpha"]),
                    "sample_index": int(dataset["sample_index"][sample_index]),
                    "block": int(dataset["block"][sample_index]),
                    "distance_mm": int(dataset["distance_mm"][sample_index]),
                    "channel_map": ",".join(str(int(x)) for x in dataset["channel_map"][sample_index]),
                    "y_true": int(val_y[row]),
                    "y_pred": int(pred[row]),
                    **{f"class_{class_id}_score": float(scores[row, class_id]) for class_id in range(4)},
                }
            )
        print(
            f"validation {arm_name}: macroF1={metrics['macro_f1']:.6f} "
            f"recall={metrics['recall_by_class']}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(args.output_dir / "validation_predictions.csv", prediction_rows)
    manifest = {
        "format": "c170-c4-dtw-template-validation-v1",
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "exploratory_only": True,
        "dataset": {
            "path": str(args.dataset.resolve()),
            "sha256": c170.dataset_sha256(args.dataset),
            "retained_nonduplicate_rows": int(len(dataset["X_raw"])),
            "duplicate_rows_excluded_by_loader": int(dataset["duplicate_rows_excluded"]),
            "raw_dtype": str(dataset["X_raw"].dtype),
            "raw_shape_after_duplicate_filter": list(dataset["X_raw"].shape),
        },
        "split": {
            "train_positions_mm": list(v2.TRAIN_DISTANCE_MM),
            "validation_positions_mm": list(v2.VAL_DISTANCE_MM),
            "registered_test_positions_mm": list(v2.REGISTERED_TEST_DISTANCE_MM),
            "excluded_positions_mm": list(v2.EXCLUDED_DISTANCE_MM),
            "train_count": int(len(train_indices)),
            "validation_count": int(len(val_indices)),
            "registered_test_count_from_split_metadata_only": int(len(splits["test"])),
            "test_waveforms_sliced_or_preprocessed_or_scored": False,
            "loader_full_npz_decode_and_X_raw_finiteness_scan": True,
            "channel_map_confirmation_train_validation_only": channel_map_confirmation,
        },
        "protocol": {
            "filters_in_tie_order": list(FILTER_ORDER),
            "filter_details": {
                "bandpass_0p5_40": "Butterworth SOS order 3, 0.5–40 Hz, filtfilt via existing helper",
                "lowpass_40": "Butterworth SOS order 3, 40 Hz low-pass, filtfilt via existing v7 helper",
            },
            "demean": "each window and channel over its 500 time samples",
            "normalization": "one common RMS over all 500×4 entries per window, floor 1e-6",
            "log_rms": "natural log of pre-normalization common RMS, retained for optional mixed distance",
            "resampling": {"method": "scipy.signal.resample_poly", "up": RESAMPLE_UP, "down": RESAMPLE_DOWN, "output_shape": [64, 4]},
            "dtw": {
                "distance": "squared Euclidean 4-channel local cost; cumulative path cost / 64",
                "sakoe_chiba_band": DTW_BAND,
                "fixed_start_and_end": True,
                "c4_orbit": "minimum across four cyclic channel rolls of the template",
                "pairwise_backend": "PyTorch CUDA anti-diagonal dynamic program, chunked over query windows",
            },
            "candidate_grid": {"k_per_class": list(K_VALUES), "alpha": list(ALPHA_VALUES), "filters": list(FILTER_ORDER)},
            "mixed_distance": "DTW + alpha*((query_logRMS-template_logRMS)/fold_train_logRMS_population_std)^2; std floor 1e-6",
            "classifier": "mean of k nearest mixed distances within each class; minimum score wins, class ID breaks ties",
            "selection": "11-fold leave-one-training-position-out mean macro-F1; ties prefer larger k, smaller alpha, then filter order",
            "shuffle": "five complete-block within-position label permutations, requested seeds 17/29/43/61/89 passed to existing helper as seed+10000; LOPO uses pseudo-labels and final validation uses true labels",
            "arms": ["true"] + [f"shuffle_seed_{seed}" for seed in SHUFFLE_SEEDS],
            "validation_role": "exploratory held-out-position score after training-position-only candidate selection",
            "test_role": "sealed; no test waveform slice, preprocessing, statistic, distance, prediction, or score",
        },
        "sanity_check_train_only": {
            "n_samples": 4,
            "passed": True,
            "max_self_distance": float(np.max(np.diag(small_d))),
            "max_symmetry_abs_error": float(np.max(np.abs(small_d - small_d.T))),
            "max_roll_abs_error": float(np.max(np.abs(small_d[:2] - rolled_d))),
        },
        "timing_seconds": {
            "preprocessing_train_validation": preprocessing_seconds,
            "pairwise_train_and_selected_validation_matrices": time.monotonic() - distance_started,
            "candidate_selection_and_validation_reporting": time.monotonic() - cv_started,
            "total": time.monotonic() - started,
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "sources": {
            "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "dataset_loader_and_splits": {"path": str(Path(c170.__file__).resolve()), "sha256": sha256_file(Path(c170.__file__))},
            "signal_helpers": {"path": str(Path(v7.__file__).resolve()), "sha256": sha256_file(Path(v7.__file__))},
            "v2_split_constants": {"path": str(Path(v2.__file__).resolve()), "sha256": sha256_file(Path(v2.__file__))},
        },
        "arms": report_arms,
        "limits": [
            "This is an exploratory comparison on the fixed C170 validation positions, not a final held-out test result.",
            "The loader decodes the compressed dataset and performs its full-array X_raw finiteness integrity scan; test rows are not sliced into the analysis or model pipeline.",
            "C4-orbit channel-roll invariance does not establish continuous physical rotation generalization.",
        ],
    }
    json_path = args.output_dir / "summary.json"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {args.output_dir / 'validation_predictions.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
