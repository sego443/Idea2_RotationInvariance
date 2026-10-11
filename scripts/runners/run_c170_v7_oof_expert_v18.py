"""Fixed selected v7 C4 group-KRR OOF expert for the v18 fusion comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from scripts.runners import run_c170_c4_kernel_ridge_v7 as v7
from scripts.runners import run_c170_experiment as c170
from scripts.runners import run_c170_ring_cuda_experiment_v2 as v2
from scripts.runners import run_c170_oof_fusion_v18 as v18
from src import c170_c4_kernel_ridge_v7 as core


PREPROCESSOR = "bandpass_0p5_40"
VIEW = "bcd"
SIGMA_FACTOR = 1.0
ALPHA = 0.1
EXPERT = "v7_group_krr"


def _probabilities(scores: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)
    totals = values.sum(axis=1, keepdims=True)
    result = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0.0)
    zero = totals[:, 0] <= 0.0
    result[zero] = 0.25
    if not np.isfinite(result).all():
        raise FloatingPointError("KRR probability conversion produced non-finite values")
    return result


def _fold_run(root: Path, data: dict[str, Any], position: int) -> None:
    fold_dir = root / f"v7_krr_position-{position:03d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    marker_path = fold_dir / "RUN_COMPLETE.json"
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            manifest = json.loads((fold_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if marker.get("heldout_train_position_mm") == position and manifest.get("status") == "complete" and (fold_dir / "fold_probabilities.npz").is_file():
                print(f"V7_OOF_REUSE position={position}", flush=True)
                return
        except Exception:
            pass

    started = time.monotonic()
    manifest: dict[str, Any] = {
        "format": "c170-v7-krr-oof-v18-fold", "status": "running",
        "config": {"expert": EXPERT, "heldout_train_position_mm": position,
                   "preprocessor": PREPROCESSOR, "view": VIEW, "kernel": "c4_group",
                   "sigma_factor": SIGMA_FACTOR, "alpha": ALPHA,
                   "fit_dtype": "torch.float64 CPU", "old_validation_used_for_fit_or_selection": False,
                   "registered_test_touched": False},
        "test_access": {"used_as_model_input": False, "preprocessed": False, "predicted_or_scored": False, "used_for_statistics": False},
    }
    v7._atomic_json(fold_dir / "run_manifest.json", manifest)
    try:
        distances = np.asarray(data["distance_mm"], dtype=np.int64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        train_idx = np.flatnonzero(np.isin(distances, v18.POSITIONS)).astype(np.int64)
        held_idx = train_idx[distances[train_idx] == position]
        fit_idx = train_idx[distances[train_idx] != position]
        val_idx = np.flatnonzero(np.isin(distances, v2.VAL_DISTANCE_MM)).astype(np.int64)
        if len(train_idx) != 879 or len(val_idx) != 320 or len(fit_idx) + len(held_idx) != 879:
            raise ValueError("unexpected v7 OOF split counts")
        if np.intersect1d(fit_idx, held_idx).size or np.intersect1d(fit_idx, val_idx).size or np.intersect1d(held_idx, val_idx).size:
            raise ValueError("v7 fold train, heldout, and oldval rows must be disjoint")

        # The selected fold alone fits the scaler and sigma; preprocessing sees
        # only fold-train, held-out-train-position, and old-validation windows.
        evaluation_idx = np.concatenate((held_idx, val_idx))
        windows = np.asarray(data["X_raw"])
        joined = np.concatenate((windows[fit_idx], windows[evaluation_idx]), axis=0)
        processed = core.preprocess_windows(joined, PREPROCESSOR)
        families = core.build_feature_families(processed)
        view_features = core.assemble_view(families, VIEW)
        n_fit = len(fit_idx)
        fit_sensor = view_features[:n_fit]
        query_sensor = view_features[n_fit:]
        scaler = core.fit_fold_scaler(fit_sensor, VIEW)
        fit_desc = core.apply_fold_scaler(fit_sensor, scaler).reshape(n_fit, 4, -1)
        query_desc = core.apply_fold_scaler(query_sensor, scaler).reshape(len(evaluation_idx), 4, -1)
        x_fit = torch.as_tensor(fit_desc, dtype=torch.float64, device="cpu")
        x_query = torch.as_tensor(query_desc, dtype=torch.float64, device="cpu")
        sigma_meta = core.estimate_sigma_base(x_fit)
        sigma = max(float(sigma_meta["sigma_base"]) * SIGMA_FACTOR, core.MIN_BASE_SIGMA)

        fit_d = v7._distance_stack(x_fit, x_fit, group=True)
        query_d = v7._distance_stack(x_query, x_fit, group=True)
        query_self_d = v7._distance_stack(x_query, x_query, group=True)
        fit_kernel, query_kernel = v7._group_kernel_from_distances(fit_d, query_d, query_self_d, sigma)
        fold_weights = v7._weights(distances[fit_idx], labels[fit_idx]).detach().to(device="cpu", dtype=torch.float64)
        targets = torch.as_tensor(v7._one_hot(labels[fit_idx]), dtype=torch.float64, device="cpu")
        coefficients, residuals = core.weighted_krr_fit_many(fit_kernel, targets, fold_weights, [ALPHA])
        scores = (0.25 + query_kernel @ coefficients[0]).detach().cpu().numpy()
        probabilities = _probabilities(scores)
        n_held = len(held_idx)
        held_prob = probabilities[:n_held]
        val_prob = probabilities[n_held:]
        if not np.isfinite(fit_desc).all() or not np.isfinite(query_desc).all():
            raise FloatingPointError("v7 KRR descriptors are non-finite")

        pred_path = fold_dir / "fold_probabilities.npz"
        core.safe_npz_save(pred_path, {
            "heldout_dataset_index": np.asarray(data["sample_index"])[held_idx].astype(np.int64),
            "heldout_distance_mm": distances[held_idx].astype(np.int64),
            "heldout_labels": labels[held_idx].astype(np.int64),
            "heldout_probabilities": held_prob.astype(np.float64),
            "oldval_dataset_index": np.asarray(data["sample_index"])[val_idx].astype(np.int64),
            "oldval_labels": labels[val_idx].astype(np.int64),
            "oldval_probabilities": val_prob.astype(np.float64),
        })
        manifest.update({
            "status": "complete", "completed_utc": v18.utc_now(),
            "split": {"fit_count": int(n_fit), "heldout_count": int(n_held), "oldval_count": int(len(val_idx)),
                      "fit_positions_mm": sorted(np.unique(distances[fit_idx]).astype(int).tolist()),
                      "heldout_position_mm": position,
                      "fit_sample_ids": np.asarray(data["sample_index"])[fit_idx].astype(int).tolist(),
                      "heldout_sample_ids": np.asarray(data["sample_index"])[held_idx].astype(int).tolist()},
            "fit": {"sigma_base": float(sigma_meta["sigma_base"]), "sigma": float(sigma),
                    "sigma_factor": SIGMA_FACTOR, "alpha": ALPHA,
                    "scaler_training_rows": int(scaler["training_sensor_rows"]),
                    "weights_mean": float(fold_weights.mean()), "solver_residual": float(residuals[0])},
            "timing_seconds": float(time.monotonic() - started), "artifacts": {"probabilities": pred_path.name},
        })
        v7._atomic_json(fold_dir / "run_manifest.json", manifest)
        v7._atomic_json(marker_path, {"format": "c170-v7-krr-oof-v18-complete", "expert": EXPERT,
                                     "heldout_train_position_mm": position, "epochs_completed": 0,
                                     "fold_probabilities": pred_path.name})
        print(f"V7_OOF_COMPLETE position={position} train={n_fit} heldout={n_held} sigma={sigma:.6f} alpha={ALPHA} seconds={time.monotonic()-started:.1f}", flush=True)
    except Exception as error:
        manifest.update({"status": "failed", "failure": {"type": type(error).__name__, "message": str(error)}})
        manifest["timing_seconds"] = time.monotonic() - started
        v7._atomic_json(fold_dir / "run_manifest.json", manifest)
        raise


def run(root: Path, dataset_path: Path) -> int:
    torch.set_num_threads(1)
    data = v18._load_dataset(dataset_path)
    failures: list[dict[str, Any]] = []
    for position in v18.POSITIONS:
        try:
            _fold_run(root, data, position)
        except Exception as error:
            failures.append({"position": position, "type": type(error).__name__, "message": str(error)})
            print(f"V7_OOF_FAILED position={position} {type(error).__name__}: {error}", flush=True)
    v7._atomic_json(root / "v7_oof_index.json", {
        "format": "c170-v7-krr-oof-v18-index", "expert": EXPERT,
        "completed_positions": [p for p in v18.POSITIONS if (root / f"v7_krr_position-{p:03d}" / "RUN_COMPLETE.json").is_file()],
        "failed": failures, "parameters": {"preprocessor": PREPROCESSOR, "view": VIEW,
                                             "kernel": "c4_group", "sigma_factor": SIGMA_FACTOR, "alpha": ALPHA},
    })
    return int(bool(failures))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=v18.DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=v18.DEFAULT_DATASET)
    args = parser.parse_args()
    raise SystemExit(run(args.root.resolve(), args.dataset.resolve()))
