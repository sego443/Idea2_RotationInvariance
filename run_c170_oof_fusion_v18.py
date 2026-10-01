"""Fixed-protocol position-held-out experts and OOF fusion for C170 (v18)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

import run_c170_experiment as c170
import run_c170_ring_cuda_experiment_v2 as v2
import run_c170_ring_cuda_experiment_v3 as v3
import run_c170_complex_orbit_experiment_v11 as v11
import run_c170_temporal_phase_invariant_experiment_v6 as v6
from src.c170_sensor_normalization_v14 import prepare_train_val
from src.c170_temporal_phase_invariant_model_v6 import TemporalPhaseInvariantClassifierV6


EXPERTS = ("complex_gram", "learned_power", "window_sensor_rms_complex_gram")
POSITIONS = tuple(v2.TRAIN_DISTANCE_MM)
SEEDS = (17, 29)
EPOCHS = 120
BATCHES_PER_EPOCH = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
DROPOUT = 0.2
FUSION_STEPS = 200
FUSION_LR = 0.01
FUSION_L2 = 0.01
DEFAULT_DATASET = v2.DEFAULT_DATASET
DEFAULT_ROOT = PROJECT / "outputs" / "c170_oof_fusion_v18_formal_20260930"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_json(path: Path, payload: Any) -> None:
    v2._atomic_write_json(path, payload)


def _load_dataset(path: Path) -> dict[str, np.ndarray | int]:
    names = (
        "X_raw", "block", "distance_mm", "direction", "repetition",
        "source_repetition", "is_duplicate", "channel_map",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(names) - set(archive.files))
        if missing:
            raise ValueError(f"dataset missing required arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in names}
    n = len(arrays["X_raw"])
    if arrays["X_raw"].ndim != 3 or arrays["X_raw"].shape[1:] != (500, 4):
        raise ValueError(f"expected windows [N,500,4], got {arrays['X_raw'].shape}")
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("dataset arrays have inconsistent sample counts")
    keep = ~arrays["is_duplicate"].astype(bool)
    data: dict[str, np.ndarray | int] = {name: value[keep].copy() for name, value in arrays.items()}
    data["is_duplicate"] = np.zeros(int(keep.sum()), dtype=bool)
    data["sample_index"] = np.flatnonzero(keep).astype(np.int64)
    data["labels"] = c170.encode_directions(np.asarray(data["direction"]))
    return data


def _fold_indices(data: dict[str, Any], heldout_position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.asarray(data["distance_mm"], dtype=np.int64)
    train_all = np.flatnonzero(np.isin(distances, POSITIONS)).astype(np.int64)
    oldval = np.flatnonzero(np.isin(distances, v2.VAL_DISTANCE_MM)).astype(np.int64)
    heldout = train_all[distances[train_all] == int(heldout_position)]
    fold_train = train_all[distances[train_all] != int(heldout_position)]
    if len(train_all) != 879 or len(oldval) != 320 or not len(heldout):
        raise ValueError(f"unexpected original train/oldval/heldout counts: {len(train_all)}/{len(oldval)}/{len(heldout)}")
    if set(np.unique(distances[fold_train]).tolist()) != set(POSITIONS) - {int(heldout_position)}:
        raise ValueError("fold training positions do not match leave-one-position-out assignment")
    if set(np.unique(distances[heldout]).tolist()) != {int(heldout_position)}:
        raise ValueError("held-out samples cross position boundaries")
    if np.intersect1d(fold_train, heldout).size or np.intersect1d(fold_train, oldval).size or np.intersect1d(heldout, oldval).size:
        raise ValueError("fold train, heldout, and old validation indices must be disjoint")
    return fold_train, heldout, oldval


def _prepare_fold(
    data: dict[str, Any], fold_train: np.ndarray, heldout: np.ndarray, oldval: np.ndarray, expert: str
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = np.asarray(data["X_raw"])
    evaluation = np.concatenate((heldout, oldval))
    if expert == "window_sensor_rms_complex_gram":
        processed, info = prepare_train_val(
            "window_sensor_rms", raw,
            {"train": fold_train, "val": evaluation},
        )
        return processed["train"], processed["val"], info
    processed, scale = v3.prepare_model_inputs_shared_scale_v3(
        raw, {"train": fold_train, "val": evaluation}
    )
    return processed["train"], processed["val"], {
        "normalization": "v3 shared scalar median(abs), fitted on this fold's 10 train positions",
        "shared_sensor_scale": float(scale),
        "fit_positions_mm": sorted(np.unique(np.asarray(data["distance_mm"])[fold_train]).astype(int).tolist()),
        "processed_splits": ["fold_train", "heldout_train_position", "old_validation"],
        "registered_test_preprocessed": False,
    }


def _fold_groups(indices: np.ndarray, labels: np.ndarray, distances: np.ndarray) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray, dict[int, float]]:
    groups: dict[tuple[int, int], np.ndarray] = {}
    by_group: dict[tuple[int, int], list[int]] = {}
    for local, original in enumerate(indices.tolist()):
        key = (int(distances[original]), int(labels[original]))
        by_group.setdefault(key, []).append(local)
    expected_positions = set(np.unique(distances[indices]).astype(int).tolist())
    if len(expected_positions) != 10:
        raise ValueError(f"expected 10 fold training positions, got {len(expected_positions)}")
    for position in expected_positions:
        if {label for current, label in by_group if current == position} != set(range(4)):
            raise ValueError(f"position {position} is missing one of four action groups")
    groups = {key: np.asarray(value, dtype=np.int64) for key, value in sorted(by_group.items())}
    position_weights = v2._position_weights(sorted(expected_positions))
    weights = np.zeros(len(indices), dtype=np.float32)
    for (position, _label), rows in groups.items():
        weights[rows] = np.float32(position_weights[position] / (4 * len(rows)))
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=2e-7):
        raise ValueError("fold training sample weights must sum to one")
    return groups, weights, position_weights


def _batches(groups: dict[tuple[int, int], np.ndarray], seed: int, epoch: int) -> list[np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    keys = sorted(groups)
    order = rng.permutation(len(keys))
    shuffled = {keys[int(i)]: rng.permutation(groups[keys[int(i)]]) for i in order}
    result: list[np.ndarray] = []
    for step in range(max(len(rows) for rows in shuffled.values())):
        batch = np.asarray([rows[step] for rows in shuffled.values() if step < len(rows)], dtype=np.int64)
        rng.shuffle(batch)
        result.append(batch)
    flattened = np.concatenate(result)
    expected = np.concatenate(list(groups.values()))
    if len(result) != BATCHES_PER_EPOCH or len(flattened) != len(expected):
        raise RuntimeError(f"sampler produced {len(result)} batches / {len(flattened)} rows")
    if len(np.unique(flattened)) != len(expected) or set(flattened.tolist()) != set(expected.tolist()):
        raise RuntimeError("sampler must use every fold-train sample exactly once")
    return result


def _model(expert: str) -> torch.nn.Module:
    if expert == "complex_gram" or expert == "window_sensor_rms_complex_gram":
        return v11.C170ComplexOrbitClassifierV11("complex_gram")
    if expert == "learned_power":
        return TemporalPhaseInvariantClassifierV6("learned_power")
    raise ValueError(f"unknown expert {expert}")


def _npz_state(path: Path, model: torch.nn.Module) -> None:
    v3._save_numpy_state_dict(path, model.state_dict())


def _predict(model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), 96):
            logits = model(inputs[start : start + 96]).float()
            probs = torch.softmax(logits, dim=1)
            if not torch.isfinite(probs).all():
                raise FloatingPointError("non-finite inference probability")
            chunks.append(probs.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def run_one(root: Path, dataset_path: Path, expert: str, position: int, seed: int, device_name: str = "cuda") -> Path:
    if expert not in EXPERTS or position not in POSITIONS or seed not in SEEDS:
        raise ValueError("task is outside the frozen v18 grid")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    run_dir = root / f"{expert}__position-{position:03d}__seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    started = time.monotonic()
    config = {
        "expert": expert,
        "heldout_train_position_mm": position,
        "seed": seed,
        "epochs": EPOCHS,
        "checkpoint": "last_epoch_120",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout": DROPOUT,
        "loss": "20 * sum(fold_sample_weight * per_sample_cross_entropy) per minibatch",
        "fold_sampler": "SeedSequence(seed,epoch), rotate/shuffle position-direction groups, every fold-train row once; 20 batches",
        "dataset": str(dataset_path.resolve()),
        "registered_test_touched": False,
    }
    manifest: dict[str, Any] = {
        "format": "c170-oof-fusion-v18-fold-run",
        "status": "running",
        "config": config,
        "started_utc": utc_now(),
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "split": {}, "preprocessing": {}, "epoch_batch_counts": [],
        "test_access": {"used_as_model_input": False, "preprocessed": False,
                         "predicted_or_scored": False, "used_for_statistics": False},
    }
    atomic_json(manifest_path, manifest)
    try:
        data = _load_dataset(dataset_path)
        distances = np.asarray(data["distance_mm"], dtype=np.int64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        fold_train, heldout, oldval = _fold_indices(data, position)
        x_train_np, x_eval_np, prep_info = _prepare_fold(data, fold_train, heldout, oldval, expert)
        if not np.isfinite(x_train_np).all() or not np.isfinite(x_eval_np).all():
            raise FloatingPointError("preprocessed fold inputs are non-finite")
        groups, weights_np, position_weights = _fold_groups(fold_train, labels, distances)
        train_targets_np = labels[fold_train]
        x_train = torch.from_numpy(np.ascontiguousarray(x_train_np)).to(device=device, dtype=torch.float32)
        y_train = torch.from_numpy(train_targets_np).to(device=device, dtype=torch.long)
        w_train = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)
        x_eval = torch.from_numpy(np.ascontiguousarray(x_eval_np)).to(device=device, dtype=torch.float32)
        manifest["split"] = {
            "fold_train_count": int(len(fold_train)), "heldout_train_position_count": int(len(heldout)),
            "old_validation_count": int(len(oldval)), "fold_train_positions_mm": sorted(position_weights),
            "heldout_position_mm": position, "fold_train_sample_indices": np.asarray(data["sample_index"])[fold_train].astype(int).tolist(),
            "heldout_sample_indices": np.asarray(data["sample_index"])[heldout].astype(int).tolist(),
            "old_validation_sample_indices": np.asarray(data["sample_index"])[oldval].astype(int).tolist(),
            "position_weights": {str(k): float(v) for k, v in position_weights.items()},
            "weights_sum": float(weights_np.sum()),
        }
        manifest["preprocessing"] = prep_info
        atomic_json(manifest_path, manifest)

        model = _model(expert).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        history: list[dict[str, Any]] = []
        train_start = time.monotonic()
        for epoch in range(EPOCHS):
            model.train()
            batch_ids = _batches(groups, seed, epoch)
            total_loss = 0.0
            for batch in batch_ids:
                idx = torch.as_tensor(batch, dtype=torch.long, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_train[idx]).float()
                per_sample = F.cross_entropy(logits, y_train[idx], reduction="none", label_smoothing=0.0)
                loss = float(BATCHES_PER_EPOCH) * torch.sum(w_train[idx] * per_sample)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError(f"non-finite gradient at epoch {epoch + 1}")
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            history.append({"epoch": epoch + 1, "optimizer_updates": len(batch_ids),
                            "batch_sizes": [int(len(b)) for b in batch_ids],
                            "weighted_objective_sum": total_loss})
            manifest["epoch_batch_counts"].append({"epoch": epoch + 1, "batches": len(batch_ids),
                                                   "samples": int(sum(map(len, batch_ids)))})
            if (epoch + 1) % 10 == 0:
                print(f"EPOCH expert={expert} position={position} seed={seed} epoch={epoch + 1}/{EPOCHS} loss={total_loss:.6f}", flush=True)
        train_seconds = time.monotonic() - train_start
        eval_prob = _predict(model, x_eval)
        n_held = len(heldout)
        held_prob = eval_prob[:n_held]
        val_prob = eval_prob[n_held:]
        if held_prob.shape != (n_held, 4) or val_prob.shape != (len(oldval), 4):
            raise RuntimeError("inference probability shapes do not match heldout/oldval rows")
        prediction_path = run_dir / "fold_probabilities.npz"
        np.savez_compressed(
            prediction_path,
            heldout_dataset_index=np.asarray(data["sample_index"])[heldout].astype(np.int64),
            heldout_distance_mm=distances[heldout].astype(np.int64),
            heldout_labels=labels[heldout].astype(np.int64),
            heldout_probabilities=held_prob,
            oldval_dataset_index=np.asarray(data["sample_index"])[oldval].astype(np.int64),
            oldval_labels=labels[oldval].astype(np.int64),
            oldval_probabilities=val_prob,
        )
        with (run_dir / "training_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=("epoch", "optimizer_updates", "batch_sizes", "weighted_objective_sum"))
            writer.writeheader()
            for row in history:
                writer.writerow({**row, "batch_sizes": json.dumps(row["batch_sizes"], separators=(",", ":"))})
        _npz_state(run_dir / "last_epoch_state.npz", model)
        manifest["status"] = "complete"
        manifest["completed_utc"] = utc_now()
        manifest["timing"] = {"training_seconds": train_seconds, "total_seconds": time.monotonic() - started}
        manifest["last_epoch_weighted_objective_sum"] = history[-1]["weighted_objective_sum"]
        manifest["artifacts"] = {
            "fold_probabilities": prediction_path.name,
            "training_history": "training_history.csv",
            "last_epoch_state": "last_epoch_state.npz",
            "epochs_completed": len(history),
        }
        atomic_json(manifest_path, manifest)
        atomic_json(run_dir / "RUN_COMPLETE.json", {
            "format": "c170-oof-fusion-v18-fold-complete", "expert": expert,
            "heldout_train_position_mm": position, "seed": seed,
            "epochs_completed": EPOCHS, "manifest": manifest_path.name,
            "fold_probabilities": prediction_path.name,
        })
        print(f"RUN_COMPLETE expert={expert} position={position} seed={seed} seconds={train_seconds:.1f}", flush=True)
        return run_dir
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["timing"] = {"total_seconds": time.monotonic() - started}
        atomic_json(manifest_path, manifest)
        raise


def _expert_probabilities(root: Path, expert: str, train_indices: np.ndarray, val_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_rows = {int(index): [] for index in train_indices.tolist()}
    val_rows: list[list[np.ndarray]] = [[] for _ in val_indices]
    val_map = {int(index): offset for offset, index in enumerate(val_indices.tolist())}
    for position in POSITIONS:
        expected = set(np.flatnonzero(np.asarray(train_indices) >= 0).tolist())
        del expected
        for seed in SEEDS:
            run_dir = root / f"{expert}__position-{position:03d}__seed-{seed}"
            marker = json.loads((run_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
            if marker.get("epochs_completed") != EPOCHS:
                raise ValueError(f"incomplete fixed-epoch marker: {run_dir}")
            with np.load(run_dir / "fold_probabilities.npz", allow_pickle=False) as archive:
                held_idx = archive["heldout_dataset_index"].astype(np.int64)
                held_dist = archive["heldout_distance_mm"].astype(np.int64)
                held_prob = np.asarray(archive["heldout_probabilities"], dtype=np.float64)
                v_idx = archive["oldval_dataset_index"].astype(np.int64)
                v_prob = np.asarray(archive["oldval_probabilities"], dtype=np.float64)
            if held_prob.shape != (len(held_idx), 4) or not np.isfinite(held_prob).all() or not np.isfinite(v_prob).all():
                raise ValueError(f"bad numeric fold probabilities: {run_dir}")
            if set(np.unique(held_dist).tolist()) != {position}:
                raise ValueError(f"heldout position mismatch: {run_dir}")
            for index, prob in zip(held_idx.tolist(), held_prob):
                if index not in train_rows:
                    raise ValueError(f"unexpected OOF training id {index} in {run_dir}")
                train_rows[index].append(prob)
            if not np.array_equal(v_idx, val_indices):
                raise ValueError(f"oldval identity/order mismatch: {run_dir}")
            for i, prob in enumerate(v_prob):
                val_rows[i].append(prob)
    if any(len(rows) != len(SEEDS) for rows in train_rows.values()):
        bad = [index for index, rows in train_rows.items() if len(rows) != len(SEEDS)]
        raise ValueError(f"OOF train coverage is not exactly 2 seeds per row: {bad[:10]}")
    if any(len(rows) != len(POSITIONS) * len(SEEDS) for rows in val_rows):
        raise ValueError("oldval needs one probability from every position/seed fold model")
    train_prob = np.stack([np.mean(train_rows[int(index)], axis=0) for index in train_indices])
    val_prob = np.stack([np.mean(rows, axis=0) for rows in val_rows])
    return train_prob, val_prob


def _v7_oof_probabilities(root: Path, train_indices: np.ndarray, val_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_rows: dict[int, np.ndarray] = {}
    val_rows: list[list[np.ndarray]] = [[] for _ in val_indices]
    expected_train = set(int(i) for i in train_indices.tolist())
    for position in POSITIONS:
        directory = root / f"v7_krr_position-{position:03d}"
        marker = json.loads((directory / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
        if marker.get("heldout_train_position_mm") != position or manifest.get("status") != "complete":
            raise ValueError(f"incomplete v7 group-KRR OOF fold {position}")
        with np.load(directory / "fold_probabilities.npz", allow_pickle=False) as archive:
            held_idx = archive["heldout_dataset_index"].astype(np.int64)
            held_pos = archive["heldout_distance_mm"].astype(np.int64)
            held_prob = np.asarray(archive["heldout_probabilities"], dtype=np.float64)
            oldval_idx = archive["oldval_dataset_index"].astype(np.int64)
            oldval_prob = np.asarray(archive["oldval_probabilities"], dtype=np.float64)
        if set(np.unique(held_pos).tolist()) != {position} or held_prob.shape != (len(held_idx), 4):
            raise ValueError(f"v7 heldout IDs/probabilities mismatch at {position} mm")
        if not np.isfinite(held_prob).all() or not np.isfinite(oldval_prob).all():
            raise FloatingPointError(f"non-finite v7 fold probability at {position} mm")
        for index, prob in zip(held_idx.tolist(), held_prob):
            if index not in expected_train or index in train_rows:
                raise ValueError(f"v7 OOF train ID missing/duplicated across folds: {index}")
            train_rows[index] = prob
        if not np.array_equal(oldval_idx, val_indices):
            raise ValueError("v7 oldval ID/order mismatch across folds")
        for offset, prob in enumerate(oldval_prob):
            val_rows[offset].append(prob)
    if set(train_rows) != expected_train or any(len(rows) != len(POSITIONS) for rows in val_rows):
        raise ValueError("v7 OOF coverage mismatch")
    train_prob = np.stack([train_rows[int(index)] for index in train_indices])
    val_prob = np.stack([np.mean(rows, axis=0) for rows in val_rows])
    return train_prob, val_prob


def _fit_fusions(root: Path, dataset_path: Path) -> dict[str, Any]:
    data = _load_dataset(dataset_path)
    distances = np.asarray(data["distance_mm"], dtype=np.int64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    train_idx = np.flatnonzero(np.isin(distances, POSITIONS)).astype(np.int64)
    val_idx = np.flatnonzero(np.isin(distances, v2.VAL_DISTANCE_MM)).astype(np.int64)
    expert_train: list[np.ndarray] = []
    expert_val: list[np.ndarray] = []
    for expert in EXPERTS:
        p_train, p_val = _expert_probabilities(root, expert, np.asarray(data["sample_index"])[train_idx], np.asarray(data["sample_index"])[val_idx])
        expert_train.append(p_train)
        expert_val.append(p_val)
    v7_train, v7_val = _v7_oof_probabilities(
        root, np.asarray(data["sample_index"])[train_idx], np.asarray(data["sample_index"])[val_idx]
    )
    train_stack = np.stack(expert_train, axis=1)  # N, expert, class
    val_stack = np.stack(expert_val, axis=1)
    if train_stack.shape != (879, 3, 4) or val_stack.shape != (320, 3, 4):
        raise ValueError(f"expected OOF/oldval tensor shapes (879,3,4)/(320,3,4), got {train_stack.shape}/{val_stack.shape}")
    if not np.isfinite(train_stack).all() or not np.isfinite(val_stack).all():
        raise FloatingPointError("non-finite fusion inputs")
    y_train_np, y_val_np = labels[train_idx], labels[val_idx]
    groups = v2._training_groups(train_idx, labels, distances)
    _position_weights, sample_weights_by_index = v2._training_sample_weights(groups)
    weights_np = np.asarray([sample_weights_by_index[int(i)] for i in train_idx], dtype=np.float32)
    device = torch.device("cpu")
    x_train = torch.log(torch.clamp(torch.from_numpy(train_stack).float().reshape(879, 12), min=1e-6))
    x_val = torch.log(torch.clamp(torch.from_numpy(val_stack).float().reshape(320, 12), min=1e-6))
    y_train = torch.from_numpy(y_train_np).long()
    y_val = torch.from_numpy(y_val_np).long()
    w_train = torch.from_numpy(weights_np).float()
    expert_train_t = torch.from_numpy(train_stack).float()
    expert_val_t = torch.from_numpy(val_stack).float()

    models: dict[str, torch.nn.Module] = {}
    stacker = torch.nn.Linear(12, 4)
    torch.nn.init.zeros_(stacker.weight); torch.nn.init.zeros_(stacker.bias)
    optimizer = torch.optim.Adam(stacker.parameters(), lr=FUSION_LR)
    for _ in range(FUSION_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = stacker(x_train)
        loss = torch.sum(w_train * F.cross_entropy(logits, y_train, reduction="none")) + FUSION_L2 * stacker.weight.square().mean()
        loss.backward(); optimizer.step()
    models["linear_stack"] = stacker
    stack_val_prob = torch.softmax(stacker(x_val), dim=1).detach().numpy()

    gate = torch.nn.Linear(12, 3)
    torch.nn.init.zeros_(gate.weight); torch.nn.init.zeros_(gate.bias)
    optimizer = torch.optim.Adam(gate.parameters(), lr=FUSION_LR)
    for _ in range(FUSION_STEPS):
        optimizer.zero_grad(set_to_none=True)
        gate_weights = torch.softmax(gate(x_train), dim=1)
        mixed = torch.sum(gate_weights.unsqueeze(-1) * expert_train_t, dim=1).clamp_min(1e-8)
        loss = -torch.sum(w_train * torch.log(mixed[torch.arange(len(y_train)), y_train])) + FUSION_L2 * gate.weight.square().mean()
        loss.backward(); optimizer.step()
    models["linear_gate"] = gate
    with torch.no_grad():
        gate_weights_val = torch.softmax(gate(x_val), dim=1)
        gate_val_prob = torch.sum(gate_weights_val.unsqueeze(-1) * expert_val_t, dim=1).numpy()

    train_stack4 = np.concatenate((train_stack, v7_train[:, None, :]), axis=1)
    val_stack4 = np.concatenate((val_stack, v7_val[:, None, :]), axis=1)
    x_train4 = torch.log(torch.clamp(torch.from_numpy(train_stack4).float().reshape(879, 16), min=1e-6))
    x_val4 = torch.log(torch.clamp(torch.from_numpy(val_stack4).float().reshape(320, 16), min=1e-6))
    experts_train4 = torch.from_numpy(train_stack4).float()
    experts_val4 = torch.from_numpy(val_stack4).float()
    stacker4 = torch.nn.Linear(16, 4)
    torch.nn.init.zeros_(stacker4.weight); torch.nn.init.zeros_(stacker4.bias)
    optimizer = torch.optim.Adam(stacker4.parameters(), lr=FUSION_LR)
    for _ in range(FUSION_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = stacker4(x_train4)
        loss = torch.sum(w_train * F.cross_entropy(logits, y_train, reduction="none")) + FUSION_L2 * stacker4.weight.square().mean()
        loss.backward(); optimizer.step()
    stack_val_prob4 = torch.softmax(stacker4(x_val4), dim=1).detach().numpy()

    gate4 = torch.nn.Linear(16, 4)
    torch.nn.init.zeros_(gate4.weight); torch.nn.init.zeros_(gate4.bias)
    optimizer = torch.optim.Adam(gate4.parameters(), lr=FUSION_LR)
    for _ in range(FUSION_STEPS):
        optimizer.zero_grad(set_to_none=True)
        gate_weights = torch.softmax(gate4(x_train4), dim=1)
        mixed = torch.sum(gate_weights.unsqueeze(-1) * experts_train4, dim=1).clamp_min(1e-8)
        loss = -torch.sum(w_train * torch.log(mixed[torch.arange(len(y_train)), y_train])) + FUSION_L2 * gate4.weight.square().mean()
        loss.backward(); optimizer.step()
    with torch.no_grad():
        gate_weights_val4 = torch.softmax(gate4(x_val4), dim=1)
        gate_val_prob4 = torch.sum(gate_weights_val4.unsqueeze(-1) * experts_val4, dim=1).numpy()

    fusion_probs = {
        "complex_gram": (expert_train[0], expert_val[0]),
        "learned_power": (expert_train[1], expert_val[1]),
        "window_sensor_rms_complex_gram": (expert_train[2], expert_val[2]),
        "v7_group_krr": (v7_train, v7_val),
        "equal_soft_mean_3_experts": (train_stack.mean(axis=1), val_stack.mean(axis=1)),
        "linear_stack": (torch.softmax(stacker(x_train), dim=1).detach().numpy(), stack_val_prob),
        "linear_gate": (torch.sum(torch.softmax(gate(x_train), dim=1).unsqueeze(-1) * expert_train_t, dim=1).detach().numpy(), gate_val_prob),
        "equal_soft_mean_4_experts": (train_stack4.mean(axis=1), val_stack4.mean(axis=1)),
        "linear_stack_4_experts": (torch.softmax(stacker4(x_train4), dim=1).detach().numpy(), stack_val_prob4),
        "linear_gate_4_experts": (torch.sum(torch.softmax(gate4(x_train4), dim=1).unsqueeze(-1) * experts_train4, dim=1).detach().numpy(), gate_val_prob4),
    }
    result: dict[str, Any] = {
        "format": "c170-oof-fusion-v18-summary",
        "protocol": {"folds": 11, "seeds_per_fold": list(SEEDS), "expert_count": 3,
                     "training_samples": int(len(train_idx)), "old_validation_samples": int(len(val_idx)),
                     "fusion_steps": FUSION_STEPS, "fusion_lr": FUSION_LR,
                     "fusion_l2_mean_weight_square": FUSION_L2,
                     "old_validation_used_for_training_or_selection": False,
                     "registered_test_touched": False},
        "metrics": {},
        "fusion_parameters": {
            "linear_stack_weight": stacker.weight.detach().numpy().tolist(),
            "linear_stack_bias": stacker.bias.detach().numpy().tolist(),
            "linear_gate_weight": gate.weight.detach().numpy().tolist(),
            "linear_gate_bias": gate.bias.detach().numpy().tolist(),
            "linear_stack_4_weight": stacker4.weight.detach().numpy().tolist(),
            "linear_stack_4_bias": stacker4.bias.detach().numpy().tolist(),
            "linear_gate_4_weight": gate4.weight.detach().numpy().tolist(),
            "linear_gate_4_bias": gate4.bias.detach().numpy().tolist(),
        },
    }
    for name, (train_prob, val_prob) in fusion_probs.items():
        train_pred = np.asarray(train_prob).argmax(axis=1)
        val_pred = np.asarray(val_prob).argmax(axis=1)
        result["metrics"][name] = {
            "oof_train": v2._classification_metrics(y_train_np, train_pred),
            "old_validation": v2._classification_metrics(y_val_np, val_pred),
            "old_validation_interpretation": "exploratory; not used to select models, seeds, folds, or fusion hyperparameters",
        }
    np.savez_compressed(
        root / "fusion_predictions.npz",
        train_dataset_index=np.asarray(data["sample_index"])[train_idx].astype(np.int64),
        train_labels=y_train_np.astype(np.int64),
        oldval_dataset_index=np.asarray(data["sample_index"])[val_idx].astype(np.int64),
        oldval_labels=y_val_np.astype(np.int64),
        expert_oof_probabilities=train_stack.astype(np.float32),
        expert_oldval_probabilities=val_stack.astype(np.float32),
        v7_oof_probabilities=v7_train.astype(np.float32),
        v7_oldval_probabilities=v7_val.astype(np.float32),
        expert_oof_probabilities_4=train_stack4.astype(np.float32),
        expert_oldval_probabilities_4=val_stack4.astype(np.float32),
        stack_oldval_probabilities=stack_val_prob.astype(np.float32),
        gate_oldval_probabilities=gate_val_prob.astype(np.float32),
    )
    atomic_json(root / "v18_fusion_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run-one")
    one.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    one.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    one.add_argument("--expert", choices=EXPERTS, required=True)
    one.add_argument("--position", type=int, required=True)
    one.add_argument("--seed", type=int, choices=SEEDS, required=True)
    one.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    summary.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    try:
        if args.command == "run-one":
            run_one(args.root.resolve(), args.dataset.resolve(), args.expert, args.position, args.seed, args.device)
        else:
            print(json.dumps(_fit_fusions(args.root.resolve(), args.dataset.resolve()), ensure_ascii=False, indent=2))
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
