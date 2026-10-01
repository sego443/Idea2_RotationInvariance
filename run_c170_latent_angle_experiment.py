"""Run one validation-only C170 latent-angle training experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_experiment as c170
import run_c170_ring_experiment as ring
from src.c170_latent_angle import LatentAngleMLPBaseline, LatentAngleTemporalClassifier


TRAIN_DISTANCE_MM = (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)
VAL_DISTANCE_MM = (5, 40, 90, 130)
REGISTERED_TEST_DISTANCE_MM = (15, 25, 110, 150)
EXCLUDED_DISTANCE_MM = (70,)
SEEDS = (17, 29, 43, 61, 89)
N_ANGLES = 64
ANGLE_PERIOD_MM = 170.0
UPDATES_PER_EPOCH = 20
N_ACTIONS = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
ANGLE_WEIGHT = 0.2
SMOOTHNESS_WEIGHT = 0.001
SHUFFLE_SEED_OFFSET = 10_000

VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "mlp": {
        "model_class": "LatentAngleMLPBaseline",
        "include_rms": True,
        "angle_weight": 0.0,
        "smoothness_weight": 0.0,
    },
    "no_angle": {
        "model_class": "LatentAngleTemporalClassifier",
        "include_rms": True,
        "angle_weight": 0.0,
        "smoothness_weight": SMOOTHNESS_WEIGHT,
    },
    "full": {
        "model_class": "LatentAngleTemporalClassifier",
        "include_rms": True,
        "angle_weight": ANGLE_WEIGHT,
        "smoothness_weight": SMOOTHNESS_WEIGHT,
    },
    "no_rms": {
        "model_class": "LatentAngleTemporalClassifier",
        "include_rms": False,
        "angle_weight": ANGLE_WEIGHT,
        "smoothness_weight": SMOOTHNESS_WEIGHT,
    },
    "shuffle": {
        "model_class": "LatentAngleTemporalClassifier",
        "include_rms": True,
        "angle_weight": ANGLE_WEIGHT,
        "smoothness_weight": SMOOTHNESS_WEIGHT,
    },
}

CURVE_FIELDS = (
    "epoch",
    "optimizer_updates",
    "epoch_seconds",
    "elapsed_seconds",
    "train_action_loss",
    "train_angle_nll_unweighted",
    "train_angle_objective_contribution",
    "train_smoothness_loss",
    "train_total_data_loss",
    "train_total_objective",
    "val_cross_entropy",
    "val_macro_f1",
    "val_recall_up",
    "val_recall_down",
    "val_recall_left",
    "val_recall_right",
    "val_unique_predicted_classes",
    "val_angle_entropy",
    "val_angle_error_up_degrees",
    "val_angle_error_down_degrees",
    "val_angle_error_left_degrees",
    "val_angle_error_right_degrees",
    "val_joint_probability_sum_max_abs_error",
    "val_f1_by_position_json",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one C170 latent-angle training seed. This entry point computes train/validation "
            "metrics only; it has no final-test phase."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--variant", choices=tuple(VARIANT_CONFIG), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dataset", type=Path, default=c170.DEFAULT_DATASET_PATH)
    parser.add_argument("--confirm-physical-order", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_jsonable,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(_jsonable(value), output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: list[dict[str, object]]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb+") as saved:
        os.fsync(saved.fileno())
    os.replace(temporary, path)


def _position_weights(train_positions: Sequence[int]) -> dict[int, float]:
    positions = sorted({int(value) for value in train_positions})
    if len(positions) < 2:
        raise ValueError("at least two train positions are required for circular weighting")
    lengths: dict[int, float] = {}
    for offset, position in enumerate(positions):
        previous = positions[offset - 1]
        following = positions[(offset + 1) % len(positions)]
        previous_gap = (position - previous) % ANGLE_PERIOD_MM
        next_gap = (following - position) % ANGLE_PERIOD_MM
        lengths[position] = 0.5 * (previous_gap + next_gap)
    total = sum(lengths.values())
    if not math.isclose(total, ANGLE_PERIOD_MM, rel_tol=0.0, abs_tol=1e-7):
        raise RuntimeError(f"circular position weights sum to {total}, expected {ANGLE_PERIOD_MM}")
    return {position: length / total for position, length in lengths.items()}


def _training_groups(
    train_indices: np.ndarray,
    original_labels: np.ndarray,
    distances: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for index in train_indices.tolist():
        key = (int(distances[index]), int(original_labels[index]))
        grouped.setdefault(key, []).append(int(index))

    expected_positions = set(TRAIN_DISTANCE_MM)
    if {position for position, _ in grouped} != expected_positions:
        raise ValueError("sampler groups do not cover the fixed train positions")
    for position in TRAIN_DISTANCE_MM:
        labels_at_position = {label for p, label in grouped if p == position}
        if labels_at_position != set(range(N_ACTIONS)):
            raise ValueError(f"position {position} mm must contain all four original labels")

    sizes = sorted(len(values) for values in grouped.values())
    if len(grouped) != 44 or sizes.count(19) != 1 or sizes.count(20) != 43:
        raise ValueError(f"expected 44 original-label groups sized 19/20, got {sizes}")
    return {key: np.asarray(values, dtype=np.int64) for key, values in sorted(grouped.items())}


def _training_sample_weights(
    groups: dict[tuple[int, int], np.ndarray],
) -> tuple[dict[int, float], dict[int, float]]:
    position_weights = _position_weights(TRAIN_DISTANCE_MM)
    sample_weights: dict[int, float] = {}
    for (position, _original_label), indices in groups.items():
        value = position_weights[position] / (N_ACTIONS * len(indices))
        for index in indices.tolist():
            sample_weights[int(index)] = value
    if not math.isclose(sum(sample_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("training sample weights must sum to one")
    return position_weights, sample_weights


def _batches_for_epoch(
    groups: dict[tuple[int, int], np.ndarray],
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    """Build a no-replacement epoch from original distance/label groups."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    keys = sorted(groups)
    order = rng.permutation(len(keys))
    shuffled_groups = {keys[int(key_index)]: rng.permutation(groups[keys[int(key_index)]]) for key_index in order}
    max_size = max(len(values) for values in shuffled_groups.values())
    batches: list[np.ndarray] = []
    for step in range(max_size):
        batch = np.asarray(
            [values[step] for values in shuffled_groups.values() if step < len(values)],
            dtype=np.int64,
        )
        rng.shuffle(batch)
        batches.append(batch)

    flattened = np.concatenate(batches)
    expected = np.concatenate(list(groups.values()))
    if len(batches) != UPDATES_PER_EPOCH:
        raise RuntimeError(f"sampler produced {len(batches)} updates, expected {UPDATES_PER_EPOCH}")
    if [len(batch) for batch in batches] != [44] * 19 + [43]:
        raise RuntimeError("sampler must produce 19 batches of 44 and a final batch of 43")
    if len(flattened) != len(expected) or len(np.unique(flattened)) != len(expected):
        raise RuntimeError("sampler must use each training sample exactly once per epoch")
    if set(flattened.tolist()) != set(expected.tolist()):
        raise RuntimeError("sampler output does not match the training index set")
    return batches


def _label_shuffle_mapping(
    dataset: dict[str, Any],
    train_indices: np.ndarray,
    shuffled_labels: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for block in sorted(np.unique(dataset["block"][train_indices]).tolist()):
        indices = train_indices[dataset["block"][train_indices] == block]
        original = np.unique(dataset["labels"][indices])
        assigned = np.unique(shuffled_labels[indices])
        if len(original) != 1 or len(assigned) != 1:
            raise ValueError(f"training block {int(block)} has non-unique direction labels")
        rows.append(
            {
                "distance_mm": int(dataset["distance_mm"][indices[0]]),
                "block": int(block),
                "original_direction": c170.DIRECTIONS[int(original[0])],
                "training_target_direction": c170.DIRECTIONS[int(assigned[0])],
            }
        )
    return rows


def _new_model(
    variant: str,
    rms_mean: Sequence[float],
    rms_std: Sequence[float],
) -> torch.nn.Module:
    config = VARIANT_CONFIG[variant]
    if config["model_class"] == "LatentAngleMLPBaseline":
        return LatentAngleMLPBaseline(
            include_rms=bool(config["include_rms"]),
            rms_mean=rms_mean,
            rms_std=rms_std,
        )
    return LatentAngleTemporalClassifier(
        num_angles=N_ANGLES,
        include_rms=bool(config["include_rms"]),
        rms_mean=rms_mean,
        rms_std=rms_std,
    )


def _validation_metrics(
    model: torch.nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    val_indices: np.ndarray,
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        outputs = model(val_inputs)
        action_log_probs = outputs["action_log_probs"].float()
        targets = val_targets.long()
        per_sample_nll = -action_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        probabilities = action_log_probs.exp().cpu().numpy().astype(np.float32)
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        diagnostics = {"cross_entropy": float(per_sample_nll.mean().cpu())}
        if isinstance(model, LatentAngleTemporalClassifier):
            required_angle_outputs = {
                "joint_log_probs",
                "angle_log_probs",
                "conditional_angle_log_probs",
            }
            missing_angle_outputs = required_angle_outputs.difference(outputs)
            if missing_angle_outputs:
                raise RuntimeError(
                    "angle-template model omitted required validation outputs: "
                    f"{sorted(missing_angle_outputs)}"
                )
            joint = outputs["joint_log_probs"].float()
            joint_sums = joint.exp().sum(dim=(1, 2))
            max_joint_sum_error = (joint_sums - 1.0).abs().max()

            angle_log_probs = outputs["angle_log_probs"].float()
            angle_probs = angle_log_probs.exp()
            angle_entropy = -(angle_probs * angle_log_probs).sum(dim=1).mean()
            true_conditionals = outputs["conditional_angle_log_probs"].float().gather(
                1,
                targets.view(-1, 1, 1).expand(-1, 1, N_ANGLES),
            ).squeeze(1)
            angle_index = true_conditionals.argmax(dim=1)
            theta_grid = torch.arange(N_ANGLES, dtype=torch.float32, device=val_inputs.device)
            theta_grid = theta_grid * (2.0 * math.pi / N_ANGLES)
            estimated_angle = theta_grid[angle_index]
            nominal_distance = dataset["distance_mm"][val_indices].astype(np.float32)
            nominal_angle = torch.from_numpy(nominal_distance).to(
                device=val_inputs.device, dtype=torch.float32
            ) * (2.0 * math.pi / ANGLE_PERIOD_MM)
            angle_error_degrees = torch.atan2(
                torch.sin(estimated_angle - nominal_angle),
                torch.cos(estimated_angle - nominal_angle),
            ).abs() * (180.0 / math.pi)
            angle_error_values = torch.stack(
                [angle_error_degrees[targets == action].mean() for action in range(N_ACTIONS)]
            )
            summary_values = torch.stack(
                [max_joint_sum_error, angle_entropy, *angle_error_values]
            ).cpu().tolist()
            diagnostics.update(
                {
                    "angle_diagnostics_applicable": True,
                    "mean_angle_posterior_entropy_nats": float(summary_values[1]),
                    "mean_absolute_nominal_angle_error_degrees_by_action": {
                        c170.DIRECTIONS[action]: float(summary_values[2 + action])
                        for action in range(N_ACTIONS)
                    },
                    "joint_probability_sum_max_abs_error": float(summary_values[0]),
                }
            )
        else:
            if not isinstance(model, LatentAngleMLPBaseline):
                raise TypeError(f"unsupported validation model type: {type(model).__name__}")
            unexpected_angle_outputs = {
                "joint_log_probs",
                "angle_log_probs",
                "conditional_angle_log_probs",
            }.intersection(outputs)
            if unexpected_angle_outputs:
                raise RuntimeError(
                    "MLP validation model unexpectedly returned angle outputs: "
                    f"{sorted(unexpected_angle_outputs)}"
                )
            diagnostics.update(
                {
                    "angle_diagnostics_applicable": False,
                    "mean_angle_posterior_entropy_nats": None,
                    "mean_absolute_nominal_angle_error_degrees_by_action": {
                        direction: None for direction in c170.DIRECTIONS
                    },
                    "joint_probability_sum_max_abs_error": None,
                }
            )

    true_labels = val_targets.detach().cpu().numpy().astype(np.int64)
    metrics = c170.evaluation_metrics(
        true_labels,
        predictions,
        dataset["distance_mm"][val_indices],
        dataset["channel_map"][val_indices],
    )
    return metrics, diagnostics, probabilities, predictions


def _validation_128_diagnostic(
    model: LatentAngleTemporalClassifier,
    val_inputs: torch.Tensor,
    probabilities_64: np.ndarray,
    true_labels: np.ndarray,
    dataset: dict[str, Any],
    val_indices: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        outputs = model(val_inputs)
        theta_128 = torch.arange(128, dtype=torch.float32, device=val_inputs.device)
        theta_128 = theta_128 * (2.0 * math.pi / 128)
        curves = model.template_curves(theta_128)
        templates = F.normalize(curves.float(), p=2.0, dim=-1, eps=1e-8)
        scores = 16.0 * torch.einsum(
            "bd,cjd->bcj", outputs["normalized_features"].float(), templates
        )
        joint = F.log_softmax(scores.flatten(1), dim=1).reshape(len(val_inputs), 4, 128)
        action_log_probs = torch.logsumexp(joint, dim=2)
        probabilities_128 = action_log_probs.exp().cpu().numpy().astype(np.float32)
        max_probability_difference = float(
            np.max(np.abs(probabilities_128 - probabilities_64))
        )
        predictions_128 = probabilities_128.argmax(axis=1).astype(np.int64)

    metrics_64 = c170.evaluation_metrics(
        true_labels,
        probabilities_64.argmax(axis=1),
        dataset["distance_mm"][val_indices],
        dataset["channel_map"][val_indices],
    )
    metrics_128 = c170.evaluation_metrics(
        true_labels,
        predictions_128,
        dataset["distance_mm"][val_indices],
        dataset["channel_map"][val_indices],
    )
    return {
        "purpose": "validation numerical-stability diagnostic only; not used for checkpoint or model selection",
        "action_argmax_agreement_fraction": float(
            np.mean(predictions_128 == probabilities_64.argmax(axis=1))
        ),
        "max_absolute_action_probability_difference": max_probability_difference,
        "macro_f1_64": float(metrics_64["overall"]["macro_f1"]),
        "macro_f1_128": float(metrics_128["overall"]["macro_f1"]),
        "macro_f1_difference_128_minus_64": float(
            metrics_128["overall"]["macro_f1"] - metrics_64["overall"]["macro_f1"]
        ),
        "f1_comparison_used_for_selection": False,
    }


def _prediction_rows(
    dataset: dict[str, Any],
    val_indices: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    variant: str,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for offset, (index, prediction) in enumerate(zip(val_indices, predictions, strict=True)):
        rows.append(
            {
                "variant": variant,
                "seed": int(seed),
                "split": "val",
                "sample_index": int(dataset["sample_index"][index]),
                "block": int(dataset["block"][index]),
                "repetition": int(dataset["repetition"][index]),
                "distance_mm": int(dataset["distance_mm"][index]),
                "channel_map": ",".join(
                    str(int(channel)) for channel in dataset["channel_map"][index]
                ),
                "truth": str(dataset["direction"][index]),
                "prediction": c170.DIRECTIONS[int(prediction)],
                "p_up": float(probabilities[offset, 0]),
                "p_down": float(probabilities[offset, 1]),
                "p_left": float(probabilities[offset, 2]),
                "p_right": float(probabilities[offset, 3]),
            }
        )
    return rows


def _curve_row(
    *,
    epoch: int,
    epoch_seconds: float,
    elapsed_seconds: float,
    action_loss: float,
    angle_nll: float | None,
    angle_weight: float,
    smoothness_loss: float | None,
    smoothness_weight: float,
    val_cross_entropy: float,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
) -> dict[str, object]:
    overall = metrics["overall"]
    recalls = overall["per_class_recall"]
    f1_by_position = {
        position: values["macro_f1"]
        for position, values in metrics["by_distance_mm"].items()
    }
    angle_errors = diagnostics["mean_absolute_nominal_angle_error_degrees_by_action"]
    data_loss = action_loss + angle_weight * angle_nll if angle_nll is not None else action_loss
    total_objective = (
        data_loss + smoothness_weight * smoothness_loss
        if smoothness_loss is not None
        else data_loss
    )
    return {
        "epoch": epoch + 1,
        "optimizer_updates": UPDATES_PER_EPOCH,
        "epoch_seconds": epoch_seconds,
        "elapsed_seconds": elapsed_seconds,
        "train_action_loss": action_loss,
        "train_angle_nll_unweighted": angle_nll,
        "train_angle_objective_contribution": (
            angle_weight * angle_nll if angle_nll is not None else None
        ),
        "train_smoothness_loss": smoothness_loss,
        "train_total_data_loss": data_loss,
        "train_total_objective": total_objective,
        "val_cross_entropy": val_cross_entropy,
        "val_macro_f1": overall["macro_f1"],
        "val_recall_up": recalls[0],
        "val_recall_down": recalls[1],
        "val_recall_left": recalls[2],
        "val_recall_right": recalls[3],
        "val_unique_predicted_classes": overall["prediction_unique_class_count"],
        "val_angle_entropy": diagnostics["mean_angle_posterior_entropy_nats"],
        "val_angle_error_up_degrees": angle_errors["UP"],
        "val_angle_error_down_degrees": angle_errors["DOWN"],
        "val_angle_error_left_degrees": angle_errors["LEFT"],
        "val_angle_error_right_degrees": angle_errors["RIGHT"],
        "val_joint_probability_sum_max_abs_error": diagnostics[
            "joint_probability_sum_max_abs_error"
        ],
        "val_f1_by_position_json": json.dumps(f1_by_position, sort_keys=True),
    }


def _collect_code_hashes() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "model": PROJECT_DIR / "src" / "c170_latent_angle.py",
        "dataset_helpers": PROJECT_DIR / "run_c170_experiment.py",
        "preprocessing_helpers": PROJECT_DIR / "run_c170_ring_experiment.py",
        "encoder_helpers": PROJECT_DIR / "src" / "rotation_poc.py",
    }
    return {name: _file_sha256(path) for name, path in files.items()}


def _manifest_base(
    config: dict[str, Any],
    config_sha256: str,
    runner_started_utc: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    return {
        "format": "c170-latent-angle-run",
        "format_version": 1,
        "status": "running",
        "config": config,
        "config_sha256": config_sha256,
        "timing": {
            "runner_started_utc": runner_started_utc,
            "preprocessing_seconds": None,
            "training_seconds": None,
            "total_elapsed_seconds": None,
        },
        "metadata": {
            "variant": variant,
            "seed": int(seed),
            "torch_version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "test_metrics_computed": False,
        },
        "preprocessing": {},
        "selection": {},
        "diagnostics": {},
        "provenance": {"code_sha256": _collect_code_hashes()},
        "artifacts": {},
        "test_access": {
            "loaded_as_part_of_complete_npz": True,
            "used_as_model_input": False,
            "used_to_estimate_statistics": False,
            "predicted_or_scored": False,
        },
        "test_metrics_computed": False,
    }


def _validate_and_prepare(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, float],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    dataset = c170.load_dataset(args.dataset)
    assignments = {
        "train": TRAIN_DISTANCE_MM,
        "val": VAL_DISTANCE_MM,
        "test": REGISTERED_TEST_DISTANCE_MM,
    }
    splits = c170.split_by_distance(dataset["distance_mm"], assignments)
    observed_positions = set(np.unique(dataset["distance_mm"]).tolist())
    expected_positions = set(TRAIN_DISTANCE_MM + VAL_DISTANCE_MM + REGISTERED_TEST_DISTANCE_MM)
    expected_positions.update(EXCLUDED_DISTANCE_MM)
    if observed_positions != expected_positions:
        raise ValueError(
            f"dataset distance positions differ from the frozen plan: "
            f"observed={sorted(observed_positions)}, expected={sorted(expected_positions)}"
        )
    counts = {name: int(len(indices)) for name, indices in splits.items()}
    if counts != {"train": 879, "val": 320, "test": 320}:
        raise ValueError(f"expected split counts 879/320/320, got {counts}")

    train_val_indices = np.concatenate([splits["train"], splits["val"]])
    train_val_metadata = {
        name: dataset[name][train_val_indices]
        for name in (
            "block",
            "distance_mm",
            "labels",
            "channel_map",
        )
    }
    c170.assert_recording_groups_intact(train_val_metadata)
    channel_map_confirmation = c170.validate_channel_maps(
        dataset["channel_map"],
        confirm_physical_order=bool(args.confirm_physical_order),
    )

    normalized, sensor_scales = c170.prepare_model_inputs(
        dataset["X_raw"],
        {"train": splits["train"], "val": splits["val"]},
    )
    for split_name in ("train", "val"):
        if not np.isfinite(normalized[split_name]).all():
            raise ValueError(f"preprocessed {split_name} inputs contain non-finite values")

    train_inputs = torch.from_numpy(normalized["train"]).to(device=device, dtype=torch.float32)
    val_inputs = torch.from_numpy(normalized["val"]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        train_log_rms = 0.5 * torch.log(
            train_inputs.square().mean(dim=1).clamp_min(1e-12)
        )
        rms_mean_tensor = train_log_rms.mean(dim=0)
        rms_std_tensor = train_log_rms.std(dim=0, unbiased=False).clamp_min(1e-6)
    rms_mean = [float(value) for value in rms_mean_tensor.cpu().tolist()]
    rms_std = [float(value) for value in rms_std_tensor.cpu().tolist()]

    groups = _training_groups(
        splits["train"],
        dataset["labels"],
        dataset["distance_mm"],
    )
    position_weights, sample_weights = _training_sample_weights(groups)
    train_weights_np = np.asarray(
        [sample_weights[int(index)] for index in splits["train"]], dtype=np.float32
    )
    if not math.isclose(float(train_weights_np.sum()), 1.0, rel_tol=0.0, abs_tol=2e-7):
        raise ValueError("training tensor weights do not sum to one")
    train_weights = torch.from_numpy(train_weights_np).to(device=device)
    train_labels_all = dataset["labels"].copy()
    shuffle_seed: int | None = None
    label_shuffle_mapping: list[dict[str, object]] = []
    if args.variant == "shuffle":
        shuffle_seed = int(args.seed) + SHUFFLE_SEED_OFFSET
        train_labels_all = c170.shuffle_training_labels_by_recording(
            dataset["labels"],
            dataset["distance_mm"],
            dataset["block"],
            splits["train"],
            seed=shuffle_seed,
        )
        label_shuffle_mapping = _label_shuffle_mapping(
            dataset,
            splits["train"],
            train_labels_all,
        )

    train_targets = torch.from_numpy(
        np.asarray(train_labels_all[splits["train"]], dtype=np.int64)
    ).to(device=device)
    val_targets = torch.from_numpy(
        np.asarray(dataset["labels"][splits["val"]], dtype=np.int64)
    ).to(device=device)
    train_angles = torch.from_numpy(
        (2.0 * np.pi * dataset["distance_mm"][splits["train"]].astype(np.float32)
         / ANGLE_PERIOD_MM).astype(np.float32)
    ).to(device=device)

    sample_rows = {
        name: {
            "count": int(len(indices)),
            "distance_mm": sorted(
                int(value) for value in np.unique(dataset["distance_mm"][indices]).tolist()
            ),
        }
        for name, indices in splits.items()
    }
    metadata = {
        "split_summary": sample_rows,
        "excluded_70mm_count": int(np.count_nonzero(dataset["distance_mm"] == 70)),
        "duplicate_rows_excluded_by_loader": int(dataset["duplicate_rows_excluded"]),
        "channel_map_confirmation": channel_map_confirmation,
        "channel_map_counts_by_split": {
            name: {
                ",".join(str(int(value)) for value in mapping): int(
                    np.count_nonzero(
                        np.all(dataset["channel_map"][indices] == np.asarray(mapping), axis=1)
                    )
                )
                for mapping in sorted(
                    {
                        tuple(int(value) for value in row)
                        for row in dataset["channel_map"][indices].tolist()
                    }
                )
            }
            for name, indices in splits.items()
        },
        "training_original_label_group_sizes": [
            {"distance_mm": key[0], "direction": c170.DIRECTIONS[key[1]], "n": len(value)}
            for key, value in groups.items()
        ],
        "position_weights": {str(key): value for key, value in position_weights.items()},
        "shuffle_seed": shuffle_seed,
        "label_shuffle_mapping": label_shuffle_mapping,
    }
    preprocessing = {
        "sample_rate_hz": 250,
        "filter_band_hz": [0.5, 40.0],
        "filter_type": "butterworth_bandpass_sosfiltfilt_order_3",
        "dc_removal": "subtract_per_trial_per_channel_temporal_mean",
        "sensor_scale_source": "training_split_median_absolute_centered_signal",
        "sensor_scales": [float(value) for value in sensor_scales],
        "log_rms_definition": "0.5*log(mean(calibrated_waveform**2,time).clamp_min(1e-12))",
        "log_rms_training_mean": rms_mean,
        "log_rms_training_std": rms_std,
        "log_rms_statistics_source": "training_split_only",
        "channel_order": ["S0", "S1", "S2", "S3"],
        "channel_map_as_model_input": False,
        "normalization_fit_splits": ["train"],
        "filtered_or_normalized_splits": ["train", "val"],
    }
    tensors = (
        train_inputs,
        val_inputs,
        train_targets,
        val_targets,
        train_weights,
        train_angles,
        torch.as_tensor(splits["train"], dtype=torch.long, device=device),
    )
    return (
        dataset,
        splits,
        normalized,
        metadata,
        preprocessing,
        *tensors,
        {"groups": groups, "sensor_scales": sensor_scales, "rms_mean": rms_mean, "rms_std": rms_std},
    )


def _run(args: argparse.Namespace) -> int:
    if args.max_epochs < 1 or args.patience < 1 or args.cpu_threads < 1:
        raise ValueError("max-epochs, patience, and cpu-threads must all be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if not args.confirm_physical_order:
        raise ValueError("pass --confirm-physical-order after confirming S0-S3 channel order")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    started_monotonic = time.monotonic()
    runner_started_utc = _utc_now()
    dataset_path = args.dataset.resolve()
    dataset_hash = _file_sha256(dataset_path)
    config = {
        "variant": args.variant,
        "seed": int(args.seed),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "cpu_threads": int(args.cpu_threads),
        "device": args.device,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "train_distance_mm": list(TRAIN_DISTANCE_MM),
        "val_distance_mm": list(VAL_DISTANCE_MM),
        "registered_test_distance_mm": list(REGISTERED_TEST_DISTANCE_MM),
        "excluded_distance_mm": list(EXCLUDED_DISTANCE_MM),
        "input_shape": [500, 4],
        "num_actions": N_ACTIONS,
        "num_angles": N_ANGLES,
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "batch_sizes_per_epoch": [44] * 19 + [43],
        "batch_loss_weight_multiplier": UPDATES_PER_EPOCH,
        "sampler_group_key": "distance_mm_and_original_direction_label",
        "sampler_rng": "numpy_seedsequence_seed_epoch",
        "sample_weight_rule": "position_voronoi_weight/(4*original_group_size)",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "angle_weight": VARIANT_CONFIG[args.variant]["angle_weight"],
        "smoothness_weight": VARIANT_CONFIG[args.variant]["smoothness_weight"],
        "angle_target_bandwidth": 8.0,
        "include_rms": bool(VARIANT_CONFIG[args.variant]["include_rms"]),
        "sampler": "original_distance_label_groups_no_replacement_19x44_plus_43",
        "label_shuffle_seed": (
            int(args.seed) + SHUFFLE_SEED_OFFSET if args.variant == "shuffle" else None
        ),
    }
    config_sha256 = _canonical_json_sha256(config)
    output_dir = args.output_dir.resolve()
    c170.create_output_directory(output_dir)
    manifest = _manifest_base(config, config_sha256, runner_started_utc, args.variant, args.seed)
    manifest["metadata"].update(
        {
            "actual_device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_driver_model": "nvidia-smi WDDM" if device.type == "cuda" else None,
        }
    )
    manifest_path = output_dir / "run_manifest.json"
    _atomic_write_json(manifest_path, manifest)

    try:
        torch.set_num_threads(args.cpu_threads)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats(device)

        preprocessing_start = time.monotonic()
        (
            dataset,
            splits,
            _normalized_cpu,
            metadata,
            preprocessing,
            train_inputs,
            val_inputs,
            train_targets,
            val_targets,
            train_weights,
            train_angles,
            train_global_indices,
            extra,
        ) = _validate_and_prepare(args, device)
        preprocessing_seconds = time.monotonic() - preprocessing_start
        manifest["metadata"].update(metadata)
        manifest["preprocessing"] = preprocessing
        manifest["timing"]["preprocessing_seconds"] = preprocessing_seconds
        _atomic_write_json(manifest_path, manifest)

        variant_config = VARIANT_CONFIG[args.variant]
        model = _new_model(args.variant, extra["rms_mean"], extra["rms_std"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        train_indices = splits["train"]
        val_indices = splits["val"]
        original_groups = extra["groups"]
        global_to_local = {int(index): offset for offset, index in enumerate(train_indices)}
        best_macro_f1 = -1.0
        best_cross_entropy = float("inf")
        best_epoch = -1
        bad_epochs = 0
        best_validation_predictions: np.ndarray | None = None
        best_validation_probabilities: np.ndarray | None = None
        best_validation_metrics: dict[str, Any] | None = None
        best_validation_diagnostics: dict[str, Any] | None = None
        curve_rows: list[dict[str, object]] = []
        training_started = time.monotonic()
        training_seconds = 0.0
        best_checkpoint_path = output_dir / "best_checkpoint.pt"

        for epoch in range(args.max_epochs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_started = time.monotonic()
            model.train()
            batches = _batches_for_epoch(original_groups, args.seed, epoch)
            action_accumulator = torch.zeros((), dtype=torch.float32, device=device)
            angle_accumulator = torch.zeros((), dtype=torch.float32, device=device)
            smoothness_accumulator = torch.zeros((), dtype=torch.float32, device=device)

            for batch_global in batches:
                local_np = np.asarray([global_to_local[int(index)] for index in batch_global], dtype=np.int64)
                local_indices = torch.from_numpy(local_np).to(device=device)
                samples = train_inputs[local_indices]
                actions = train_targets[local_indices]
                weights = train_weights[local_indices]
                optimizer.zero_grad(set_to_none=True)
                outputs = model(samples)
                if isinstance(model, LatentAngleTemporalClassifier):
                    terms = model.loss_terms(outputs, actions, train_angles[local_indices])
                    action_nll = terms["action_nll"]
                    angle_nll = terms["angle_nll"]
                    smoothness_loss = model.smoothness_loss()
                else:
                    log_probs = outputs["action_log_probs"].float()
                    action_nll = -log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                    angle_nll = torch.zeros_like(action_nll)
                    smoothness_loss = torch.zeros((), dtype=torch.float32, device=device)

                angle_weight = float(variant_config["angle_weight"])
                smoothness_weight = float(variant_config["smoothness_weight"])
                weighted_per_sample = action_nll + angle_weight * angle_nll
                batch_loss = (
                    UPDATES_PER_EPOCH * torch.sum(weights * weighted_per_sample)
                    + smoothness_weight * smoothness_loss
                )
                batch_loss.backward()
                optimizer.step()

                action_accumulator = action_accumulator + torch.sum(weights * action_nll).detach()
                angle_accumulator = angle_accumulator + torch.sum(weights * angle_nll).detach()
                smoothness_accumulator = smoothness_accumulator + smoothness_loss.detach()

            metrics, diagnostics, probabilities, predictions = _validation_metrics(
                model,
                val_inputs,
                val_targets,
                val_indices,
                dataset,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_seconds = time.monotonic() - epoch_started
            training_seconds = time.monotonic() - training_started
            elapsed_seconds = training_seconds
            action_loss, angle_nll_value, smoothness_value = (
                torch.stack(
                    (
                        action_accumulator,
                        angle_accumulator,
                        smoothness_accumulator / UPDATES_PER_EPOCH,
                    )
                )
                .detach()
                .cpu()
                .tolist()
            )

            overall = metrics["overall"]
            current_macro = float(overall["macro_f1"])
            current_cross_entropy = float(diagnostics["cross_entropy"])

            improved = current_macro > best_macro_f1 or (
                current_macro == best_macro_f1 and current_cross_entropy < best_cross_entropy
            )
            if improved:
                best_macro_f1 = current_macro
                best_cross_entropy = current_cross_entropy
                best_epoch = epoch
                bad_epochs = 0
                best_validation_predictions = predictions.copy()
                best_validation_probabilities = probabilities.copy()
                best_validation_metrics = metrics
                best_validation_diagnostics = diagnostics
                checkpoint_payload = {
                    "format": "c170-latent-angle-checkpoint",
                    "format_version": 1,
                    "variant": args.variant,
                    "seed": int(args.seed),
                    "best_epoch_zero_based": int(best_epoch),
                    "state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    },
                    "model_config": {
                        "class": variant_config["model_class"],
                        "num_angles": N_ANGLES if isinstance(model, LatentAngleTemporalClassifier) else None,
                        "include_rms": bool(variant_config["include_rms"]),
                        "rms_mean": extra["rms_mean"],
                        "rms_std": extra["rms_std"],
                    },
                    "class_order": list(c170.DIRECTIONS),
                    "preprocessing": preprocessing,
                    "selection": {
                        "criterion": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
                        "validation_macro_f1": best_macro_f1,
                        "validation_cross_entropy": best_cross_entropy,
                        "best_epoch_zero_based": best_epoch,
                    },
                    "experiment_config": config,
                    "experiment_config_sha256": config_sha256,
                    "code_sha256": manifest["provenance"]["code_sha256"],
                    "channel_map_confirmation": metadata["channel_map_confirmation"],
                    "label_shuffle_mapping": metadata["label_shuffle_mapping"],
                    "registered_test_metrics_computed": False,
                }
                _atomic_torch_save(best_checkpoint_path, checkpoint_payload)
            else:
                bad_epochs += 1

            has_angle_head = isinstance(model, LatentAngleTemporalClassifier)
            row = _curve_row(
                epoch=epoch,
                epoch_seconds=epoch_seconds,
                elapsed_seconds=elapsed_seconds,
                action_loss=float(action_loss),
                angle_nll=float(angle_nll_value) if has_angle_head else None,
                angle_weight=float(variant_config["angle_weight"]),
                smoothness_loss=float(smoothness_value) if has_angle_head else None,
                smoothness_weight=float(variant_config["smoothness_weight"]),
                val_cross_entropy=current_cross_entropy,
                metrics=metrics,
                diagnostics=diagnostics,
            )
            curve_rows.append(row)
            _atomic_write_csv(output_dir / "learning_curve.csv", CURVE_FIELDS, curve_rows)

            manifest["timing"].update(
                {
                    "training_seconds": training_seconds,
                    "total_elapsed_seconds": time.monotonic() - started_monotonic,
                }
            )
            manifest["selection"] = {
                "best_epoch_zero_based": best_epoch,
                "validation_macro_f1": best_macro_f1,
                "validation_cross_entropy": best_cross_entropy,
            }
            manifest["artifacts"] = {
                "checkpoint": best_checkpoint_path.name if best_epoch >= 0 else None,
                "learning_curve": "learning_curve.csv",
                "validation_predictions": "validation_predictions.csv" if best_epoch >= 0 else None,
            }
            _atomic_write_json(manifest_path, manifest)
            print(
                f"variant={args.variant} seed={args.seed} epoch={epoch + 1} "
                f"updates={UPDATES_PER_EPOCH} val_macro_f1={current_macro:.6f} "
                f"val_ce={current_cross_entropy:.6f} epoch_seconds={epoch_seconds:.3f}",
                flush=True,
            )
            if bad_epochs >= args.patience:
                break

        if best_epoch < 0 or best_validation_metrics is None or best_validation_predictions is None:
            raise RuntimeError("training completed without selecting a validation checkpoint")

        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            reloaded_outputs = model(val_inputs)
            reloaded_probabilities = reloaded_outputs["action_log_probs"].float().exp().cpu().numpy()
        reload_max_diff = float(np.max(np.abs(reloaded_probabilities - best_validation_probabilities)))
        if reload_max_diff > 1e-6:
            raise RuntimeError(
                f"checkpoint reload validation probabilities changed by {reload_max_diff:.3g}"
            )

        diagnostic_128: dict[str, Any] | None = None
        if args.variant == "full":
            diagnostic_128 = _validation_128_diagnostic(
                model,
                val_inputs,
                best_validation_probabilities,
                val_targets.detach().cpu().numpy().astype(np.int64),
                dataset,
                val_indices,
            )

        prediction_rows = _prediction_rows(
            dataset,
            val_indices,
            best_validation_predictions,
            best_validation_probabilities,
            args.variant,
            args.seed,
        )
        prediction_fields = tuple(prediction_rows[0].keys())
        _atomic_write_csv(
            output_dir / "validation_predictions.csv",
            prediction_fields,
            prediction_rows,
        )
        curve_sha256 = _file_sha256(output_dir / "learning_curve.csv")
        prediction_sha256 = _file_sha256(output_dir / "validation_predictions.csv")
        checkpoint_sha256 = _file_sha256(best_checkpoint_path)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory_bytes = None

        manifest["status"] = "completed"
        manifest["timing"].update(
            {
                "training_seconds": training_seconds,
                "total_elapsed_seconds": time.monotonic() - started_monotonic,
                "peak_cuda_memory_allocated_bytes": peak_memory_bytes,
                "completed_utc": _utc_now(),
            }
        )
        manifest["selection"] = {
            "best_epoch_zero_based": int(best_epoch),
            "validation_macro_f1": float(best_macro_f1),
            "validation_cross_entropy": float(best_cross_entropy),
            "validation_overall": best_validation_metrics["overall"],
            "validation_by_distance_mm": best_validation_metrics["by_distance_mm"],
            "validation_by_distance_mm_and_channel_map": best_validation_metrics[
                "by_distance_mm_and_channel_map"
            ],
        }
        manifest["diagnostics"] = {
            **(best_validation_diagnostics or {}),
            "checkpoint_reload_max_probability_difference": reload_max_diff,
            "integration_resolution_64_vs_128": diagnostic_128,
        }
        manifest["artifacts"] = {
            "checkpoint": {
                "path": best_checkpoint_path.name,
                "sha256": checkpoint_sha256,
            },
            "learning_curve": {"path": "learning_curve.csv", "sha256": curve_sha256},
            "validation_predictions": {
                "path": "validation_predictions.csv",
                "sha256": prediction_sha256,
            },
        }
        manifest["test_metrics_computed"] = False
        _atomic_write_json(manifest_path, manifest)
        manifest_sha256 = _file_sha256(manifest_path)
        complete_marker = {
            "format": "c170-latent-angle-complete",
            "format_version": 1,
            "status": "completed",
            "config_sha256": config_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_sha256,
            "selected_epoch": int(best_epoch),
            "test_metrics_computed": False,
        }
        _atomic_write_json(output_dir / "RUN_COMPLETE.json", complete_marker)
        print(
            f"RUN_COMPLETE variant={args.variant} seed={args.seed} "
            f"best_epoch={best_epoch} val_macro_f1={best_macro_f1:.6f} "
            f"total_updates={len(curve_rows) * UPDATES_PER_EPOCH}",
            flush=True,
        )
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["timing"]["total_elapsed_seconds"] = time.monotonic() - started_monotonic
        try:
            _atomic_write_json(manifest_path, manifest)
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
