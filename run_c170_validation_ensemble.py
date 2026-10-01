from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_experiment as c170

DEFAULT_BASELINE_OUTPUT_DIR = PROJECT_DIR / "outputs" / "c170_distance_holdout_20260927"
DEFAULT_CANDIDATE_OUTPUT_DIR = PROJECT_DIR / "outputs" / "c170_distance_holdout_candidates_20260927"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "c170_distance_holdout_ensemble_20260927"
DEFAULT_VALIDATION_DISTANCES = (5, 40, 90, 130)
SOURCE_FOR_MODEL_KIND = {
    "raw": "baseline",
    "orbit": "baseline",
    "harmonic": "candidates",
    "orbit_prob": "candidates",
}
ENSEMBLE_AGGREGATION = "equal_mean_softmax_probabilities"
ENSEMBLE_DEFINITIONS = {
    "orbit_prob_5seed": ("orbit_prob",),
    "three_model_15seed": ("orbit", "orbit_prob", "harmonic"),
}


def ensemble_definitions() -> dict[str, tuple[str, ...]]:
    return dict(ENSEMBLE_DEFINITIONS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_config(path: Path) -> dict[str, object]:
    config_path = Path(path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("phase") != "validation-only":
        raise ValueError(f"Source configuration is not validation-only: {config_path}")
    if not config.get("confirm_physical_order_flag"):
        raise ValueError(f"Source configuration lacks physical sensor-order confirmation: {config_path}")
    if tuple(config.get("seeds", ())) != c170.DEFAULT_SEEDS:
        raise ValueError(f"Source seeds differ from the fixed experiment: {config_path}")
    if config.get("split_distance_mm", {}).get("val") != list(DEFAULT_VALIDATION_DISTANCES):
        raise ValueError(f"Source validation positions differ from the fixed experiment: {config_path}")
    config["_config_path"] = str(config_path.resolve())
    config["_config_sha256"] = file_sha256(config_path)
    return config


def read_source_configs(source_outputs: dict[str, Path]) -> dict[str, dict[str, object]]:
    if set(source_outputs) != {"baseline", "candidates"}:
        raise ValueError("source_outputs must contain baseline and candidates")
    configs = {
        name: _read_source_config(path)
        for name, path in source_outputs.items()
    }
    baseline = configs["baseline"]
    candidates = configs["candidates"]
    for field in ("dataset_sha256", "dataset_path", "split_distance_mm", "direction_order"):
        if baseline[field] != candidates[field]:
            raise ValueError(f"The two validation-only configurations differ in {field}")
    if not {"orbit"}.issubset(baseline["model_kinds"]):
        raise ValueError("Baseline configuration lacks an orbit checkpoint")
    if not {"harmonic", "orbit_prob"}.issubset(candidates["model_kinds"]):
        raise ValueError("Candidate configuration lacks a harmonic or orbit_prob checkpoint")
    return configs


def validation_indices(
    dataset: dict[str, np.ndarray],
    validation_distances: Sequence[int],
) -> np.ndarray:
    distances = np.asarray(dataset["distance_mm"], dtype=np.int64)
    positions = tuple(int(value) for value in validation_distances)
    if positions != DEFAULT_VALIDATION_DISTANCES:
        raise ValueError("Validation positions differ from the fixed experiment")
    indices = np.flatnonzero(np.isin(distances, positions))
    if not len(indices) or set(distances[indices].tolist()) != set(positions):
        raise ValueError("C170 data lacks the requested validation positions")
    labels = np.asarray(dataset["labels"], dtype=np.int64)[indices]
    if not np.array_equal(np.unique(labels), np.arange(len(c170.DIRECTIONS))):
        raise ValueError("Validation samples do not contain all four direction classes")
    return indices


def load_member_checkpoints(
    source_outputs: dict[str, Path],
    ensemble_name: str,
    experiment_name: str,
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    definitions = ensemble_definitions()
    if ensemble_name not in definitions:
        raise ValueError(f"Unknown ensemble_name: {ensemble_name}")
    selected_seeds = tuple(int(seed) for seed in seeds)
    if selected_seeds != c170.DEFAULT_SEEDS:
        raise ValueError("The ensemble must use the five fixed seeds")
    if experiment_name not in {"true_training_labels", "shuffled_training_labels"}:
        raise ValueError(f"Unknown experiment_name: {experiment_name}")

    configs = read_source_configs(source_outputs)
    expected_dataset_sha256 = str(configs["baseline"]["dataset_sha256"])
    members: list[dict[str, object]] = []
    for model_kind in definitions[ensemble_name]:
        source_name = SOURCE_FOR_MODEL_KIND[model_kind]
        source_output = Path(source_outputs[source_name]).resolve()
        config = configs[source_name]
        records = config["candidate_selection"]["all_checkpoints"]
        matched = [
            record
            for record in records
            if record["experiment"] == experiment_name
            and record["model_kind"] == model_kind
        ]
        if {int(record["seed"]) for record in matched} != set(selected_seeds):
            raise ValueError(f"Incomplete checkpoint seeds for {model_kind} in {source_name}")
        if len(matched) != len(selected_seeds):
            raise ValueError(f"Duplicate checkpoint records for {model_kind} in {source_name}")
        for record in sorted(matched, key=lambda item: int(item["seed"])):
            checkpoint_path = source_output / str(record["path"])
            checkpoint_sha256 = file_sha256(checkpoint_path)
            model, payload = c170.load_checkpoint(checkpoint_path)
            if payload["experiment_name"] != experiment_name:
                raise ValueError(f"Checkpoint label condition differs from the source configuration: {checkpoint_path}")
            if payload["model_kind"] != model_kind:
                raise ValueError(f"Checkpoint model kind differs from the source configuration: {checkpoint_path}")
            if payload["architecture"] != c170.model_architecture(model_kind):
                raise ValueError(f"Checkpoint architecture differs from the source configuration: {checkpoint_path}")
            if int(payload["training_hyperparameters"]["seed"]) != int(record["seed"]):
                raise ValueError(f"Checkpoint seed differs from the source configuration: {checkpoint_path}")
            if payload["dataset_sha256"] != expected_dataset_sha256:
                raise ValueError(f"Checkpoint dataset SHA-256 differs from the source configuration: {checkpoint_path}")
            if payload["split_distance_mm"]["val"] != list(DEFAULT_VALIDATION_DISTANCES):
                raise ValueError(f"Checkpoint validation positions differ from the fixed experiment: {checkpoint_path}")
            preprocessing = dict(payload["preprocessing"])
            if "model_input_transform" not in preprocessing:
                if source_name != "baseline" or model_kind not in {"raw", "orbit"}:
                    raise ValueError(
                        f"Checkpoint lacks model_input_transform: {checkpoint_path}"
                    )
                preprocessing["model_input_transform"] = "identity"
            expected_transform = (
                "harmonic_c4_invariants" if model_kind == "harmonic" else "identity"
            )
            if preprocessing["model_input_transform"] != expected_transform:
                raise ValueError(f"Checkpoint model_input_transform mismatch: {checkpoint_path}")
            members.append(
                {
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "checkpoint_sha256": checkpoint_sha256,
                    "source_output_dir": str(source_output),
                    "source_config_path": config["_config_path"],
                    "source_config_sha256": config["_config_sha256"],
                    "experiment_name": experiment_name,
                    "model_kind": model_kind,
                    "seed": int(record["seed"]),
                    "architecture": payload["architecture"],
                    "preprocessing": preprocessing,
                    "dataset_sha256": payload["dataset_sha256"],
                    "split_distance_mm": payload["split_distance_mm"],
                    "state_dict": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    },
                    "model": model,
                }
            )
    expected_count = len(definitions[ensemble_name]) * len(selected_seeds)
    if len(members) != expected_count:
        raise RuntimeError(f"Incorrect ensemble checkpoint count: {len(members)} != {expected_count}")
    return members


def save_ensemble_checkpoint(
    path: Path,
    ensemble_name: str,
    experiment_name: str,
    members: list[dict[str, object]],
    dataset_sha256_value: str,
    validation_distances: Sequence[int],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "c170-validation-ensemble-checkpoint",
        "format_version": 1,
        "ensemble_name": ensemble_name,
        "experiment_name": experiment_name,
        "aggregation": ENSEMBLE_AGGREGATION,
        "class_order": list(c170.DIRECTIONS),
        "validation_distance_mm": [int(value) for value in validation_distances],
        "dataset_sha256": dataset_sha256_value,
        "model_kinds": list(ensemble_definitions()[ensemble_name]),
        "members": [
            {
                key: member[key]
                for key in (
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "source_output_dir",
                    "source_config_path",
                    "source_config_sha256",
                    "experiment_name",
                    "model_kind",
                    "seed",
                    "architecture",
                    "preprocessing",
                    "dataset_sha256",
                    "split_distance_mm",
                    "state_dict",
                )
            }
            for member in members
        ],
    }
    with path.open("xb") as checkpoint_file:
        torch.save(payload, checkpoint_file)


def load_ensemble_checkpoint(path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "c170-validation-ensemble-checkpoint":
        raise ValueError(f"Unsupported ensemble checkpoint format: {path}")
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported ensemble checkpoint version: {path}")
    if payload.get("aggregation") != ENSEMBLE_AGGREGATION:
        raise ValueError(f"Ensemble checkpoint aggregation-rule mismatch: {path}")
    if payload.get("class_order") != list(c170.DIRECTIONS):
        raise ValueError(f"Ensemble checkpoint class-order mismatch: {path}")
    if payload.get("model_kinds") != list(ensemble_definitions()[payload["ensemble_name"]]):
        raise ValueError(f"Ensemble checkpoint model_kinds mismatch: {path}")
    members: list[dict[str, object]] = []
    for member_payload in payload["members"]:
        model_kind = str(member_payload["model_kind"])
        if member_payload["architecture"] != c170.model_architecture(model_kind):
            raise ValueError(f"Ensemble member architecture mismatch: {path}")
        model = c170.build_model(model_kind)
        model.load_state_dict(member_payload["state_dict"], strict=True)
        model.eval()
        members.append({**member_payload, "model": model})
    if len(members) != len(payload["model_kinds"]) * len(c170.DEFAULT_SEEDS):
        raise ValueError(f"Incorrect ensemble checkpoint member count: {path}")
    return members, payload


def predict_ensemble_probabilities(
    members: list[dict[str, object]],
    raw_windows: np.ndarray,
) -> np.ndarray:
    if not members:
        raise ValueError("Ensemble members must not be empty")
    windows = np.asarray(raw_windows)
    probabilities: list[np.ndarray] = []
    for member in members:
        inputs = c170.prepare_checkpoint_inputs(windows, member["preprocessing"])
        tensor = torch.from_numpy(inputs).permute(0, 2, 1)
        model = member["model"]
        model.eval()
        with torch.no_grad():
            member_probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()
        probabilities.append(member_probabilities)
    averaged = np.stack(probabilities, axis=0).mean(axis=0)
    if averaged.shape != (len(windows), len(c170.DIRECTIONS)):
        raise RuntimeError(f"Incorrect ensemble probability shape: {averaged.shape}")
    if not np.isfinite(averaged).all():
        raise ValueError("Ensemble probabilities contain non-finite values")
    if not np.allclose(averaged.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Ensemble probability rows do not sum to 1")
    return averaged


def create_output_directory(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json_exclusive(path: Path, value: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")


def _prediction_rows(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    probabilities: np.ndarray,
    ensemble_name: str,
    experiment_name: str,
) -> list[dict[str, object]]:
    predictions = probabilities.argmax(axis=1)
    rows: list[dict[str, object]] = []
    for offset, (index, predicted) in enumerate(zip(indices, predictions, strict=True)):
        mapping = ",".join(str(int(channel)) for channel in dataset["channel_map"][index])
        row: dict[str, object] = {
            "ensemble_name": ensemble_name,
            "experiment": experiment_name,
            "split": "val",
            "sample_index": int(dataset["sample_index"][index]),
            "block": int(dataset["block"][index]),
            "repetition": int(dataset["repetition"][index]),
            "source_repetition": int(dataset["source_repetition"][index]),
            "distance_mm": int(dataset["distance_mm"][index]),
            "channel_map": mapping,
            "truth": c170.DIRECTIONS[int(dataset["labels"][index])],
            "prediction": c170.DIRECTIONS[int(predicted)],
        }
        row.update(
            {
                f"probability_{direction}": float(probabilities[offset, class_id])
                for class_id, direction in enumerate(c170.DIRECTIONS)
            }
        )
        rows.append(row)
    return rows


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "ensemble_name",
        "experiment",
        "split",
        "sample_index",
        "block",
        "repetition",
        "source_repetition",
        "distance_mm",
        "channel_map",
        "truth",
        "prediction",
        *(f"probability_{direction}" for direction in c170.DIRECTIONS),
    )
    with path.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_validation_ensembles(
    dataset_path: Path,
    baseline_output_dir: Path,
    candidate_output_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    source_outputs = {
        "baseline": Path(baseline_output_dir),
        "candidates": Path(candidate_output_dir),
    }
    source_configs = read_source_configs(source_outputs)
    dataset_hash = c170.dataset_sha256(dataset_path)
    if dataset_hash != source_configs["baseline"]["dataset_sha256"]:
        raise ValueError("C170 dataset SHA-256 differs from the source configuration")
    dataset = c170.load_dataset(dataset_path)
    if c170.dataset_sha256(dataset_path) != dataset_hash:
        raise RuntimeError("dataset.npz changed while being loaded")
    val_indices = validation_indices(dataset, DEFAULT_VALIDATION_DISTANCES)
    c170.validate_channel_maps(
        dataset["channel_map"][val_indices],
        confirm_physical_order=True,
    )
    val_windows = dataset["X_raw"][val_indices]
    val_labels = dataset["labels"][val_indices]
    val_distances = dataset["distance_mm"][val_indices]
    val_channel_maps = dataset["channel_map"][val_indices]

    output_dir = create_output_directory(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=False)
    metric_runs: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []

    for ensemble_name in ensemble_definitions():
        for experiment_name in ("true_training_labels", "shuffled_training_labels"):
            members = load_member_checkpoints(
                source_outputs,
                ensemble_name,
                experiment_name,
                c170.DEFAULT_SEEDS,
            )
            probabilities = predict_ensemble_probabilities(members, val_windows)
            checkpoint_path = checkpoint_dir / f"{ensemble_name}__{experiment_name}.pt"
            save_ensemble_checkpoint(
                checkpoint_path,
                ensemble_name,
                experiment_name,
                members,
                dataset_hash,
                DEFAULT_VALIDATION_DISTANCES,
            )
            checkpoint_sha256 = file_sha256(checkpoint_path)
            reloaded_members, reloaded_payload = load_ensemble_checkpoint(checkpoint_path)
            if reloaded_payload["dataset_sha256"] != dataset_hash:
                raise ValueError(f"Ensemble checkpoint dataset SHA-256 mismatch: {checkpoint_path}")
            for member in reloaded_payload["members"]:
                if file_sha256(Path(member["checkpoint_path"])) != member["checkpoint_sha256"]:
                    raise ValueError(
                        f"Ensemble member SHA-256 differs from its source checkpoint: {member['checkpoint_path']}"
                    )
            reloaded_probabilities = predict_ensemble_probabilities(reloaded_members, val_windows)
            if not np.array_equal(probabilities, reloaded_probabilities):
                raise RuntimeError(f"Ensemble inference differs after checkpoint reload: {checkpoint_path}")

            predictions = probabilities.argmax(axis=1)
            metrics = c170.evaluation_metrics(
                val_labels,
                predictions,
                val_distances,
                val_channel_maps,
            )
            member_sources = [
                {
                    "checkpoint_path": member["checkpoint_path"],
                    "checkpoint_sha256": member["checkpoint_sha256"],
                    "model_kind": member["model_kind"],
                    "seed": member["seed"],
                    "source_config_path": member["source_config_path"],
                    "source_config_sha256": member["source_config_sha256"],
                    "model_input_transform": member["preprocessing"]["model_input_transform"],
                    "training_derived_sensor_scales": member["preprocessing"][
                        "training_derived_sensor_scales"
                    ],
                    "sample_rate_hz": member["preprocessing"]["sample_rate_hz"],
                    "filter_band_hz": member["preprocessing"]["filter_band_hz"],
                }
                for member in members
            ]
            metric_runs.append(
                {
                    "ensemble_name": ensemble_name,
                    "experiment": experiment_name,
                    "split": "val",
                    "aggregation": ENSEMBLE_AGGREGATION,
                    "model_kinds": list(ensemble_definitions()[ensemble_name]),
                    "member_count": len(members),
                    "seeds": list(c170.DEFAULT_SEEDS),
                    "ensemble_checkpoint_path": str(checkpoint_path.resolve()),
                    "ensemble_checkpoint_sha256": checkpoint_sha256,
                    "reloaded_probabilities_exact_match": True,
                    "member_checkpoints": member_sources,
                    "metrics": metrics,
                }
            )
            checkpoint_records.append(
                {
                    "ensemble_name": ensemble_name,
                    "experiment": experiment_name,
                    "path": str(checkpoint_path.relative_to(output_dir)),
                    "sha256": checkpoint_sha256,
                    "member_count": len(members),
                }
            )
            prediction_rows.extend(
                _prediction_rows(
                    dataset,
                    val_indices,
                    probabilities,
                    ensemble_name,
                    experiment_name,
                )
            )

    if c170.dataset_sha256(dataset_path) != dataset_hash:
        raise RuntimeError("dataset.npz changed during ensemble validation")
    source_records = {
        name: {
            "output_dir": str(Path(source_outputs[name]).resolve()),
            "config_path": source_configs[name]["_config_path"],
            "config_sha256": source_configs[name]["_config_sha256"],
        }
        for name in source_outputs
    }
    config = {
        "phase": "validation-only",
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_sha256": dataset_hash,
        "source_outputs": source_records,
        "validation_distance_mm": list(DEFAULT_VALIDATION_DISTANCES),
        "validation_sample_count": int(len(val_indices)),
        "direction_order": list(c170.DIRECTIONS),
        "seeds": list(c170.DEFAULT_SEEDS),
        "aggregation": ENSEMBLE_AGGREGATION,
        "candidate_definitions": {
            name: {
                "model_kinds": list(model_kinds),
                "checkpoint_count": len(model_kinds) * len(c170.DEFAULT_SEEDS),
            }
            for name, model_kinds in ensemble_definitions().items()
        },
        "label_conditions": ["true_training_labels", "shuffled_training_labels"],
        "ensemble_checkpoints": checkpoint_records,
    }
    metrics_output = {
        "phase": "validation-only",
        "runs": metric_runs,
        "prediction_count": len(prediction_rows),
    }
    _write_json_exclusive(output_dir / "config.json", config)
    _write_json_exclusive(output_dir / "validation_ensemble_metrics.json", metrics_output)
    _write_predictions(output_dir / "validation_ensemble_predictions.csv", prediction_rows)
    return {"config": config, "metrics": metrics_output}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C170 fixed validation ensemble evaluation")
    parser.add_argument("--dataset", type=Path, default=c170.DEFAULT_DATASET_PATH)
    parser.add_argument("--baseline-output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR)
    parser.add_argument("--candidate-output-dir", type=Path, default=DEFAULT_CANDIDATE_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_validation_ensembles(
        args.dataset,
        args.baseline_output_dir,
        args.candidate_output_dir,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "phase": result["config"]["phase"],
                "output_dir": str(args.output_dir),
                "prediction_count": result["metrics"]["prediction_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
