"""One-shot evaluation of a frozen v21 stage on W02 wears.

Verifies the freeze manifest (checkpoint and scorer hashes), exclusively creates EVALUATION_STARTED.json, then
scores the body (non-prefix) strokes of every trial recording of the requested parts:
  E1  in-recording calibration: prototypes from the same recording's labeled C prefix
  E1b cross-recording calibration: prototypes from the prefixes of the other recordings of the same wear
  E2  causal unlabeled refinement: the recording's prefix (labels ignored) plus earlier body strokes
  E3  zero calibration
  E4  training-free comparator: nearest centroid on raw-waveform features from the same-recording prefix
Each endpoint uses the frozen family (backbone) and tau named in the manifest; matched-null models are scored on E1. Results include gate status, wear-cluster bootstrap intervals, per-wear and
per-quadrant F1, confusion matrix and prediction counts. Do not rerun on the same wears: a second run is not new
evidence. The output directory must not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from sklearn.metrics import confusion_matrix, f1_score

from src import v21_data as data_module
from src import v21_wear_calibrated as v21
from src.c170_shared_scale_preprocessing_v3 import calibrate_windows_shared_scale_v3

CLASSES = data_module.CLASSES
MIN_UNLABELED = 4


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bandpass(windows: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [0.5, 40], btype="bandpass", fs=250, output="sos")
    return signal.sosfiltfilt(sos, windows, axis=1).astype(np.float32)


@torch.no_grad()
def predict_one_model(checkpoint: dict, data: dict, taus: dict[str, float]) -> dict[str, np.ndarray]:
    model = v21.WearCalibratedClassifier(checkpoint["backbone"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    values, _ = calibrate_windows_shared_scale_v3(bandpass(data["X"]), scale=checkpoint["shared_scale"])
    embeddings = torch.cat([model.embed(torch.as_tensor(values[i:i + 256])) for i in range(0, len(values), 256)])
    prototypes = model.global_prototypes()
    labels = torch.as_tensor(data["y"])
    out = {mode: np.full(len(values), -1) for mode in ("E1", "E1b", "E2", "E3")}
    out["E3"] = v21.cosine_logits(embeddings, prototypes).argmax(1).numpy()
    for group in np.unique(data["group"]):
        in_group = data["group"] == group
        support = np.flatnonzero(in_group & data["is_calibration"])
        body = np.flatnonzero(in_group & ~data["is_calibration"])
        body = body[np.argsort(data["repetition"][body], kind="stable")]
        wear = data["wear_group"][in_group][0]
        cross = np.flatnonzero((data["wear_group"] == wear) & ~in_group & data["is_calibration"])
        calibrated = v21.calibrated_prototypes(prototypes, embeddings[support], labels[support], taus["E1"])
        out["E1"][body] = v21.cosine_logits(embeddings[body], calibrated).argmax(1).numpy()
        if len(cross):
            calibrated = v21.calibrated_prototypes(prototypes, embeddings[cross], labels[cross], taus["E1b"])
            out["E1b"][body] = v21.cosine_logits(embeddings[body], calibrated).argmax(1).numpy()
        else:
            out["E1b"][body] = out["E3"][body]
        for position, index in enumerate(body):
            unlabeled = np.concatenate([support, body[:position]])
            refined = (v21.transductive_prototypes(prototypes, embeddings[unlabeled], taus["E2"])
                       if len(unlabeled) >= MIN_UNLABELED else prototypes)
            out["E2"][index] = int(v21.cosine_logits(embeddings[index:index + 1], refined).argmax())
    return out


def raw_features(windows: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [0.5, 12], btype="bandpass", fs=250, output="sos")
    filtered = signal.sosfiltfilt(sos, windows, axis=1)
    filtered = (filtered - filtered.mean(axis=1, keepdims=True))[:, ::10, :]
    gain = np.sqrt((filtered ** 2).mean(axis=(1, 2), keepdims=True)) + 1e-12
    return np.concatenate([(filtered / gain).reshape(len(windows), -1), np.log(gain.reshape(-1, 1))], axis=1)


def nearest_centroid_in_recording(data: dict) -> np.ndarray:
    """E4: per recording, z-score with the prefix statistics and assign each body stroke to the nearest class mean."""
    features = raw_features(data["X"])
    predicted = np.full(len(features), -1)
    for group in np.unique(data["group"]):
        in_group = data["group"] == group
        support = np.flatnonzero(in_group & data["is_calibration"])
        body = np.flatnonzero(in_group & ~data["is_calibration"])
        mean, std = features[support].mean(axis=0), features[support].std(axis=0) + 1e-6
        z = (features - mean) / std
        centroids = np.stack([z[support][data["y"][support] == c].mean(axis=0) if np.any(data["y"][support] == c)
                              else np.full(z.shape[1], np.inf) for c in range(4)])
        distances = ((z[body][:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        predicted[body] = distances.argmin(axis=1)
    return predicted


def vote(predictions: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(predictions)
    return np.array([np.bincount(column, minlength=4).argmax() for column in stacked.T])


def macro_f1(true, predicted) -> float:
    return float(f1_score(true, predicted, average="macro", labels=list(range(4))))


def report(true: np.ndarray, predicted: np.ndarray, groups: np.ndarray, angles: np.ndarray, bootstrap: dict) -> dict:
    recall = {name: float(np.mean(predicted[true == c] == c)) for c, name in enumerate(CLASSES)}
    rng = np.random.default_rng(bootstrap["seed"])
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    samples = []
    for _ in range(bootstrap["resamples"]):
        index = np.concatenate([members[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        samples.append(macro_f1(true[index], predicted[index]))
    quadrant = np.where(angles >= 0, ((angles + 45) % 360) // 90, -1)
    return {
        "macro_f1": macro_f1(true, predicted),
        "accuracy": float(np.mean(true == predicted)),
        "recall": recall,
        "bootstrap_95ci": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "per_wear_f1": {str(g): macro_f1(true[groups == g], predicted[groups == g]) for g in unique},
        "per_quadrant_f1": {f"{int(q) * 90}deg": macro_f1(true[quadrant == q], predicted[quadrant == q])
                            for q in np.unique(quadrant) if q >= 0},
        "confusion_true_rows": confusion_matrix(true, predicted, labels=list(range(4))).tolist(),
        "prediction_counts": dict(zip(CLASSES, np.bincount(predicted, minlength=4).tolist())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--w02-dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze_manifest.read_text())
    if sha256(Path(__file__)) != freeze["scorer_sha256"]:
        raise RuntimeError("scorer source differs from the frozen scorer hash")
    for entry in freeze["real_runs"] + freeze["null_runs"]:
        if sha256(Path(entry["run"]) / "checkpoint.pt") != entry["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {entry['run']}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "EVALUATION_STARTED.json").open("x") as file:
        json.dump({"started_utc": datetime.now(timezone.utc).isoformat(), "freeze_sha256": sha256(args.freeze_manifest),
                   "w02_dataset_sha256": sha256(args.w02_dataset)}, file, indent=2)

    data = data_module.load_w02(args.w02_dataset)
    data = data_module.subset(data, np.isin(data["part"], freeze["evaluate_parts"]))
    scored = ~data["is_calibration"]
    endpoints = freeze["endpoints"]
    taus = {name: spec.get("tau", 0.0) for name, spec in endpoints.items()}
    predictions = {}
    for entry in freeze["real_runs"] + freeze["null_runs"]:
        checkpoint = torch.load(Path(entry["run"]) / "checkpoint.pt", weights_only=False)
        predictions[entry["run"]] = predict_one_model(checkpoint, data, taus)
    results = {}
    for name, spec in endpoints.items():
        members = [predictions[e["run"]][name][scored] for e in freeze["real_runs"]
                   if e["arguments"]["backbone"] == spec["backbone"]]
        predicted = vote(members)
        results[name] = report(data["y"][scored], predicted, data["wear_group"][scored],
                               data["position"][scored], freeze["bootstrap"])
        results[name]["n_models"] = len(members)
    results["E4"] = report(data["y"][scored], nearest_centroid_in_recording(data)[scored], data["wear_group"][scored],
                           data["position"][scored], freeze["bootstrap"])
    primary_spec = endpoints[freeze["gate"]["applies_to"]]
    results["matched_null_primary_macro_f1"] = [
        macro_f1(data["y"][scored], predictions[e["run"]][freeze["gate"]["applies_to"]][scored])
        for e in freeze["null_runs"] if e["arguments"]["backbone"] == primary_spec["backbone"]]
    primary = results[freeze["gate"]["applies_to"]]
    gate = freeze["gate"]
    results["gate_passed"] = bool(primary["macro_f1"] >= gate["macro_f1_min"]
                                  and min(primary["recall"].values()) >= gate["per_class_recall_min"])
    results["n_scored"] = int(scored.sum())
    results["n_wears"] = int(len(np.unique(data["wear_group"])))
    results["n_recordings"] = int(len(np.unique(data["group"])))
    (args.output_dir / "frozen_evaluation.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("gate_passed", "n_scored", "n_wears")} |
                     {e: round(results[e]["macro_f1"], 4) for e in (*endpoints, "E4")}, indent=1))


if __name__ == "__main__":
    main()
