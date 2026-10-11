"""One-shot validation-null gate and final frozen-F2 test evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from scripts.runners import run_c170_complex_orbit_experiment_v11 as cg_runner
from scripts.runners import run_c170_experiment as c170
from scripts.runners import run_c170_ring_cuda_experiment_v2 as v2
from scripts.runners import run_c170_ring_cuda_experiment_v3 as v3
from scripts.runners import run_c170_temporal_phase_invariant_experiment_v6_power_shuffle as power_runner
from scripts.runners import run_c170_c4_kernel_ridge_v7 as v7
from src.c170_c4_kernel_ridge_v7 import apply_fold_scaler, assemble_view, build_feature_families, preprocess_windows
from src.c170_complex_orbit_model_v11 import C170ComplexOrbitClassifierV11
from src.c170_shared_scale_preprocessing_v3 import calibrate_windows_shared_scale_v3
from src.c170_temporal_phase_invariant_model_v6_power_shuffle import TemporalPhaseInvariantClassifierV6

OUT = PROJECT / "outputs" / "c170_f2_matched_nulls_20260929"
V11_ROOT = PROJECT / "outputs" / "c170_v11_complex_orbit_formal_20260929"
V6_ROOT = PROJECT / "outputs" / "c170_temporal_phase_invariant_v6_formal_cuda_20260929"
V6_EXT_ROOT = PROJECT / "outputs" / "c170_v4v6_power_extensions_formal_20260929"
V7_ROOT = PROJECT / "outputs" / "c170_c4_kernel_ridge_v7_formal_20260929"
FUSION_CSV = V11_ROOT / "fixed_hard_fusion_predictions.csv"
DATASET = PROJECT / "data" / "processed_c170_20260926" / "dataset.npz"
CG_SEEDS = (17, 29, 43, 61, 89)
POWER_SEEDS = (17, 29, 43, 61, 89, 101, 131, 173, 211, 257)
REPS = (17, 29, 43, 61, 89)
CLASS_NAMES = tuple(c170.DIRECTIONS)
NAME_TO_CLASS = {name: index for index, name in enumerate(CLASS_NAMES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def one_hot_average(predictions: list[np.ndarray]) -> np.ndarray:
    matrix = np.stack(predictions, axis=0)
    return (matrix[:, :, None] == np.arange(4)[None, None, :]).mean(axis=0)


def family_hard_majority(predictions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return seed vote frequencies, lowest-class-ID family winner, and its one-hot vote."""
    frequencies = one_hot_average(predictions)
    winner = np.argmax(frequencies, axis=1)
    return frequencies, winner, np.eye(4, dtype=float)[winner]


def equal_family_hard_majority(family_predictions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Each family contributes one hard vote; ties resolve to the lowest class ID."""
    counts = np.stack([np.eye(4, dtype=float)[prediction] for prediction in family_predictions], axis=0).sum(axis=0)
    return counts, np.argmax(counts, axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, distances: np.ndarray) -> dict:
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth), int(pred)] += 1
    recall = np.divide(np.diag(confusion), confusion.sum(axis=1), out=np.zeros(4, dtype=float), where=confusion.sum(axis=1) > 0)
    precision = np.divide(np.diag(confusion), confusion.sum(axis=0), out=np.zeros(4, dtype=float), where=confusion.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4, dtype=float), where=(precision + recall) > 0)
    by_position = {}
    for distance in sorted(np.unique(distances).tolist()):
        mask = distances == distance
        local = metrics_without_positions(y_true[mask], y_pred[mask])
        by_position[str(int(distance))] = local
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1)),
        "class_order": list(CLASS_NAMES),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_by_class": f1.tolist(),
        "confusion_rows_true_cols_pred": confusion.tolist(),
        "unique_predicted_classes": int(len(np.unique(y_pred))),
        "per_position": by_position,
    }


def metrics_without_positions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth), int(pred)] += 1
    recall = np.divide(np.diag(confusion), confusion.sum(axis=1), out=np.zeros(4, dtype=float), where=confusion.sum(axis=1) > 0)
    precision = np.divide(np.diag(confusion), confusion.sum(axis=0), out=np.zeros(4, dtype=float), where=confusion.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4, dtype=float), where=(precision + recall) > 0)
    return {"n": int(len(y_true)), "accuracy": float(np.mean(y_true == y_pred)), "macro_f1": float(np.mean(f1)), "recall": recall.tolist(), "unique_predicted_classes": int(len(np.unique(y_pred)))}


def nn_run_dir(rep: int, family: str, seed: int) -> Path:
    if family == "complex_gram":
        if rep == 17 and seed == 17:
            return V11_ROOT / "complex_gram_shuffle__seed-17"
        return OUT / f"rep-{rep}" / f"complex_gram__seed-{seed}"
    if family == "learned_power":
        if seed == rep:
            return V6_EXT_ROOT / f"v6_learned_power_shuffle__seed-{seed}"
        return OUT / f"rep-{rep}" / f"learned_power__seed-{seed}"
    raise ValueError(family)


def read_nn_predictions(path: Path, variant: str, seed: int) -> tuple[list[dict], np.ndarray]:
    rows = read_csv(path)
    if len(rows) != 320 or any(row["variant"] != variant or int(row["seed"]) != seed for row in rows):
        raise RuntimeError(f"invalid prediction rows or identity: {path}")
    rows.sort(key=lambda row: int(row["sample_index"]))
    pred = np.asarray([NAME_TO_CLASS[row["prediction"]] for row in rows], dtype=np.int64)
    return rows, pred


def v7_rows_for_mode(mode: str) -> tuple[list[dict], np.ndarray]:
    rows = [row for row in read_csv(V7_ROOT / "validation_predictions.csv") if row["mode"] == mode]
    rows.sort(key=lambda row: int(row["row"]))
    if len(rows) != 320 or [int(row["row"]) for row in rows] != list(range(320)):
        raise RuntimeError(f"invalid v7 validation rows for {mode}")
    return rows, np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)


def analyze_null_validation() -> dict:
    fusion_identity = read_csv(FUSION_CSV)
    if len(fusion_identity) != 320:
        raise RuntimeError("F2 validation identity file does not contain 320 rows")
    fusion_identity.sort(key=lambda row: int(row["sample_index"]))
    sample_ids = np.asarray([int(row["sample_index"]) for row in fusion_identity], dtype=np.int64)
    distances = np.asarray([int(row["distance_mm"]) for row in fusion_identity], dtype=np.int64)
    y_true = np.asarray([NAME_TO_CLASS[row["true_direction"]] for row in fusion_identity], dtype=np.int64)
    identity = [(row["sample_index"], row["distance_mm"], row["block"], row["true_direction"], row["channel_map"]) for row in fusion_identity]

    results = []
    all_rows = []
    for rep in REPS:
        family_predictions: dict[str, list[np.ndarray]] = {"complex_gram": [], "learned_power": []}
        reference_identity = None
        for family, seeds, variant in (
            ("complex_gram", CG_SEEDS, "complex_gram_shuffle"),
            ("learned_power", POWER_SEEDS, "learned_power_shuffle"),
        ):
            for seed in seeds:
                run_dir = nn_run_dir(rep, family, seed)
                manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
                if manifest.get("test_access", {}).get("test_metrics_computed") is True:
                    raise RuntimeError(f"null run touched test metrics: {run_dir}")
                rows, pred = read_nn_predictions(run_dir / "validation_predictions.csv", variant, seed)
                current_identity = [(row["sample_index"], row["distance_mm"], row["block"], row["true_direction"], row["channel_map"]) for row in rows]
                if current_identity != identity:
                    raise RuntimeError(f"validation identity mismatch for {run_dir}")
                if reference_identity is not None and current_identity != reference_identity:
                    raise RuntimeError(f"member identity mismatch within rep {rep}")
                reference_identity = current_identity
                family_predictions[family].append(pred)

        v7_null_rows, v7_pred = v7_rows_for_mode(f"group_shuffle_{rep}")
        v7_y_true = np.asarray([int(row["y_true"]) for row in v7_null_rows], dtype=np.int64)
        if not np.array_equal(v7_y_true, y_true):
            raise RuntimeError(f"v7 null validation row identity mismatch for rep {rep}")
        # Existing fixed fusion binds v7's group_true row order to public validation IDs.
        if rep == 17:
            true_rows, true_v7_pred = v7_rows_for_mode("group_true")
            expected_v7 = np.asarray([NAME_TO_CLASS[row["v7group_pred"]] for row in fusion_identity], dtype=np.int64)
            if not np.array_equal(true_v7_pred, expected_v7):
                raise RuntimeError("v7 public validation row order no longer matches saved F2 identity table")

        cg_frequency, cg_pred, cg_vote = family_hard_majority(family_predictions["complex_gram"])
        power_frequency, power_pred, power_vote = family_hard_majority(family_predictions["learned_power"])
        v7_vote = np.eye(4, dtype=float)[v7_pred]
        family_vectors = {"complex_gram": cg_vote, "learned_power": power_vote, "v7_group": v7_vote}
        scores, predictions = equal_family_hard_majority([cg_pred, power_pred, v7_pred])
        score = metrics(y_true, predictions, distances)
        results.append({"rep": rep, "label_shuffle_seed": rep + 10_000, "metrics": score})
        for index in range(320):
            row = {
                "rep": rep,
                "sample_index": int(sample_ids[index]),
                "distance_mm": int(distances[index]),
                "block": int(fusion_identity[index]["block"]),
                "true_direction": CLASS_NAMES[int(y_true[index])],
                "complex_gram_pred": CLASS_NAMES[int(cg_pred[index])],
                "learned_power_pred": CLASS_NAMES[int(power_pred[index])],
                "v7_group_pred": CLASS_NAMES[int(v7_pred[index])],
                "fused_pred": CLASS_NAMES[int(predictions[index])],
            }
            for family, vector in family_vectors.items():
                for class_id, class_name in enumerate(CLASS_NAMES):
                    row[f"{family}_vote_{class_name.lower()}"] = float(vector[index, class_id])
            for family, vector in (("complex_gram_seed_frequency", cg_frequency), ("learned_power_seed_frequency", power_frequency)):
                for class_id, class_name in enumerate(CLASS_NAMES):
                    row[f"{family}_{class_name.lower()}"] = float(vector[index, class_id])
            for class_id, class_name in enumerate(CLASS_NAMES):
                row[f"fused_vote_{class_name.lower()}"] = float(scores[index, class_id])
            all_rows.append(row)

    maximum = max(item["metrics"]["macro_f1"] for item in results)
    result = {
        "format": "c170-frozen-f2-matched-null-validation-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_fusion": "argmax raw member logits/scores per member; within each neural family, majority vote across members with lowest class ID on ties; each of complex_gram, learned_power, and v7_group then casts one hard vote; majority vote across the three family votes with lowest class ID on ties",
        "validation_only": True,
        "rep_results": results,
        "max_null_validation_macro_f1": maximum,
        "test_gate_threshold": 0.5,
        "test_stage_authorized_by_gate": bool(maximum < 0.5),
        "test_metrics_computed": False,
    }
    atomic_json(OUT / "matched_null_validation_metrics_frozen_f2.json", result)
    fields = list(all_rows[0])
    write_csv(OUT / "matched_null_validation_predictions_frozen_f2.csv", all_rows, fields)
    print("NULL_VALIDATION_METRICS " + json.dumps({"rep_macro_f1": {str(x['rep']): x['metrics']['macro_f1'] for x in results}, "max": maximum, "all_below_gate": maximum < 0.5}, separators=(",", ":")), flush=True)
    return result


def _construct_models_and_inputs(dataset: dict, splits: dict, device: torch.device, shared_scale: float):
    raw = dataset["X_raw"]
    test_indices = np.asarray(splits["test"], dtype=np.int64)
    if len(test_indices) != 320:
        raise RuntimeError(f"expected 320 test windows, got {len(test_indices)}")
    test_raw = raw[test_indices]
    filtered = c170.filter_windows(test_raw.copy(), sample_rate=250)
    test_normalized, used_scale = calibrate_windows_shared_scale_v3(filtered, scale=shared_scale)
    if used_scale != shared_scale:
        raise RuntimeError("test normalization changed train-fitted scalar")
    test_tensor = torch.from_numpy(test_normalized.astype(np.float32, copy=False)).to(device=device)
    return test_tensor


def _load_state_dict_safely(runner, model, path: Path, device):
    expected = model.state_dict()
    loaded = runner._load_numpy_state_dict(path, expected, device)
    model.load_state_dict(loaded, strict=True)
    return model


def _neural_scores_f2(dataset: dict, splits: dict, prepared: dict, device: torch.device, *, run_dir: Path, family: str, seed: int, variant: str, evaluate_test: bool):
    if family == "complex_gram":
        runner = cg_runner
        model = C170ComplexOrbitClassifierV11(variant="complex_gram").to(device)
    else:
        runner = power_runner
        model = TemporalPhaseInvariantClassifierV6(variant="learned_power").to(device)
    _load_state_dict_safely(runner, model, run_dir / "best_weights.npz", device)
    model.eval()
    with torch.no_grad():
        scores = model(prepared["validation_inputs"]).detach().float().cpu().numpy()
    old_rows, old_pred = read_nn_predictions(run_dir / "validation_predictions.csv", variant, seed)
    pred = scores.argmax(axis=1)
    if not np.array_equal(pred, old_pred):
        raise RuntimeError(f"safe-state validation replay differs from frozen CSV: {run_dir}")
    test_scores = None
    if evaluate_test:
        raise RuntimeError("test scoring must be called through the gated final stage")
    return scores, pred, old_rows


def replay_real_validation() -> tuple[dict, list[dict], list[dict]]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("final frozen-F2 validation replay requires the original CUDA environment")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args = SimpleNamespace(dataset=DATASET, confirm_physical_order=True, variant="shared_scale_ce", seed=17)
    dataset, splits, _, prepared = v3._validate_and_prepare_v3(args, device)
    shared_scale = float(prepared["preprocessing"]["shared_sensor_scale"])
    if shared_scale != 11123.037109375:
        raise RuntimeError(f"train-only shared scale changed: {shared_scale}")
    identity_rows = read_csv(FUSION_CSV)
    identity_rows.sort(key=lambda row: int(row["sample_index"]))
    if len(identity_rows) != 320:
        raise RuntimeError("frozen validation identity has wrong size")

    cg_predictions, power_predictions = [], []
    all_member_replays = []
    for family, seeds, source_roots, variant in (
        ("complex_gram", CG_SEEDS, [V11_ROOT] * len(CG_SEEDS), "complex_gram"),
        ("learned_power", POWER_SEEDS, [V6_ROOT] * 5 + [V6_EXT_ROOT] * 5, "learned_power"),
    ):
        for seed, source_root in zip(seeds, source_roots):
            run_dir = source_root / (f"complex_gram__seed-{seed}" if family == "complex_gram" else (f"learned_power__seed-{seed}" if source_root == V6_ROOT else f"v6_learned_power__seed-{seed}"))
            if family == "complex_gram":
                run_dir = V11_ROOT / f"complex_gram__seed-{seed}"
            scores, pred, saved_rows = _neural_scores_f2(dataset, splits, prepared["tensors"], device, run_dir=run_dir, family=family, seed=seed, variant=variant, evaluate_test=False)
            if len(saved_rows) != 320:
                raise RuntimeError(f"validation replay row mismatch: {run_dir}")
            for row, identity in zip(saved_rows, identity_rows):
                if int(row["sample_index"]) != int(identity["sample_index"]) or row["true_direction"] != identity["true_direction"]:
                    raise RuntimeError(f"validation identity mismatch: {run_dir}")
            if family == "complex_gram":
                cg_predictions.append(pred)
            else:
                power_predictions.append(pred)
            all_member_replays.append({"family": family, "seed": seed, "run_dir": str(run_dir.resolve()), "prediction_match": True, "logits_sha256": hashlib.sha256(scores.tobytes()).hexdigest()})

    cg_frequency, cg_family_pred, cg_vote = family_hard_majority(cg_predictions)
    power_frequency, power_family_pred, power_vote = family_hard_majority(power_predictions)
    v7_rows, v7_pred = v7_rows_for_mode("group_true")
    y_true = np.asarray([NAME_TO_CLASS[row["true_direction"]] for row in identity_rows], dtype=np.int64)
    v7_true = np.asarray([int(row["y_true"]) for row in v7_rows], dtype=np.int64)
    if not np.array_equal(v7_true, y_true):
        raise RuntimeError("v7 validation targets do not match frozen identity")
    v7_vote = np.eye(4, dtype=float)[v7_pred]
    fused, pred = equal_family_hard_majority([cg_family_pred, power_family_pred, v7_pred])
    cached = np.asarray([NAME_TO_CLASS[row["F2_gram5_power10_v7_equal_3family"]] for row in identity_rows], dtype=np.int64)
    if not np.array_equal(pred, cached):
        mismatches = np.flatnonzero(pred != cached)
        raise RuntimeError(f"historical two-level hard-majority F2 replay differs at {len(mismatches)}/320 rows; first indices={mismatches[:10].tolist()}")
    for index, row in enumerate(identity_rows):
        if row["complex_gram_vote"] != CLASS_NAMES[int(cg_family_pred[index])]:
            raise RuntimeError(f"stored complex_gram family vote differs at validation row {index}")
        if row["v6power10_vote"] != CLASS_NAMES[int(power_family_pred[index])]:
            raise RuntimeError(f"stored v6power10 family vote differs at validation row {index}")
        if row["v7group_pred"] != CLASS_NAMES[int(v7_pred[index])]:
            raise RuntimeError(f"stored v7 group vote differs at validation row {index}")
    if abs(metrics(y_true, pred, np.asarray([int(row["distance_mm"]) for row in identity_rows], dtype=np.int64))["macro_f1"] - 0.8268715072387495) > 1e-12:
        raise RuntimeError("historical two-level hard-majority F2 macro-F1 does not reproduce the recorded value")
    val_metrics = metrics(y_true, pred, np.asarray([int(row["distance_mm"]) for row in identity_rows], dtype=np.int64))
    replay = {
        "format": "c170-frozen-f2-validation-replay",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "real_neural_members": all_member_replays,
        "all_15_neural_predictions_match_saved_validation_csv": True,
        "historical_family_votes_match_fixed_hard_fusion_csv_320_of_320": True,
        "f2_prediction_match_saved_fusion_csv": True,
        "actual_fusion_rule": "argmax raw member logits/scores per member; per-family hard majority (lowest class ID on ties); three-family hard majority (lowest class ID on ties)",
        "f2_metrics": val_metrics,
        "expected_exploration_macro_f1": 0.8268715072,
        "test_metrics_computed": False,
    }
    atomic_json(OUT / "frozen_f2_validation_replay.json", replay)
    print(f"FROZEN_F2_VALIDATION_REPLAY_OK macro_f1={val_metrics['macro_f1']:.8f} matched_models=15", flush=True)
    return {"dataset": dataset, "splits": splits, "prepared": prepared, "device": device, "scale": shared_scale}, cg_predictions, power_predictions


def _v7_test_predictions(dataset: dict, test_indices: np.ndarray, rep: int | None) -> tuple[np.ndarray, np.ndarray, Path, dict]:
    summary_rows = read_csv(V7_ROOT / "selected_validation_summary.csv")
    mode = "group_true" if rep is None else f"group_shuffle_{rep}"
    selected = [row for row in summary_rows if row["mode"] == mode]
    if len(selected) != 1:
        raise RuntimeError(f"v7 selected row missing/ambiguous: {mode}")
    selected = selected[0]
    npz_path = V7_ROOT / selected["state_npz"]
    state = np.load(npz_path, allow_pickle=False)
    scaler_mean = np.asarray(state["feature_mean"], dtype=np.float64)
    scaler_std = np.asarray(state["feature_std"], dtype=np.float64)
    view = selected["view"]
    family_names = v7.core.VIEW_FAMILIES[view]
    offset = 0
    family_slices, family_scales = {}, {}
    for family in family_names:
        end = offset + v7.core.FAMILY_DIMS[family]
        family_slices[family] = (offset, end)
        family_scales[family] = 1.0 / math.sqrt(v7.core.FAMILY_DIMS[family] * len(family_names))
        offset = end
    scaler = {"view": view, "mean": scaler_mean, "std": scaler_std, "family_slices": family_slices, "family_scales": family_scales}
    preprocessor = selected["preprocessor"]
    raw = dataset["X_raw"][test_indices]
    processed = preprocess_windows(raw.copy(), preprocessor)
    families = build_feature_families(processed)
    test_sensor = assemble_view(families, view)
    test_desc = apply_fold_scaler(test_sensor, scaler).reshape(len(test_indices), 4, -1)
    train_desc = np.asarray(state["train_descriptors"], dtype=np.float64)
    coeff = np.asarray(state["coefficients"], dtype=np.float64)
    x_train = torch.as_tensor(train_desc, dtype=torch.float64, device="cuda")
    x_test = torch.as_tensor(test_desc, dtype=torch.float64, device="cuda")
    train_d = v7._distance_stack(x_train, x_train, group=True)
    query_d = v7._distance_stack(x_test, x_train, group=True)
    query_self_d = v7._distance_stack(x_test, x_test, group=True)
    sigma = float(selected["sigma_base"]) * float(selected["selected_sigma_factor"])
    _, query_kernel = v7._group_kernel_from_distances(train_d, query_d, query_self_d, sigma)
    scores = 0.25 + query_kernel @ torch.as_tensor(coeff, dtype=torch.float64, device="cuda")
    scores_np = scores.detach().cpu().numpy()
    return scores_np, scores_np.argmax(axis=1), npz_path, {"mode": mode, "preprocessor": preprocessor, "view": view, "sigma": sigma, "alpha": float(selected["selected_alpha"]), "model_sha256": sha256(npz_path)}


def run_final_test(context: dict) -> None:
    dataset, splits, device, scale = context["dataset"], context["splits"], context["device"], context["scale"]
    test_indices = np.asarray(splits["test"], dtype=np.int64)
    test_tensor = _construct_models_and_inputs(dataset, splits, device, scale)
    test_labels = np.asarray(dataset["labels"][test_indices], dtype=np.int64)
    distances = np.asarray(dataset["distance_mm"][test_indices], dtype=np.int64)
    blocks = np.asarray(dataset["block"][test_indices], dtype=np.int64)
    channel_maps = np.asarray(dataset["channel_map"][test_indices], dtype=np.int64)

    true_logits: dict[str, np.ndarray] = {}
    true_predictions: dict[str, np.ndarray] = {"complex_gram": [], "learned_power": []}
    true_probabilities: dict[str, np.ndarray] = {}
    for family, seeds, source_root, variant in (
        ("complex_gram", CG_SEEDS, V11_ROOT, "complex_gram"),
        ("learned_power", POWER_SEEDS, None, "learned_power"),
    ):
        per_family = []
        for seed in seeds:
            if family == "complex_gram":
                runner = cg_runner
                model = C170ComplexOrbitClassifierV11(variant=variant).to(device)
                run_dir = V11_ROOT / f"complex_gram__seed-{seed}"
            else:
                runner = power_runner
                model = TemporalPhaseInvariantClassifierV6(variant=variant).to(device)
                power_root = V6_ROOT if seed in (17, 29, 43, 61, 89) else V6_EXT_ROOT
                dirname = f"learned_power__seed-{seed}" if power_root == V6_ROOT else f"v6_learned_power__seed-{seed}"
                run_dir = power_root / dirname
            _load_state_dict_safely(runner, model, run_dir / "best_weights.npz", device)
            model.eval()
            with torch.no_grad():
                logits = model(test_tensor).detach().float().cpu().numpy()
            probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
            pred = logits.argmax(axis=1)
            per_family.append(pred)
            key = f"{family}_seed_{seed}"
            true_logits[key] = logits
            true_probabilities[key] = probs
        true_predictions[family] = per_family

    v7_true_scores, v7_true_pred, v7_true_path, v7_true_config = _v7_test_predictions(dataset, test_indices, None)
    cg_frequency, cg_family_pred, cg_family_vote = family_hard_majority(true_predictions["complex_gram"])
    power_frequency, power_family_pred, power_family_vote = family_hard_majority(true_predictions["learned_power"])
    v7_family_vote = np.eye(4, dtype=float)[v7_true_pred]
    f2_vectors = {"complex_gram": cg_family_vote, "learned_power": power_family_vote, "v7_group": v7_family_vote}
    f2_scores, f2_pred = equal_family_hard_majority([cg_family_pred, power_family_pred, v7_true_pred])

    ensembles = {"f2_real": {"scores": f2_scores, "pred": f2_pred, "vectors": f2_vectors}}
    model_outputs = {"real": {"neural_logits": true_logits, "neural_probabilities": true_probabilities, "v7_scores": v7_true_scores, "v7_model": str(v7_true_path.resolve()), "v7_config": v7_true_config}}
    for rep in REPS:
        member_predictions = {"complex_gram": [], "learned_power": []}
        member_logits, member_probs = {}, {}
        for family, seeds in (("complex_gram", CG_SEEDS), ("learned_power", POWER_SEEDS)):
            for seed in seeds:
                run_dir = nn_run_dir(rep, family, seed)
                if family == "complex_gram":
                    runner = cg_runner
                    model = C170ComplexOrbitClassifierV11(variant="complex_gram").to(device)
                else:
                    runner = power_runner
                    model = TemporalPhaseInvariantClassifierV6(variant="learned_power").to(device)
                _load_state_dict_safely(runner, model, run_dir / "best_weights.npz", device)
                model.eval()
                with torch.no_grad():
                    logits = model(test_tensor).detach().float().cpu().numpy()
                probs = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
                pred = logits.argmax(axis=1)
                member_predictions[family].append(pred)
                member_logits[f"{family}_seed_{seed}"] = logits
                member_probs[f"{family}_seed_{seed}"] = probs
        v7_scores, v7_pred, v7_path, v7_config = _v7_test_predictions(dataset, test_indices, rep)
        _, cg_null_family_pred, cg_null_vote = family_hard_majority(member_predictions["complex_gram"])
        _, power_null_family_pred, power_null_vote = family_hard_majority(member_predictions["learned_power"])
        v7_null_vote = np.eye(4, dtype=float)[v7_pred]
        vectors = {"complex_gram": cg_null_vote, "learned_power": power_null_vote, "v7_group": v7_null_vote}
        ensemble_scores, ensemble_pred = equal_family_hard_majority([cg_null_family_pred, power_null_family_pred, v7_pred])
        ensembles[f"null_rep_{rep}"] = {"scores": ensemble_scores, "pred": ensemble_pred, "vectors": vectors}
        model_outputs[f"null_rep_{rep}"] = {"neural_logits": member_logits, "neural_probabilities": member_probs, "v7_scores": v7_scores, "v7_model": str(v7_path.resolve()), "v7_config": v7_config}

    output_rows = []
    metrics_out = {}
    for name, item in ensembles.items():
        scores, predictions = item["scores"], item["pred"]
        metrics_out[name] = metrics(test_labels, predictions, distances)
    for i, sample_index in enumerate(test_indices.tolist()):
        row = {
            "sample_index": int(sample_index),
            "distance_mm": int(distances[i]),
            "block": int(blocks[i]),
            "true_direction": CLASS_NAMES[int(test_labels[i])],
            "channel_map": ",".join(str(int(x)) for x in channel_maps[i]),
        }
        for model_key, logits in model_outputs["real"]["neural_logits"].items():
            for k, name in enumerate(CLASS_NAMES):
                row[f"real_{model_key}_logit_{name.lower()}"] = float(logits[i, k])
                row[f"real_{model_key}_prob_{name.lower()}"] = float(model_outputs["real"]["neural_probabilities"][model_key][i, k])
            row[f"real_{model_key}_pred"] = CLASS_NAMES[int(np.argmax(logits[i]))]
        for rep in REPS:
            for model_key, logits in model_outputs[f"null_rep_{rep}"]["neural_logits"].items():
                for k, name in enumerate(CLASS_NAMES):
                    row[f"null{rep}_{model_key}_logit_{name.lower()}"] = float(logits[i, k])
                    row[f"null{rep}_{model_key}_prob_{name.lower()}"] = float(model_outputs[f"null_rep_{rep}"]["neural_probabilities"][model_key][i, k])
                row[f"null{rep}_{model_key}_pred"] = CLASS_NAMES[int(np.argmax(logits[i]))]
        for k, name in enumerate(CLASS_NAMES):
            row[f"real_v7_group_score_{name.lower()}"] = float(v7_true_scores[i, k])
        row["real_v7_group_pred"] = CLASS_NAMES[int(np.argmax(v7_true_scores[i]))]
        for rep in REPS:
            v7_scores = model_outputs[f"null_rep_{rep}"]["v7_scores"]
            for k, name in enumerate(CLASS_NAMES):
                row[f"null{rep}_v7_group_score_{name.lower()}"] = float(v7_scores[i, k])
            row[f"null{rep}_v7_group_pred"] = CLASS_NAMES[int(np.argmax(v7_scores[i]))]
        for name, item in ensembles.items():
            for family, vector in item["vectors"].items():
                for k, class_name in enumerate(CLASS_NAMES):
                    row[f"{name}_{family}_vote_{class_name.lower()}"] = float(vector[i, k])
            for k, class_name in enumerate(CLASS_NAMES):
                row[f"{name}_ensemble_score_{class_name.lower()}"] = float(item["scores"][i, k])
            row[f"{name}_prediction"] = CLASS_NAMES[int(item["pred"][i])]
        output_rows.append(row)

    write_csv(OUT / "frozen_f2_and_null_test_predictions.csv", output_rows, list(output_rows[0]))
    report = {
        "format": "c170-frozen-f2-final-test-evaluation",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "completed",
        "test_was_used_once_after_null_validation_gate": True,
        "test_policy": "no fitting, normalization fitting, checkpoint selection, threshold changes, or model/fusion modification; train-fitted neural shared scalar and v7 feature scaler only",
        "test_count": len(test_indices),
        "test_positions_mm": sorted(int(x) for x in np.unique(distances)),
        "frozen_real_fusion": "argmax raw member logits/scores per member; hard majority within complex_gram5 and learned_power10; then hard majority of the two family winners and v7_group prediction; lowest class ID resolves ties at both levels",
        "null_fusions": "same fixed two-level hard-majority rule using each rep's single shared training-label permutation",
        "metrics": metrics_out,
        "acceptance_threshold": {"macro_f1": 0.80, "each_class_recall": 0.65},
        "f2_passes_threshold": metrics_out["f2_real"]["macro_f1"] >= 0.80 and min(metrics_out["f2_real"]["recall"]) >= 0.65,
        "source_artifacts": {
            "frozen_f2_validation_replay": sha256(OUT / "frozen_f2_validation_replay.json"),
            "null_validation_metrics": sha256(OUT / "matched_null_validation_metrics_frozen_f2.json"),
            "frozen_fusion_rule_revision": sha256(OUT / "frozen_f2_rule_revision.json"),
            "source_snapshot_manifest": sha256(OUT / "source_snapshot" / "snapshot_manifest.json"),
        },
        "test_metrics_computed": True,
    }
    atomic_json(OUT / "frozen_f2_final_test_metrics.json", report)
    print("FROZEN_F2_FINAL_TEST " + json.dumps({"macro_f1": metrics_out["f2_real"]["macro_f1"], "recall": metrics_out["f2_real"]["recall"], "pass": report["f2_passes_threshold"], "null_macro_f1": {k: metrics_out[k]["macro_f1"] for k in metrics_out if k.startswith("null_")}}, separators=(",", ":")), flush=True)


def _write_frozen_rule_revision(null_result: dict, context: dict) -> dict:
    fusion_rows = read_csv(FUSION_CSV)
    fusion_rows.sort(key=lambda row: int(row["sample_index"]))
    replay = json.loads((OUT / "frozen_f2_validation_replay.json").read_text(encoding="utf-8"))
    if len(fusion_rows) != 320 or not replay.get("f2_prediction_match_saved_fusion_csv"):
        raise RuntimeError("cannot freeze rule: historical F2 rowwise replay has not passed")
    if not null_result.get("test_stage_authorized_by_gate"):
        raise RuntimeError("cannot freeze rule for test: matched null validation gate failed")

    members = []
    for family, seeds in (("complex_gram", CG_SEEDS), ("learned_power", POWER_SEEDS)):
        for seed in seeds:
            if family == "complex_gram":
                run_dir = V11_ROOT / f"complex_gram__seed-{seed}"
            elif seed in (17, 29, 43, 61, 89):
                run_dir = V6_ROOT / f"learned_power__seed-{seed}"
            else:
                run_dir = V6_EXT_ROOT / f"v6_learned_power__seed-{seed}"
            members.append({
                "family": family,
                "seed": seed,
                "run_dir": str(run_dir.resolve()),
                "manifest_sha256": sha256(run_dir / "run_manifest.json"),
                "safe_weights_sha256": sha256(run_dir / "best_weights.npz"),
            })
    selected_v7 = [row for row in read_csv(V7_ROOT / "selected_validation_summary.csv") if row["mode"] == "group_true"]
    if len(selected_v7) != 1:
        raise RuntimeError("v7 group_true selected artifact is missing or ambiguous")
    v7_state = V7_ROOT / selected_v7[0]["state_npz"]
    source_paths = [
        PROJECT / 'scripts/runners/run_c170_complex_orbit_experiment_v11.py',
        PROJECT / "src" / "c170_complex_orbit_model_v11.py",
        PROJECT / 'scripts/runners/run_c170_temporal_phase_invariant_experiment_v6_power_shuffle.py',
        PROJECT / "src" / "c170_temporal_phase_invariant_model_v6_power_shuffle.py",
        PROJECT / 'scripts/runners/run_c170_c4_kernel_ridge_v7.py',
        PROJECT / "src" / "c170_c4_kernel_ridge_v7.py",
        PROJECT / "src" / "c170_shared_scale_preprocessing_v3.py",
    ]
    source_hashes = {str(path.relative_to(PROJECT)): sha256(path) for path in source_paths if path.is_file()}
    prior_metrics = OUT / "matched_null_validation_metrics.json"
    prior_record = None
    if prior_metrics.is_file():
        prior_record = {
            "path": str(prior_metrics.resolve()),
            "sha256": sha256(prior_metrics),
            "previous_written_rule": json.loads(prior_metrics.read_text(encoding="utf-8")).get("frozen_fusion"),
            "preserved_without_overwrite": True,
        }
    revision = {
        "format": "c170-frozen-f2-rule-revision",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "revision_reason": "An earlier evaluator description averaged neural-family vote-frequency vectors directly. The saved historical F2 CSV instead encodes hard majority within each family followed by one hard vote per family. The frozen predictions remain unchanged; this record documents the implementation discrepancy and correction.",
        "historical_evidence": {
            "fixed_hard_fusion_predictions_csv": str(FUSION_CSV.resolve()),
            "fixed_hard_fusion_predictions_sha256": sha256(FUSION_CSV),
            "fixed_hard_fusion_metrics_json": str((V11_ROOT / "fixed_hard_fusion_metrics.json").resolve()),
            "fixed_hard_fusion_metrics_sha256": sha256(V11_ROOT / "fixed_hard_fusion_metrics.json"),
            "validation_rows": len(fusion_rows),
            "family_vote_columns_match_replayed_family_majority": True,
            "f2_rows_match_replayed_two_level_hard_majority": 320,
            "f2_metrics_match_recorded_metrics": True,
        },
        "prior_incorrect_description_preserved": prior_record,
        "canonical_fusion_rule": "Per member, argmax raw neural logits or raw v7 KRR scores. Within complex_gram5 and learned_power10, choose the most frequent member argmax, ties to lowest class ID. The two neural family winners and v7 group_true prediction each cast one vote; choose the most frequent family vote, ties to lowest class ID.",
        "frozen_validation_macro_f1": replay["f2_metrics"]["macro_f1"],
        "frozen_validation_replay_sha256": sha256(OUT / "frozen_f2_validation_replay.json"),
        "corrected_matched_null_validation_sha256": sha256(OUT / "matched_null_validation_metrics_frozen_f2.json"),
        "matched_null_max_validation_macro_f1": null_result["max_null_validation_macro_f1"],
        "null_gate_threshold": 0.5,
        "null_gate_passed": null_result["test_stage_authorized_by_gate"],
        "real_member_checkpoints": members,
        "v7_selected_model": {"path": str(v7_state.resolve()), "sha256": sha256(v7_state), "mode": "group_true"},
        "dataset_sha256": sha256(DATASET),
        "split_counts": {key: int(len(value)) for key, value in context["splits"].items()},
        "source_sha256": source_hashes,
        "evaluator_sha256": sha256(Path(__file__)),
        "test_metrics_computed": False,
    }
    canonical = json.dumps(revision, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    revision["freeze_sha256"] = hashlib.sha256(canonical).hexdigest()
    atomic_json(OUT / "frozen_f2_rule_revision.json", revision)
    print(f"FROZEN_F2_RULE_SHA256 {revision['freeze_sha256']}", flush=True)
    return revision


def main() -> int:
    complete_path = OUT / "NULL_RUNS_COMPLETE.json"
    if not complete_path.is_file():
        raise FileNotFoundError("matched null queue is not complete")
    completion = json.loads(complete_path.read_text(encoding="utf-8"))
    if completion.get("status") != "completed" or completion.get("new_tasks_completed") != 69:
        raise RuntimeError("matched null queue did not finish all 69 new tasks successfully")
    context, _, _ = replay_real_validation()
    null_result = analyze_null_validation()
    if not null_result["test_stage_authorized_by_gate"]:
        print("TEST_STAGE_SKIPPED max null validation macro F1 did not fall below 0.5", flush=True)
        return 0
    _write_frozen_rule_revision(null_result, context)
    run_final_test(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
