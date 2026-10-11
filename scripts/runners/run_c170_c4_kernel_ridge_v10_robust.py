from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from scripts.runners import run_c170_c4_kernel_ridge_v7 as shared
VERSION_ROOTS = {
    "v7": PROJECT_DIR / "outputs/c170_c4_kernel_ridge_v7_formal_20260929",
    "v8": PROJECT_DIR / "outputs/c170_c4_kernel_ridge_v8_formal_retry1_20260929",
    "v9": PROJECT_DIR / "outputs/c170_c4_kernel_ridge_v9_formal_20260929",
}
VERSION_CORE_PATHS = {
    "v7": PROJECT_DIR / "src/c170_c4_kernel_ridge_v7.py",
    "v8": PROJECT_DIR / "src/c170_c4_kernel_ridge_v8.py",
    "v9": PROJECT_DIR / "src/c170_c4_kernel_ridge_v9.py",
}
VERSION_RUNNER_PATHS = {
    "v7": PROJECT_DIR / 'scripts/runners/run_c170_c4_kernel_ridge_v7.py',
    "v8": PROJECT_DIR / 'scripts/runners/run_c170_c4_kernel_ridge_v8.py',
    "v9": PROJECT_DIR / 'scripts/runners/run_c170_c4_kernel_ridge_v9.py',
}
VERSION_ORDER = {"v7": 0, "v8": 1, "v9": 2}
OUTPUT_ROOT = PROJECT_DIR / "outputs/c170_c4_kernel_ridge_v10_robust_selection_20260929"
MODES = tuple(shared.LABEL_MODES)
DATASET_PATH = PROJECT_DIR / "data/processed_c170_20260926/dataset.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def verify_prior_root(version: str, root: Path) -> dict[str, Any]:
    manifest_path = root / "run_manifest.json"
    marker_path = root / "RUN_COMPLETE.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"missing completed-run files in {root}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    if manifest.get("status") != "completed" or marker.get("status") != "completed":
        raise ValueError(f"prior run is not completed: {root}")
    if marker.get("test_metrics_computed") is not False or manifest.get("config", {}).get("test_metrics_computed") is not False:
        raise ValueError(f"prior run violates sealed-test protocol: {root}")
    if sha256(manifest_path) != marker.get("manifest_sha256"):
        raise ValueError(f"manifest SHA mismatch for {root}")
    config_sha = canonical_sha(manifest["config"])
    if config_sha != marker.get("config_sha256"):
        raise ValueError(f"config SHA mismatch for {root}")
    artifacts = manifest.get("artifacts", {})
    if artifacts != marker.get("artifact_sha256"):
        raise ValueError(f"marker/manifest artifact map mismatch for {root}")
    for rel, expected in artifacts.items():
        artifact = root / rel
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"artifact SHA mismatch: {artifact}")
    snapshot_manifest_path = root / "source_snapshot/snapshot_manifest.json"
    if not snapshot_manifest_path.is_file():
        raise FileNotFoundError(f"missing source snapshot manifest for {root}")
    snapshot = json.loads(snapshot_manifest_path.read_text(encoding="utf-8-sig"))
    if snapshot.get("dataset_sha256") != manifest["config"].get("dataset_sha256"):
        raise ValueError(f"source snapshot dataset hash mismatch for {root}")
    for rel, expected in snapshot.get("source_sha256", {}).items():
        normalized = rel.replace("\\", "/")
        candidates = [root / "source_snapshot/code" / normalized, root / "source_snapshot" / normalized]
        saved = next((candidate for candidate in candidates if candidate.is_file()), None)
        if saved is None or sha256(saved) != expected:
            raise ValueError(f"saved source snapshot mismatch: {root} :: {rel}")
    input_csv = root / "cv_candidates.csv"
    if not input_csv.is_file() or "cv_candidates.csv" not in artifacts:
        raise FileNotFoundError(f"missing CV candidate table in {root}")
    return {"manifest": manifest, "marker": marker, "candidate_csv": input_csv, "candidate_sha256": sha256(input_csv)}


def read_candidate_rows(version: str, path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    output: list[dict[str, Any]] = []
    for raw in raw_rows:
        row = {key.lstrip("\ufeff"): value for key, value in raw.items()}
        row["version"] = version
        row["alpha"] = float(row["alpha"])
        row["sigma_factor"] = float(row["sigma_factor"])
        row["candidate_order"] = int(row["candidate_order"])
        row["fold_count"] = int(row["fold_count"])
        folds = json.loads(row["fold_macro_f1"])
        if row["fold_count"] != 11 or len(folds) != 11 or not np.isfinite(folds).all():
            raise ValueError(f"invalid 11-fold scores: {version}/{row['mode']}/{row['candidate_order']}")
        row["fold_macro_f1_values"] = [float(value) for value in folds]
        computed_mean = float(np.mean(folds))
        if abs(computed_mean - float(row["mean_macro_f1"])) > 1e-10:
            raise ValueError(f"saved fold mean mismatch in {version} candidate table")
        row["robust_mean_macro_f1"] = computed_mean
        row["fold_population_sd_macro_f1"] = float(np.std(folds, ddof=0))
        row["robust_score"] = row["robust_mean_macro_f1"] - 0.5 * row["fold_population_sd_macro_f1"]
        output.append(row)
    return output


def robust_select(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    decorated: list[dict[str, Any]] = []
    for mode in MODES:
        pool = [row for row in rows if row["mode"] == mode]
        if len(pool) != 324 or {row["version"] for row in pool} != set(VERSION_ORDER):
            raise ValueError(f"{mode} must have exactly 324 candidates from v7/v8/v9")
        best = max(row["robust_score"] for row in pool)
        tied = [row for row in pool if best - row["robust_score"] <= 1e-8]
        tied.sort(key=lambda row: (-row["alpha"], VERSION_ORDER[row["version"]], row["candidate_order"]))
        chosen = tied[0]
        chosen = dict(chosen)
        chosen["candidate_pool_count"] = len(pool)
        chosen["selection_tie_count"] = len(tied)
        selected[mode] = chosen
        for row in pool:
            row["selected"] = int(row is chosen or (
                row["version"] == chosen["version"] and row["candidate_order"] == chosen["candidate_order"]
            ))
            decorated.append(row)
    return decorated, selected


def load_core(version: str):
    module_name = f"src.c170_c4_kernel_ridge_{version}"
    return importlib.import_module(module_name)


def extract_one(core: Any, raw_train: np.ndarray, raw_val: np.ndarray, preprocessor: str, view: str) -> dict[str, np.ndarray]:
    joined = np.concatenate([raw_train, raw_val], axis=0)
    processed = core.preprocess_windows(joined, preprocessor)
    families = core.build_feature_families(processed)
    combined = core.assemble_view(families, view)
    split = len(raw_train)
    return {"train": combined[:split], "val": combined[split:]}


def score_from_saved_model(core: Any, root: Path, summary: dict[str, Any], feature_data: dict[str, np.ndarray], val_positions: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
    npz = root / summary["state_npz"]
    state = core.safe_npz_load(npz)
    train_desc = np.asarray(state["train_descriptors"], dtype=np.float64).reshape(len(state["train_descriptors"]), 4, -1)
    scaler = core.fit_fold_scaler(feature_data["train"], summary["view"])
    if not np.array_equal(np.asarray(state["feature_mean"]), np.asarray(scaler["mean"])) or not np.array_equal(np.asarray(state["feature_std"]), np.asarray(scaler["std"])):
        raise ValueError(f"saved scaler mismatch for {summary['mode']}")
    val_desc = core.apply_fold_scaler(feature_data["val"], scaler).reshape(len(feature_data["val"]), 4, -1)
    x_train = torch.as_tensor(train_desc, dtype=torch.float64, device="cuda")
    x_val = torch.as_tensor(val_desc, dtype=torch.float64, device="cuda")
    mode = summary["kernel_mode"]
    if mode == "c4_group":
        train_d = shared._distance_stack(x_train, x_train, group=True)
        val_d = shared._distance_stack(x_val, x_train, group=True)
        val_self_d = shared._distance_stack(x_val, x_val, group=True)
        _, query_kernel = shared._group_kernel_from_distances(train_d, val_d, val_self_d, float(summary["sigma_base"]) * float(summary["selected_sigma_factor"]))
    elif mode == "plain_rbf":
        train_d = shared._distance_stack(x_train, x_train, group=False)
        val_d = shared._distance_stack(x_val, x_train, group=False)
        _, query_kernel = shared._plain_kernel_from_distances(train_d, val_d, float(summary["sigma_base"]) * float(summary["selected_sigma_factor"]))
    else:
        raise ValueError(f"unknown kernel mode {mode}")
    coefficients = torch.as_tensor(state["coefficients"], dtype=torch.float64, device="cuda")
    scores = (0.25 + query_kernel @ coefficients).detach().cpu().numpy()
    if scores.shape != (len(val_positions), 4) or not np.isfinite(scores).all():
        raise ValueError(f"invalid reconstructed score shape/values for {summary['mode']}")
    rows = []
    for index in range(len(val_positions)):
        rows.append({
            "preprocessor": summary["preprocessor"], "view": summary["view"], "mode": summary["mode"], "version": summary["source_version"],
            "row": index, "validation_position_mm": int(val_positions[index]), "y_true": int(summary["_val_labels"][index]),
            "y_pred": int(scores[index].argmax()), "score_UP": float(scores[index, 0]), "score_DOWN": float(scores[index, 1]),
            "score_LEFT": float(scores[index, 2]), "score_RIGHT": float(scores[index, 3]),
        })
    return rows, {"train_descriptor_count": len(train_desc), "validation_score_max_abs": float(np.max(np.abs(scores)))}


def copy_snapshot(root: Path, config: dict[str, Any]) -> dict[str, str]:
    snapshot_root = root / "source_snapshot"
    code_root = snapshot_root / "code"
    input_root = snapshot_root / "cv_inputs"
    code_root.mkdir(parents=True, exist_ok=False)
    input_root.mkdir(parents=True, exist_ok=False)
    sources = {
        'scripts/runners/run_c170_c4_kernel_ridge_v10_robust.py': Path(__file__).resolve(),
        'scripts/runners/run_c170_c4_kernel_ridge_v7.py': PROJECT_DIR / 'scripts/runners/run_c170_c4_kernel_ridge_v7.py',
        'scripts/runners/run_c170_c4_kernel_ridge_v8.py': PROJECT_DIR / 'scripts/runners/run_c170_c4_kernel_ridge_v8.py',
        'scripts/runners/run_c170_c4_kernel_ridge_v9.py': PROJECT_DIR / 'scripts/runners/run_c170_c4_kernel_ridge_v9.py',
        "src/c170_c4_kernel_ridge_v7.py": VERSION_CORE_PATHS["v7"],
        "src/c170_c4_kernel_ridge_v8.py": VERSION_CORE_PATHS["v8"],
        "src/c170_c4_kernel_ridge_v9.py": VERSION_CORE_PATHS["v9"],
        'scripts/runners/run_c170_experiment.py': PROJECT_DIR / 'scripts/runners/run_c170_experiment.py',
        'scripts/runners/run_c170_ring_cuda_experiment_v2.py': PROJECT_DIR / 'scripts/runners/run_c170_ring_cuda_experiment_v2.py',
        "src/rotation_poc.py": PROJECT_DIR / "src/rotation_poc.py",
    }
    hashes: dict[str, str] = {}
    for rel, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = code_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[rel.replace("/", "\\")] = sha256(target)
    input_copies = {}
    for version, prior in config["prior_runs"].items():
        sub = input_root / version
        sub.mkdir()
        for filename in ("cv_candidates.csv", "run_manifest.json", "RUN_COMPLETE.json", "source_snapshot/snapshot_manifest.json"):
            source = VERSION_ROOTS[version] / filename
            dest_name = filename.replace("/", "__")
            dest = sub / dest_name
            shutil.copyfile(source, dest)
            input_copies[f"{version}/{dest_name}"] = sha256(dest)
    snapshot_manifest = {
        "format": "c170-c4-kernel-ridge-v10-source-snapshot",
        "source_sha256": hashes,
        "cv_input_copy_sha256": input_copies,
        "dataset_path": config["dataset_path"],
        "dataset_sha256": config["dataset_sha256"],
        "config_sha256": canonical_sha(config),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "test_metrics_computed": False,
    }
    atomic_json(snapshot_root / "snapshot_manifest.json", snapshot_manifest)
    return hashes


def run() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite v10 output root: {OUTPUT_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("v10 final fits require CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verified = {version: verify_prior_root(version, root) for version, root in VERSION_ROOTS.items()}
    prior_manifests = {version: verified[version]["manifest"] for version in VERSION_ROOTS}
    dataset_hashes = {manifest["config"]["dataset_sha256"] for manifest in prior_manifests.values()}
    if len(dataset_hashes) != 1 or not DATASET_PATH.is_file() or sha256(DATASET_PATH) != next(iter(dataset_hashes)):
        raise ValueError("v7-v9/current dataset hash mismatch")
    reference_cfg = prior_manifests["v7"]["config"]
    for version, manifest in prior_manifests.items():
        cfg = manifest["config"]
        for key in ("train_distance_mm", "validation_distance_mm", "registered_test_distance_mm", "excluded_distance_mm", "train_count", "validation_count"):
            if cfg.get(key) != reference_cfg.get(key):
                raise ValueError(f"split/config mismatch for {version}: {key}")
        if cfg.get("candidate_count_per_arm") != 108 or manifest.get("counts", {}).get("train") != 879 or manifest.get("counts", {}).get("validation") != 320:
            raise ValueError(f"unexpected prior protocol counts for {version}")

    all_rows: list[dict[str, Any]] = []
    prior_inputs: dict[str, Any] = {}
    for version in VERSION_ORDER:
        info = verified[version]
        rows = read_candidate_rows(version, info["candidate_csv"])
        if len(rows) != 756 or {row["mode"] for row in rows} != set(MODES):
            raise ValueError(f"unexpected candidate table shape in {version}")
        if any(sum(row["mode"] == mode for row in rows) != 108 for mode in MODES):
            raise ValueError(f"expected 108 candidates per arm in {version}")
        all_rows.extend(rows)
        prior_inputs[version] = {
            "root": str(VERSION_ROOTS[version].resolve()),
            "manifest_sha256": info["marker"]["manifest_sha256"],
            "marker_sha256": sha256(VERSION_ROOTS[version] / "RUN_COMPLETE.json"),
            "cv_candidates_sha256": info["candidate_sha256"],
            "config_sha256": info["marker"]["config_sha256"],
        }
    candidate_rows, selected = robust_select(all_rows)
    config = {
        "format": "c170-c4-kernel-ridge-v10-robust-selection",
        "dataset_path": str(DATASET_PATH.resolve()), "dataset_sha256": sha256(DATASET_PATH),
        "prior_runs": prior_inputs,
        "train_distance_mm": reference_cfg["train_distance_mm"], "validation_distance_mm": reference_cfg["validation_distance_mm"],
        "registered_test_distance_mm": reference_cfg["registered_test_distance_mm"], "excluded_distance_mm": reference_cfg["excluded_distance_mm"],
        "train_count": 879, "validation_count": 320, "test_preprocessed": False, "test_metrics_computed": False,
        "candidate_versions": ["v7", "v8", "v9"], "candidates_per_version_per_arm": 108, "candidates_per_arm": 324,
        "fold_count": 11, "selection_score": "mean(fold_macro_f1)-0.5*population_sd(fold_macro_f1)",
        "tie_break": "within 1e-8: higher alpha, then v7/v8/v9, then candidate_order",
        "refit_rule": "one selected configuration per each of seven arms; fit all 879 train rows, score 320 validation rows once",
        "kernel_protocol": "reuse source-version feature/scaler/kernel protocol; C4 group kernel for group arms, plain RBF for plain_true",
        "python_version": platform.python_version(), "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (OUTPUT_ROOT / "models").mkdir()
    started = time.perf_counter()
    manifest: dict[str, Any] = {"format": config["format"], "format_version": 1, "status": "running", "config": config,
                                "config_sha256": canonical_sha(config), "test_metrics_computed": False,
                                "source_data_access": {"prior_train_cv_scores_reused": True, "train_preprocessed": True,
                                    "validation_preprocessed_for_selected_models_only": True, "test_preprocessed": False,
                                    "test_predictions_computed": False, "test_metrics_computed": False},
                                "selection": {mode: {key: selected[mode][key] for key in (
                                    "version", "preprocessor", "view", "sigma_factor", "alpha", "candidate_order",
                                    "robust_mean_macro_f1", "fold_population_sd_macro_f1", "robust_score", "fold_macro_f1_values",
                                    "candidate_pool_count", "selection_tie_count")} for mode in MODES},
                                "timing": {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)
    try:
        source_hashes = copy_snapshot(OUTPUT_ROOT, config)
        manifest["source_sha256"] = source_hashes
        atomic_csv(OUTPUT_ROOT / "robust_candidate_table.csv", candidate_rows,
                   ["version", "mode", "preprocessor", "view", "sigma_factor", "alpha", "candidate_order",
                    "fold_count", "robust_mean_macro_f1", "fold_population_sd_macro_f1", "robust_score", "selected"])
        print(f"validated_inputs versions=3 candidates_per_arm=324 folds=11 dataset_sha256={config['dataset_sha256']}", flush=True)
        data = shared._prepare_train_val(DATASET_PATH)
        null_targets, null_mappings = shared._make_null_labels(data["train"]["labels"], data["train"]["positions"], data["train"]["blocks"])
        manifest["split_metadata"] = {"train_positions_mm": sorted(np.unique(data["train"]["positions"]).astype(int).tolist()),
                                       "validation_positions_mm": sorted(np.unique(data["val"]["positions"]).astype(int).tolist()),
                                       "shuffle_mapping": null_mappings}
        summary_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []
        c4_diagnostics: dict[str, Any] = {}
        feature_cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
        grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        for mode, row in selected.items():
            grouped.setdefault((row["version"], row["preprocessor"], row["view"]), {})[mode] = row
        fit_started = time.perf_counter()
        for (version, preprocessor, view), selected_here in grouped.items():
            core = load_core(version)
            shared.core = core
            cache_key = (version, preprocessor, view)
            if cache_key not in feature_cache:
                feature_cache[cache_key] = extract_one(core, data["train_windows"], data["val_windows"], preprocessor, view)
            features = feature_cache[cache_key]
            rows, predictions, c4_error = shared._actual_fit_and_validate(
                preprocessor, view, features["train"], features["val"], data["train"]["labels"], data["val"]["labels"],
                data["train"]["positions"], data["train"]["blocks"], null_targets, selected_here, OUTPUT_ROOT)
            torch.cuda.synchronize()
            by_mode = {row["mode"]: row for row in rows}
            for mode, summary in by_mode.items():
                chosen = selected[mode]
                summary["source_version"] = version
                summary["robust_mean_macro_f1"] = chosen["robust_mean_macro_f1"]
                summary["fold_population_sd_macro_f1"] = chosen["fold_population_sd_macro_f1"]
                summary["robust_selection_score"] = chosen["robust_score"]
                summary["selected_candidate_order"] = chosen["candidate_order"]
                summary["candidate_pool_count"] = 324
                summary["_val_labels"] = data["val"]["labels"]
                reconstructed, _ = score_from_saved_model(core, OUTPUT_ROOT, summary, features, data["val"]["positions"])
                predicted_by_row = {int(item["row"]): int(item["y_pred"]) for item in predictions if item["mode"] == mode}
                if any(predicted_by_row.get(item["row"]) != item["y_pred"] for item in reconstructed):
                    raise ValueError(f"score reconstruction argmax mismatch for {mode}")
                prediction_rows.extend(reconstructed)
                summary_rows.append({key: value for key, value in summary.items() if key != "_val_labels"})
                for position in sorted(np.unique(data["val"]["positions"]).astype(int).tolist()):
                    mask = data["val"]["positions"] == position
                    local = shared._macro_metrics(data["val"]["labels"][mask], np.asarray([item["y_pred"] for item in reconstructed])[mask])
                    position_rows.append({"mode": mode, "source_version": version, "validation_position_mm": int(position),
                                          "count": int(mask.sum()), "macro_f1": local["macro_f1"],
                                          **{f"recall_{label}": local["recall_by_class"][index] for index, label in enumerate(("UP", "DOWN", "LEFT", "RIGHT"))}})
            if c4_error:
                c4_diagnostics[f"{version}/{preprocessor}/{view}"] = c4_error
            print(f"refit_done source={version} feature={preprocessor}/{view} arms={len(rows)}", flush=True)
        fit_seconds = time.perf_counter() - fit_started
        if len(summary_rows) != 7 or len(prediction_rows) != 7 * 320 or len(position_rows) != 7 * 4:
            raise RuntimeError(f"unexpected final output counts: summaries={len(summary_rows)} predictions={len(prediction_rows)} positions={len(position_rows)}")
        atomic_csv(OUTPUT_ROOT / "selected_validation_summary.csv", summary_rows, list(summary_rows[0].keys()))
        atomic_csv(OUTPUT_ROOT / "validation_predictions.csv", prediction_rows,
                   ["preprocessor", "view", "mode", "version", "row", "validation_position_mm", "y_true", "y_pred", "score_UP", "score_DOWN", "score_LEFT", "score_RIGHT"])
        atomic_csv(OUTPUT_ROOT / "validation_per_position_metrics.csv", position_rows, list(position_rows[0].keys()))
        manifest["c4_score_diagnostics"] = c4_diagnostics
        manifest["timing"].update({"fit_and_validation_seconds": fit_seconds, "total_elapsed_seconds": time.perf_counter() - started})
        manifest["counts"] = {"candidate_rows": len(candidate_rows), "modes": 7, "selected_models": len(summary_rows),
                               "validation_predictions": len(prediction_rows), "validation_position_rows": len(position_rows)}
        manifest["artifacts"] = {}
        for path in sorted(OUTPUT_ROOT.rglob("*")):
            if path.is_file() and path.name not in ("run_manifest.json", "RUN_COMPLETE.json") and not path.name.endswith(".tmp"):
                manifest["artifacts"][str(path.relative_to(OUTPUT_ROOT)).replace("/", "\\")] = sha256(path)
        manifest["status"] = "completed"
        atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)
        marker = {"format": "c170-c4-kernel-ridge-v10-robust-complete", "status": "completed",
                  "config_sha256": canonical_sha(config), "manifest_sha256": sha256(OUTPUT_ROOT / "run_manifest.json"),
                  "artifact_sha256": manifest["artifacts"], "selected_modes": list(MODES), "test_metrics_computed": False}
        atomic_json(OUTPUT_ROOT / "RUN_COMPLETE.json", marker)
        print(f"RUN_COMPLETE root={OUTPUT_ROOT} elapsed_sec={manifest['timing']['total_elapsed_seconds']:.3f}", flush=True)
        for row in summary_rows:
            print(f"RESULT {row['mode']} source={row['source_version']} {row['preprocessor']}/{row['view']} sigma={row['selected_sigma_factor']} alpha={row['selected_alpha']} robust={row['robust_selection_score']:.6f} val_f1={row['validation_macro_f1']:.6f} recalls={row['validation_recall_by_class']}", flush=True)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        manifest["timing"]["total_elapsed_seconds"] = time.perf_counter() - started
        atomic_json(OUTPUT_ROOT / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(run())
