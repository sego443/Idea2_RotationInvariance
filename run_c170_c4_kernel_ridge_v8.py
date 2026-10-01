from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_c4_kernel_ridge_v7 as shared_runner
from src import c170_c4_kernel_ridge_v8 as feature_core
from src import c170_c4_kernel_ridge_v7 as kernel_core

shared_runner.core = feature_core
_base_config = shared_runner._config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _v8_config(dataset_path: Path, phase: str) -> dict[str, Any]:
    config = _base_config(dataset_path, phase)
    config["format"] = "c170-c4-kernel-ridge-v8-run"
    config["feature_family_dims"] = feature_core.FAMILY_DIMS
    config["feature_views"] = list(feature_core.VIEWS)
    config["feature_view_families"] = {name: list(families) for name, families in feature_core.VIEW_FAMILIES.items()}
    config["feature_definition"] = {
        "e": "log1p(segment RMS / full-window per-channel RMS), np.array_split lengths 2,4,8,16,32, 62 values",
        "f": "signed segment mean / full-window per-channel RMS, np.array_split lengths 2,4,8,16,32, 62 values",
        "h": "asinh(mean non-wrapping x[t]*x[t-lag]/(RMS^2+eps)); lag 1,4,16,64 and segment counts 1,4,16, 84 values",
        "c": "v7 five Hann periodogram log-relative-power bands with total 0.5-125Hz denominator excluding DC",
        "d": "v7 log full-window per-channel RMS",
        "view_balancing": "per-fold shared sensor scaler; family dimension and included family count square-root balancing",
    }
    config["source_sha256"]["v8_feature_module"] = _sha256(PROJECT_DIR / "src" / "c170_c4_kernel_ridge_v8.py")
    config["source_sha256"]["v8_runner_wrapper"] = _sha256(Path(__file__).resolve())
    config["source_sha256"]["v7_shared_runner"] = _sha256(PROJECT_DIR / "run_c170_c4_kernel_ridge_v7.py")
    return config


def _v8_source_snapshot(root: Path, dataset_path: Path, config: dict[str, Any]) -> None:
    snapshot = root / "source_snapshot"
    code_root = snapshot / "code"
    code_root.mkdir(parents=True, exist_ok=False)
    sources = {
        "run_c170_c4_kernel_ridge_v8.py": Path(__file__).resolve(),
        "run_c170_c4_kernel_ridge_v7.py": PROJECT_DIR / "run_c170_c4_kernel_ridge_v7.py",
        "src/c170_c4_kernel_ridge_v8.py": PROJECT_DIR / "src" / "c170_c4_kernel_ridge_v8.py",
        "src/c170_c4_kernel_ridge_v7.py": PROJECT_DIR / "src" / "c170_c4_kernel_ridge_v7.py",
        "run_c170_experiment.py": PROJECT_DIR / "run_c170_experiment.py",
        "run_c170_ring_cuda_experiment_v2.py": PROJECT_DIR / "run_c170_ring_cuda_experiment_v2.py",
        "src/rotation_poc.py": PROJECT_DIR / "src" / "rotation_poc.py",
    }
    hashes = {}
    for rel, source in sources.items():
        target = code_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[rel.replace("/", "\\")] = _sha256(target)
    shared_runner._atomic_json(snapshot / "snapshot_manifest.json", {
        "format": "c170-c4-kernel-ridge-v8-source-snapshot",
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


shared_runner._config = _v8_config
shared_runner._source_snapshot = _v8_source_snapshot

_base_actual_fit = shared_runner._actual_fit_and_validate
_base_atomic_csv = shared_runner._atomic_csv


def _actual_fit_with_raw_scores(*args: Any, **kwargs: Any):
    arm_rows, predictions, c4_error = _base_actual_fit(*args, **kwargs)
    (preprocessor, view, train_sensor_features, val_sensor_features, train_labels,
     _val_labels, _train_positions, _train_blocks, _shuffled_targets, selected, output_root) = args
    scaler = feature_core.fit_fold_scaler(train_sensor_features, view)
    val_descriptors = feature_core.apply_fold_scaler(val_sensor_features, scaler).reshape(len(val_sensor_features), 4, -1)
    rows_by_mode = {row["mode"]: row for row in arm_rows}
    scores_by_mode: dict[str, np.ndarray] = {}
    for mode, summary in rows_by_mode.items():
        arrays = feature_core.safe_npz_load(Path(output_root) / summary["state_npz"])
        train_descriptors = torch.as_tensor(arrays["train_descriptors"], dtype=torch.float64, device="cuda")
        query_descriptors = torch.as_tensor(val_descriptors, dtype=torch.float64, device="cuda")
        sigma = float(summary["sigma_base"]) * float(summary["selected_sigma_factor"])
        kernel_mode = "c4_group" if mode != "plain_true" else "plain_rbf"
        query_kernel = kernel_core.kernel_matrix(query_descriptors, train_descriptors, sigma, kernel_mode)
        coefficients = torch.as_tensor(arrays["coefficients"], dtype=torch.float64, device="cuda")
        scores = (0.25 + query_kernel @ coefficients).detach().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite raw KRR scores for {mode}")
        scores_by_mode[mode] = scores
    score_names = ("score_UP", "score_DOWN", "score_LEFT", "score_RIGHT")
    for row in predictions:
        values = scores_by_mode[row["mode"]][int(row["row"])]
        row.update({name: float(value) for name, value in zip(score_names, values.tolist(), strict=True)})
    return arm_rows, predictions, c4_error


def _atomic_csv_with_scores(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    if Path(path).name == "validation_predictions.csv":
        fields = [*fields, "score_UP", "score_DOWN", "score_LEFT", "score_RIGHT"]
    return _base_atomic_csv(path, rows, fields)


shared_runner._actual_fit_and_validate = _actual_fit_with_raw_scores
shared_runner._atomic_csv = _atomic_csv_with_scores


def _check_actual_feature_roll(dataset_path: Path) -> None:
    data = shared_runner._prepare_train_val(dataset_path)
    raw = data["train_windows"][:12].copy()
    if len(raw) != 12:
        raise ValueError("actual train split is too small for the v8 feature check")
    maximum_delta = 0.0
    checked_dimensions: dict[str, int] = {}
    for preprocessor in feature_core.PREPROCESSORS:
        processed = feature_core.preprocess_windows(raw, preprocessor)
        features = feature_core.build_feature_families(processed)
        rolled = feature_core.build_feature_families(np.roll(processed, shift=1, axis=2))
        for name, values in features.items():
            expected = np.roll(values, shift=1, axis=1)
            delta = float(np.max(np.abs(rolled[name] - expected)))
            maximum_delta = max(maximum_delta, delta)
            if delta > 1e-10 or not np.isfinite(values).all():
                raise RuntimeError(f"v8 actual feature family {name} failed finite/C4 roll check: delta={delta}")
            checked_dimensions[name] = int(values.shape[-1])
    print(f"v8_actual_feature_check rows=12 dimensions={checked_dimensions} max_roll_abs_delta={maximum_delta:.3e}", flush=True)


def main() -> int:
    args = shared_runner.parse_args()
    if args.phase == "smoke":
        _check_actual_feature_roll(args.dataset.resolve())
        # The shared v7 smoke selects its first view by the legacy name "ab".
        # Alias that one smoke-only selection to the first v8 view without
        # changing the official v8 view list or formal candidate count.
        feature_core.VIEW_FAMILIES["ab"] = feature_core.VIEW_FAMILIES["ecd"]
        base_extract = shared_runner._extract_views

        def _extract_with_smoke_alias(train: np.ndarray, val: np.ndarray):
            result = base_extract(train, val)
            for values in result.values():
                values["ab"] = values["ecd"]
            return result

        shared_runner._extract_views = _extract_with_smoke_alias
        try:
            result = shared_runner.run(args)
        finally:
            shared_runner._extract_views = base_extract
            feature_core.VIEW_FAMILIES.pop("ab", None)
    else:
        result = shared_runner.run(args)
    if result == 0:
        marker_path = args.output_dir.resolve() / "RUN_COMPLETE.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["format"] = "c170-c4-kernel-ridge-v8-complete"
        shared_runner._atomic_json(marker_path, marker)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
