"""One-off CUDA driver for the fixed C170 temporal phase-invariant v6 suite."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = PROJECT_DIR / "run_c170_temporal_phase_invariant_experiment_v6.py"
DATASET_PATH = PROJECT_DIR / "data" / "processed_c170_20260926" / "dataset.npz"
PYTHON_EXE = Path(r"C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe")
SEEDS = (17, 29, 43, 61, 89)
FORMAL_VARIANTS = ("learned_gram", "fixed_gram", "learned_power", "learned_shuffle")
SMOKE_TASKS = (("learned_gram", 17), ("learned_shuffle", 17))
FORMAL_EPOCHS = 160
FORMAL_PATIENCE = 25
CPU_THREADS = 2
FORMAL_CONCURRENCY = 8
SMOKE_CONCURRENCY = 2
UPDATES_PER_EPOCH = 20
EXPECTED_SHARED_SCALE = 11123.037109375
FEATURE_LAGS = [1, 2, 4, 8, 16, 32, 64]
FEATURE_NAMES = [
    "asinh_s",
    *[f"log1p_p{index}" for index in range(8)],
    *[name for lag in FEATURE_LAGS for name in (f"q_lag{lag}_re", f"q_lag{lag}_im")],
]
FEATURE_DEFINITION = {
    "s": "(x0+x1+x2+x3)/4",
    "z": "a+ib with a=(x0-x2)/2 and b=(x1-x3)/2",
    "fir_bank": "8 real shared kernels applied independently to a and b",
    "identity_filter": "fixed buffer with tap[64]=1",
    "learned_filter_initial_lags": FEATURE_LAGS,
    "learned_filter_initial_taps": [64 - lag for lag in FEATURE_LAGS],
    "filter_l2_normalization": "w/sqrt(sum(w^2)+1e-8) per filter",
    "feature_order": "asinh(s), log1p(p_r) for r=0..7, then interleaved Re/Im q_r for r=1..7",
    "q_r": "z0*conj(zr)/sqrt((1+p0)*(1+pr))",
    "padding": "64 zeros on left only; causal FIR/features, no future samples",
    "global_phase_action": "constant phase on all times commutes with each shared real a/b filter and cancels in p/q",
    "causality_scope": "FIR bank and per-time features only; symmetric CNN padding and whole-window pooling are not streaming-causal",
}
ACTIVE_FEATURES = {
    "learned_gram": list(range(23)),
    "fixed_gram": list(range(23)),
    "learned_power": list(range(9)),
    "learned_shuffle": list(range(23)),
}
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_COMPLETE_NAME = "RUN_COMPLETE.json"
LAUNCHER_MANIFEST_NAME = "launcher_manifest.json"
SOURCE_PATHS = {
    "runner": PROJECT_DIR / "run_c170_temporal_phase_invariant_experiment_v6.py",
    "model": PROJECT_DIR / "src" / "c170_temporal_phase_invariant_model_v6.py",
    "v4_residual_backend_reference": PROJECT_DIR / "src" / "c170_harmonic_phase_model_v4.py",
    "shared_scale_v3_runner_helpers": PROJECT_DIR / "run_c170_ring_cuda_experiment_v3.py",
    "shared_scale_preprocessing": PROJECT_DIR / "src" / "c170_shared_scale_preprocessing_v3.py",
    "ring_representation_import_dependency": PROJECT_DIR / "src" / "c170_ring_representation.py",
    "frozen_v2_training_helpers": PROJECT_DIR / "run_c170_ring_cuda_experiment_v2.py",
    "dataset_preprocessing_helpers": PROJECT_DIR / "run_c170_experiment.py",
    "preprocessing_transitive_signal_processing": PROJECT_DIR / "src" / "rotation_poc.py",
}
SNAPSHOT_DOC_COPIES = {
    "docs/designs/2026-09-29-c170-temporal-phase-invariant-v6.md": Path("docs") / "design.md",
    "docs/plans/2026-09-29-c170-temporal-phase-invariant-v6.md": Path("docs") / "plan.md",
}


class DriverError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriverError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DriverError(f"expected an object in {path}")
    return value


def _verify_formal_snapshot(root: Path) -> None:
    snapshot_dir = root / "source_snapshot"
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    snapshot = _read_json(manifest_path)
    if "v6" not in str(snapshot.get("format", "")).lower():
        raise DriverError(f"unexpected v6 source snapshot format: {manifest_path}")
    expected_code_hashes = {name: _sha256(path) for name, path in SOURCE_PATHS.items()}
    code_hashes = snapshot.get("provenance_code_sha256")
    if code_hashes != expected_code_hashes:
        raise DriverError(f"source snapshot code hashes do not match the frozen source set: {manifest_path}")
    copied_hashes = snapshot.get("snapshot_file_sha256")
    if not isinstance(copied_hashes, dict):
        raise DriverError(f"source snapshot is missing copied-file hashes: {manifest_path}")
    if not set(SNAPSHOT_DOC_COPIES).issubset(copied_hashes):
        raise DriverError(f"source snapshot does not include the v6 design and plan copies: {manifest_path}")
    for relative, expected_hash in copied_hashes.items():
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise DriverError(f"invalid copied-file hash entry: {manifest_path}")
        physical = snapshot_dir / SNAPSHOT_DOC_COPIES[relative] if relative in SNAPSHOT_DOC_COPIES else snapshot_dir / "code" / relative
        if not physical.is_file() or _sha256(physical) != expected_hash:
            raise DriverError(f"source snapshot copied-file hash mismatch: {physical}")
    dataset = snapshot.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("sha256") != _sha256(DATASET_PATH):
        raise DriverError(f"source snapshot dataset hash does not match the fixed dataset: {manifest_path}")


def _hidden_options() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return {"startupinfo": startup, "creationflags": subprocess.CREATE_NO_WINDOW}


def _task_specs(phase: str) -> list[dict[str, Any]]:
    if phase == "smoke":
        return [
            {"variant": variant, "seed": seed, "max_epochs": 1, "patience": 1}
            for variant, seed in SMOKE_TASKS
        ]
    return [
        {
            "variant": variant,
            "seed": seed,
            "max_epochs": FORMAL_EPOCHS,
            "patience": FORMAL_PATIENCE,
        }
        for variant in FORMAL_VARIANTS
        for seed in SEEDS
    ]


def _command(run_dir: Path, spec: dict[str, Any]) -> list[str]:
    return [
        str(PYTHON_EXE),
        str(RUNNER_PATH),
        "--output-dir",
        str(run_dir),
        "--variant",
        str(spec["variant"]),
        "--seed",
        str(spec["seed"]),
        "--max-epochs",
        str(spec["max_epochs"]),
        "--patience",
        str(spec["patience"]),
        "--cpu-threads",
        str(CPU_THREADS),
        "--device",
        "cuda",
        "--dataset",
        str(DATASET_PATH),
        "--confirm-physical-order",
    ]


def _check_run(run_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Verify completion/config/artifact digests after the child has exited."""
    manifest_path = run_dir / RUN_MANIFEST_NAME
    complete_path = run_dir / RUN_COMPLETE_NAME
    manifest = _read_json(manifest_path)
    complete = _read_json(complete_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise DriverError(f"runner config is missing: {manifest_path}")
    config_sha = hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()
    if manifest.get("config_sha256") != config_sha or complete.get("config_sha256") != config_sha:
        raise DriverError(f"config hash mismatch: {run_dir}")
    expected_config = {
        "variant": spec["variant"],
        "model_variant": "temporal_phase_invariant_classifier_v6",
        "active_feature_indices": ACTIVE_FEATURES[spec["variant"]],
        "feature_names": FEATURE_NAMES,
        "feature_definition": FEATURE_DEFINITION,
        "shuffle_labels": spec["variant"] == "learned_shuffle",
        "seed": spec["seed"],
        "max_epochs": spec["max_epochs"],
        "patience": spec["patience"],
        "cpu_threads": CPU_THREADS,
        "device": "cuda:0",
        "dataset_path": str(DATASET_PATH.resolve()),
        "dataset_sha256": _sha256(DATASET_PATH),
        "train_distance_mm": [0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160],
        "validation_distance_mm": [5, 40, 90, 130],
        "registered_test_distance_mm": [15, 25, 110, 150],
        "excluded_distance_mm": [70],
        "expected_split_counts": {"train": 879, "validation": 320, "registered_test": 320},
        "batch_sizes_per_epoch": [44] * 19 + [43],
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "sampler_group_key": "original_distance_mm_and_direction_label",
        "sampler_rng": "numpy_seedsequence_seed_epoch_independent_of_model_rng",
        "sample_weight_rule": "position_voronoi_weight/(4*original_group_size)",
        "position_weight_period_mm": 170.0,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "cross_entropy_label_smoothing": 0.0,
        "batch_ce_objective": "20*sum(sample_weight_i*per_sample_ce_i)",
        "supcon_weight": 0.0,
        "preprocessing": "v3_filter_per_trial_demean_one_shared_train_median_scale",
        "preprocessing_scale_required": EXPECTED_SHARED_SCALE,
        "preprocessing_scale_fit_split": "train",
        "feature_lags": FEATURE_LAGS,
        "fir_lags": FEATURE_LAGS,
        "lag_boundary": "causal_left_zero_pad_64_no_future",
        "variant_feature_masks": ACTIVE_FEATURES,
        "fir_bank": {
            "filters": 8,
            "kernel_size": 65,
            "learned_parameter_count": 7 * 65,
            "identity_is_fixed_buffer": True,
            "fixed_gram_freezes_seven_filter_parameters": True,
        },
        "architecture": {
            "stem": "conv1d_23_to_24_k9_stride2_pad4_biasfalse_gn4_silu",
            "residual_dilations": [1, 2, 4],
            "residual_block": "depthwise24_k7_gn4_silu_pointwise24_gn4_residual_silu",
            "temporal_pooling": "adaptive_avg_lengths_1_2_4_plus_population_std_unbiased_false",
            "dropout": 0.2,
            "classifier": "linear_192_to_4",
            "total_parameters": 8763,
            "trainable_parameters": 8308 if spec["variant"] == "fixed_gram" else 8763,
            "initialization": "same_torch_manual_seed_and_parameter_shapes_for_all_variants",
        },
        "checkpoint_selection": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
        "safe_state_dict_format": "numpy_npz_numeric_arrays",
        "safe_state_dict_reload": "numpy_load_allow_pickle_false_then_strict_state_dict",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
        "physical_order_confirmed_by_operator": True,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise DriverError(
                f"config mismatch for {key} in {run_dir}: expected {expected!r}, got {config.get(key)!r}"
            )
    if manifest.get("format") != "c170-temporal-phase-invariant-v6-run" or manifest.get("format_version") != 1:
        raise DriverError(f"unexpected runner manifest format: {run_dir}")
    if manifest.get("status") != "completed" or complete.get("status") != "completed":
        raise DriverError(f"run is not completed: {run_dir}")
    if manifest.get("test_metrics_computed") is not False or complete.get("test_metrics_computed") is not False:
        raise DriverError(f"test metrics were not explicitly disabled: {run_dir}")
    test_access = manifest.get("test_access")
    if not isinstance(test_access, dict) or any(
        test_access.get(key) is not False
        for key in ("used_as_model_input", "preprocessed", "used_to_estimate_statistics", "predicted_or_scored")
    ):
        raise DriverError(f"test split was used by preprocessing or scoring: {run_dir}")
    if test_access.get("loaded_as_part_of_complete_npz") is not True:
        raise DriverError(f"unexpected complete-dataset loading declaration: {run_dir}")
    if complete.get("manifest_sha256") != _sha256(manifest_path):
        raise DriverError(f"manifest hash mismatch: {run_dir}")
    provenance = manifest.get("provenance")
    code_hashes = provenance.get("code_sha256") if isinstance(provenance, dict) else None
    expected_code_hashes = {name: _sha256(path) for name, path in SOURCE_PATHS.items()}
    if code_hashes != expected_code_hashes:
        raise DriverError(f"source provenance hashes do not match the frozen source set: {run_dir}")
    split = manifest.get("split")
    if not isinstance(split, dict) or split.get("split_counts") != {"train": 879, "val": 320, "test": 320}:
        raise DriverError(f"split counts do not match the v6 protocol: {run_dir}")
    if split.get("v6_variant") != spec["variant"]:
        raise DriverError(f"split variant does not match the requested run: {run_dir}")
    shuffled = spec["variant"] == "learned_shuffle"
    if split.get("label_shuffle_seed") != (int(spec["seed"]) + 10_000 if shuffled else None):
        raise DriverError(f"label shuffle seed does not match the v6 protocol: {run_dir}")
    shuffle_mapping = split.get("label_shuffle_mapping")
    if not isinstance(shuffle_mapping, list) or (shuffled and not shuffle_mapping) or (not shuffled and shuffle_mapping):
        raise DriverError(f"v6 label shuffle mapping does not match the requested branch: {run_dir}")

    artifact_names = {
        "best_checkpoint": "best_checkpoint.pt",
        "safe_state_dict": "best_weights.npz",
        "learning_curve": "learning_curve.csv",
        "validation_predictions": "validation_predictions.csv",
    }
    marker_hashes = complete.get("artifact_sha256")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(marker_hashes, dict) or not isinstance(manifest_artifacts, dict):
        raise DriverError(f"artifact hashes are missing: {run_dir}")
    checked_hashes: dict[str, str] = {}
    for key, filename in artifact_names.items():
        path = run_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise DriverError(f"artifact is missing or empty: {path}")
        digest = _sha256(path)
        summary = manifest_artifacts.get(key)
        if isinstance(summary, dict):
            summary_hash = summary.get("sha256")
            summary_path = summary.get("path")
        else:
            summary_hash = None
            summary_path = None
        if marker_hashes.get(key) != digest or summary_hash != digest or summary_path != filename:
            raise DriverError(f"artifact hash mismatch for {key}: {path}")
        if key == "safe_state_dict":
            try:
                with np.load(path, allow_pickle=False) as safe_state:
                    if not safe_state.files:
                        raise DriverError(f"safe selected weights are empty: {path}")
                    for tensor_name in safe_state.files:
                        array = safe_state[tensor_name]
                        if array.dtype.hasobject or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                            raise DriverError(f"safe selected weights contain invalid tensor {tensor_name}: {path}")
            except (OSError, ValueError) as exc:
                raise DriverError(f"selected weights are not a valid allow_pickle=False NPZ: {path}: {exc}") from exc
        checked_hashes[key] = digest
    if complete.get("format") != "c170-temporal-phase-invariant-v6-complete" or complete.get("format_version") != 1:
        raise DriverError(f"unexpected completion marker format: {run_dir}")
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise DriverError(f"runner selection metadata is missing: {run_dir}")
    selected_epoch = selection.get("best_epoch_one_based")
    if not isinstance(selected_epoch, int) or selected_epoch < 1:
        raise DriverError(f"runner selected epoch is invalid: {run_dir}")
    if complete.get("selected_epoch_one_based") != selected_epoch:
        raise DriverError(f"selected epoch does not match completion marker: {run_dir}")
    try:
        with (run_dir / "learning_curve.csv").open("r", newline="", encoding="utf-8-sig") as stream:
            curve = list(csv.DictReader(stream))
    except OSError as exc:
        raise DriverError(f"cannot read completed learning curve in {run_dir}: {exc}") from exc
    if len(curve) != selection.get("actual_epochs") or not curve:
        raise DriverError(f"learning curve length does not match completed epochs: {run_dir}")
    for expected_epoch, row in enumerate(curve, start=1):
        try:
            epoch = int(row["epoch"])
            updates = int(row["optimizer_updates"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DriverError(f"learning curve has an invalid epoch row in {run_dir}") from exc
        if epoch != expected_epoch or updates != UPDATES_PER_EPOCH:
            raise DriverError(f"learning curve epoch/update sequence is invalid in {run_dir}")
    return {
        "config_sha256": config_sha,
        "artifact_sha256": checked_hashes,
        "selected_epoch_one_based": selected_epoch,
        "actual_epochs": len(curve),
        "optimizer_updates": len(curve) * UPDATES_PER_EPOCH,
    }


def _read_progress(log_path: Path) -> tuple[int, int]:
    # Read only the append-only log. Never open learning_curve.csv while it is replaced.
    if not log_path.is_file():
        return 0, 0
    try:
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 32_768), os.SEEK_SET)
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return 0, 0
    for line in reversed(lines):
        epoch = re.search(r"\bepoch=(\d+)\b", line)
        updates = re.search(r"\bupdates=(\d+)\b", line)
        if epoch and updates:
            return int(epoch.group(1)), int(updates.group(1))
    return 0, 0


def _prepare_root(root: Path, phase: str) -> None:
    if not root.exists():
        root.mkdir(parents=True)
    entries = list(root.iterdir())
    if phase == "smoke":
        if entries:
            raise DriverError(f"smoke output root must be empty: {root}")
        return
    # The approved formal source snapshot is the sole allowed preexisting entry.
    if len(entries) != 1 or entries[0].name != "source_snapshot" or not entries[0].is_dir():
        raise DriverError(f"formal output root must contain only source_snapshot: {root}")
    if not (entries[0] / "snapshot_manifest.json").is_file():
        raise DriverError(f"formal source snapshot is incomplete: {entries[0]}")
    _verify_formal_snapshot(root)


def _schedule(root: Path, phase: str) -> None:
    if not PYTHON_EXE.is_file() or not RUNNER_PATH.is_file() or not DATASET_PATH.is_file():
        raise DriverError("required Python, v6 runner, or dataset is missing")
    _prepare_root(root, phase)
    specs = _task_specs(phase)
    concurrency = SMOKE_CONCURRENCY if phase == "smoke" else FORMAL_CONCURRENCY
    records: dict[str, dict[str, Any]] = {}
    for spec in specs:
        job_id = f"{spec['variant']}__seed-{spec['seed']}"
        records[job_id] = {
            **spec,
            "run_dir": str(root / job_id),
            "stdout_log": str(root / f"{job_id}.stdout.log"),
            "status": "pending",
            "latest_epoch": 0,
            "latest_updates": 0,
        }
    launcher = {
        "format": "c170-temporal-phase-invariant-v6-driver",
        "format_version": 1,
        "phase": phase,
        "status": "running",
        "started_utc": _utc_now(),
        "root": str(root),
        "runner": str(RUNNER_PATH),
        "runner_sha256": _sha256(RUNNER_PATH),
        "dataset_sha256": _sha256(DATASET_PATH),
        "concurrency": concurrency,
        "cpu_threads_per_run": CPU_THREADS,
        "jobs": records,
    }
    launcher_path = root / LAUNCHER_MANIFEST_NAME
    manifest_tmp = root / (LAUNCHER_MANIFEST_NAME + ".tmp")
    _write_json_atomic(launcher_path, launcher)
    pending = list(records)
    active: dict[str, dict[str, Any]] = {}
    failure: str | None = None
    start = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < concurrency and failure is None:
                job_id = pending.pop(0)
                record = records[job_id]
                run_dir = Path(record["run_dir"])
                log_path = Path(record["stdout_log"])
                if run_dir.exists() or log_path.exists():
                    failure = f"refusing to overwrite existing task output: {job_id}"
                    record["status"] = "failed"
                    record["failure_reason"] = failure
                    break
                command = _command(run_dir, record)
                record.update(
                    {
                        "status": "starting",
                        "command": command,
                        "command_display": subprocess.list2cmdline(command),
                        "started_utc": _utc_now(),
                    }
                )
                _write_json_atomic(launcher_path, launcher)
                log_stream = log_path.open("x", encoding="utf-8", newline="\n")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **_hidden_options(),
                )
                active[job_id] = {
                    "process": process,
                    "log_stream": log_stream,
                    "started_monotonic": time.monotonic(),
                }
                record.update({"status": "running", "pid": process.pid, "last_progress_utc": _utc_now()})
                print(f"started {job_id} pid={process.pid}", flush=True)
                _write_json_atomic(launcher_path, launcher)

            if failure:
                break
            changed = False
            for job_id, state in list(active.items()):
                record = records[job_id]
                epoch, updates = _read_progress(Path(record["stdout_log"]))
                if epoch > int(record["latest_epoch"]):
                    record.update(
                        {
                            "latest_epoch": epoch,
                            "latest_updates": updates,
                            "last_progress_utc": _utc_now(),
                        }
                    )
                    changed = True
                process: subprocess.Popen[Any] = state["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                state["log_stream"].close()
                record["returncode"] = return_code
                record["process_wall_seconds"] = max(0.0, time.monotonic() - state["started_monotonic"])
                record["ended_utc"] = _utc_now()
                if return_code != 0:
                    failure = f"{job_id} exited {return_code}; see {record['stdout_log']}"
                    record.update({"status": "failed", "failure_reason": failure})
                else:
                    try:
                        checked = _check_run(Path(record["run_dir"]), record)
                        record.update({"status": "completed", **checked})
                        print(f"completed {job_id} wall={record['process_wall_seconds']:.1f}s", flush=True)
                    except (DriverError, OSError) as exc:
                        failure = f"{job_id} completion verification failed: {exc}"
                        record.update({"status": "failed", "failure_reason": failure})
                del active[job_id]
                changed = True
                _write_json_atomic(launcher_path, launcher)
                if failure:
                    break
            if failure:
                break
            if changed:
                _write_json_atomic(launcher_path, launcher)
            time.sleep(0.5)
    finally:
        if failure:
            for job_id, state in active.items():
                proc: subprocess.Popen[Any] = state["process"]
                if proc.poll() is None:
                    proc.terminate()
                state["log_stream"].close()
                records[job_id].update({"status": "cancelled", "ended_utc": _utc_now()})
        launcher["elapsed_wall_seconds"] = max(0.0, time.monotonic() - start)
        launcher["ended_utc"] = _utc_now()
        launcher["status"] = "failed" if failure else "completed"
        if failure:
            launcher["failure_reason"] = failure
        _write_json_atomic(launcher_path, launcher)
    if failure:
        raise DriverError(failure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        _schedule(args.output_dir.resolve(), args.phase)
    except DriverError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
