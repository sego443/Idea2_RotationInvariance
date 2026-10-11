from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
PYTHON = Path(r"C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe")
DATASET = PROJECT / "data/processed_c170_20260926/dataset.npz"
V11_RUNNER = PROJECT / 'scripts/runners/run_c170_complex_orbit_experiment_v11.py'
OUTPUT_ROOT = PROJECT / "outputs/c170_v11_complex_orbit_formal_20260929"
OLD_SEEDS = (17, 29, 43, 61, 89)
CONCURRENCY = 10
CPU_THREADS = 1
MAX_EPOCHS = 160
PATIENCE = 25
V11_ACTIVE = {name: [0, 1, 2, 3] for name in ("complex_power", "complex_gram", "orbit_mean", "orbit_lse", "complex_power_shuffle", "complex_gram_shuffle", "orbit_mean_shuffle", "orbit_lse_shuffle")}


class DriverError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def task_specs() -> list[dict[str, Any]]:
    tasks = []
    for variant in ("complex_power", "complex_gram", "orbit_mean", "orbit_lse"):
        tasks.extend({"family": "v11", "variant": variant, "seed": seed} for seed in OLD_SEEDS)
        tasks.append({"family": "v11", "variant": variant + "_shuffle", "seed": 17})
    for item in tasks:
        item.update({"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "cpu_threads": CPU_THREADS, "device": "cuda"})
    return tasks

def source_files() -> dict[str, Path]:
    return {
        'scripts/launchers/launch_c170_complex_orbit_v11.py': Path(__file__).resolve(),
        'scripts/runners/run_c170_complex_orbit_experiment_v11.py': V11_RUNNER,
        "src/c170_complex_orbit_model_v11.py": PROJECT / "src/c170_complex_orbit_model_v11.py",
        "src/c170_harmonic_phase_model_v4.py": PROJECT / "src/c170_harmonic_phase_model_v4.py",
        'scripts/runners/run_c170_ring_cuda_experiment_v3.py': PROJECT / 'scripts/runners/run_c170_ring_cuda_experiment_v3.py',
        "src/c170_shared_scale_preprocessing_v3.py": PROJECT / "src/c170_shared_scale_preprocessing_v3.py",
        "src/c170_ring_representation.py": PROJECT / "src/c170_ring_representation.py",
        'scripts/runners/run_c170_ring_cuda_experiment_v2.py': PROJECT / 'scripts/runners/run_c170_ring_cuda_experiment_v2.py',
        'scripts/runners/run_c170_experiment.py': PROJECT / 'scripts/runners/run_c170_experiment.py',
        "src/rotation_poc.py": PROJECT / "src/rotation_poc.py",
    }

def make_snapshot(root: Path, task_list: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot = root / "source_snapshot"
    code_root = snapshot / "code"
    code_root.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for rel, source in source_files().items():
        if not source.is_file():
            raise FileNotFoundError(source)
        dest = code_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        hashes[rel.replace("/", "\\")] = sha256(dest)
    v11 = importlib.import_module("scripts.runners.run_c170_complex_orbit_experiment_v11")
    manifest = {
        "format": "c170-complex-orbit-v11-source-snapshot",
        "source_sha256": hashes,
        "runner_provenance_sha256": {"v11": v11._collect_code_hashes()},
        "dataset_path": str(DATASET.resolve()),
        "dataset_sha256": sha256(DATASET),
        "tasks": task_list,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "test_metrics_computed": False,
    }
    write_json_atomic(snapshot / "snapshot_manifest.json", manifest)
    return manifest


def hidden_popen(command: list[str], logfile) -> subprocess.Popen:
    options: dict[str, Any] = {}
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        options = {"startupinfo": startup, "creationflags": subprocess.CREATE_NO_WINDOW}
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    return subprocess.Popen(command, cwd=PROJECT, stdout=logfile, stderr=subprocess.STDOUT, env=env, **options)


def command_for(spec: dict[str, Any], run_dir: Path) -> list[str]:
    runner = V11_RUNNER
    return [str(PYTHON), str(runner), "--output-dir", str(run_dir), "--variant", spec["variant"],
            "--seed", str(spec["seed"]), "--max-epochs", str(MAX_EPOCHS), "--patience", str(PATIENCE),
            "--cpu-threads", str(CPU_THREADS), "--device", "cuda", "--dataset", str(DATASET),
            "--confirm-physical-order"]


def task_id(spec: dict[str, Any]) -> str:
    return f"{spec['variant']}__seed-{spec['seed']}"


def read_progress(log_path: Path) -> tuple[int, int]:
    if not log_path.is_file():
        return 0, 0
    try:
        data = log_path.read_bytes()[-16384:].decode("utf-8", errors="replace")
    except OSError:
        return 0, 0
    epochs = [int(v) for v in re.findall(r"\bepoch[= ](\d+)", data)]
    updates = [int(v) for v in re.findall(r"\bupdates[= ](\d+)", data)]
    return (max(epochs, default=0), max(updates, default=0))


def verify_run(spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    marker_path = run_dir / "RUN_COMPLETE.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise DriverError(f"missing run completion artifacts: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    config = manifest.get("config", {})
    config_sha = canonical_sha(config)
    if manifest.get("status") != "completed" or marker.get("status") != "completed":
        raise DriverError(f"run did not complete: {run_dir}")
    if config_sha != manifest.get("config_sha256") or config_sha != marker.get("config_sha256"):
        raise DriverError(f"config hash mismatch: {run_dir}")
    if sha256(manifest_path) != marker.get("manifest_sha256"):
        raise DriverError(f"manifest hash mismatch: {run_dir}")
    if marker.get("test_metrics_computed") is not False or manifest.get("test_metrics_computed") is not False:
        raise DriverError(f"test metrics were unexpectedly computed: {run_dir}")
    if config.get("variant") != spec["variant"] or config.get("seed") != spec["seed"]:
        raise DriverError(f"wrong run identity: {run_dir}")
    if config.get("max_epochs") != MAX_EPOCHS or config.get("patience") != PATIENCE or config.get("cpu_threads") != CPU_THREADS or config.get("device") != "cuda:0":
        raise DriverError(f"run config changed from protocol: {run_dir}")
    is_shuffle = spec["variant"].endswith("_shuffle")
    if config.get("shuffle_labels") is not is_shuffle:
        raise DriverError(f"shuffle flag mismatch: {run_dir}")
    expected_mask = V11_ACTIVE[spec["variant"]]
    if config.get("active_feature_indices") != expected_mask:
        raise DriverError(f"feature mask mismatch: {run_dir}")
    test_access = manifest.get("test_access", {})
    if test_access.get("preprocessed") is not False or test_access.get("predicted_or_scored") is not False:
        raise DriverError(f"test data access boundary was violated: {run_dir}")
    split = manifest.get("split", {})
    variant_key = "v11_variant"
    if split.get(variant_key) != spec["variant"]:
        raise DriverError(f"split variant mismatch: {run_dir}")
    if split.get("label_shuffle_seed") != (spec["seed"] + 10_000 if is_shuffle else None):
        raise DriverError(f"shuffle seed mismatch: {run_dir}")
    mapping = split.get("label_shuffle_mapping")
    if not isinstance(mapping, list) or (is_shuffle and not mapping) or (not is_shuffle and mapping):
        raise DriverError(f"shuffle mapping mismatch: {run_dir}")
    expected_provenance = importlib.import_module("scripts.runners.run_c170_complex_orbit_experiment_v11")
    if manifest.get("provenance", {}).get("code_sha256") != expected_provenance._collect_code_hashes():
        raise DriverError(f"runner/source provenance mismatch: {run_dir}")
    artifact_map = marker.get("artifact_sha256", {})
    manifest_artifacts = manifest.get("artifacts", {})
    if set(artifact_map) != set(manifest_artifacts):
        raise DriverError(f"artifact key mismatch: {run_dir}")
    for name, expected_hash in artifact_map.items():
        entry = manifest_artifacts[name]
        path = run_dir / entry["path"]
        if entry["sha256"] != expected_hash or not path.is_file() or sha256(path) != expected_hash:
            raise DriverError(f"artifact hash mismatch: {run_dir}/{name}")
    return {"config_sha256": config_sha, "manifest_sha256": marker["manifest_sha256"], "artifact_sha256": artifact_map,
            "best_epoch": marker.get("selected_epoch_one_based"), "test_metrics_computed": False}


def schedule(root: Path) -> None:
    if not PYTHON.is_file() or not DATASET.is_file() or not V11_RUNNER.is_file():
        raise DriverError("required Python310, dataset, or adapter runner is missing")
    if root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {root}")
    if not torch.cuda.is_available():
        raise DriverError("CUDA is unavailable")
    task_list = task_specs()
    if len(task_list) != 24 or len({task_id(item) for item in task_list}) != 24:
        raise DriverError("fixed 24-task v11 tasklist is invalid")
    root.mkdir(parents=True, exist_ok=False)
    snapshot = make_snapshot(root, task_list)
    records: dict[str, dict[str, Any]] = {}
    for spec in task_list:
        key = task_id(spec)
        records[key] = {**spec, "run_dir": str(root / key), "stdout_log": str(root / f"{key}.stdout.log"),
                        "status": "pending", "latest_epoch": 0, "latest_updates": 0}
    launcher = {"format": "c170-complex-orbit-v11-24run-driver", "format_version": 1, "status": "running",
                "root": str(root), "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python": str(PYTHON), "python_version": platform.python_version(), "torch_version": torch.__version__,
                "cuda_build": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0),
                "dataset_sha256": sha256(DATASET), "runner_source_sha256": snapshot["runner_provenance_sha256"],
                "concurrency": CONCURRENCY, "cpu_threads_per_run": CPU_THREADS, "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE, "jobs": records, "test_metrics_computed": False}
    launcher_path = root / "launcher_manifest.json"
    write_json_atomic(launcher_path, launcher)
    pending = list(records)
    active: dict[str, dict[str, Any]] = {}
    failed = False
    started = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < CONCURRENCY and not failed:
                key = pending.pop(0)
                record = records[key]
                run_dir = Path(record["run_dir"])
                log_path = Path(record["stdout_log"])
                if run_dir.exists() or log_path.exists():
                    raise DriverError(f"refusing pre-existing run/log path: {run_dir}")
                log_handle = log_path.open("ab", buffering=0)
                proc = hidden_popen(command_for(record, run_dir), log_handle)
                active[key] = {"proc": proc, "log_handle": log_handle}
                record["status"] = "running"
                record["pid"] = proc.pid
                record["actual_command"] = command_for(record, run_dir)
                print(f"START {key} pid={proc.pid}", flush=True)
                write_json_atomic(launcher_path, launcher)
            for key, state in list(active.items()):
                proc = state["proc"]
                epoch, updates = read_progress(Path(records[key]["stdout_log"]))
                records[key]["latest_epoch"] = epoch
                records[key]["latest_updates"] = updates
                code = proc.poll()
                if code is None:
                    continue
                state["log_handle"].close()
                records[key]["returncode"] = code
                records[key]["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                if code == 0:
                    try:
                        verified = verify_run(records[key], Path(records[key]["run_dir"]))
                        records[key]["verified_completion"] = verified
                        records[key]["status"] = "completed"
                        print(f"DONE {key} best_epoch={verified['best_epoch']}", flush=True)
                    except Exception as error:
                        records[key]["status"] = "failed"
                        records[key]["failure"] = f"{type(error).__name__}: {error}"
                        failed = True
                else:
                    records[key]["status"] = "failed"
                    records[key]["failure"] = f"child exit code {code}"
                    failed = True
                    tail = read_progress(Path(records[key]["stdout_log"]))
                    records[key]["latest_epoch"], records[key]["latest_updates"] = tail
                    print(f"FAILED {key} exit={code} epoch={tail[0]}", flush=True)
                active.pop(key)
                write_json_atomic(launcher_path, launcher)
            time.sleep(1.0)
        launcher["status"] = "failed" if failed else "completed"
        launcher["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        launcher["elapsed_seconds"] = time.monotonic() - started
        launcher["completed_count"] = sum(item["status"] == "completed" for item in records.values())
        launcher["failed_count"] = sum(item["status"] == "failed" for item in records.values())
        launcher["pending_count"] = sum(item["status"] == "pending" for item in records.values())
        write_json_atomic(launcher_path, launcher)
        if failed or launcher["completed_count"] != 24:
            raise DriverError(f"24-run matrix incomplete: {launcher['completed_count']}/24 completed")
        print(f"ALL_COMPLETE root={root} elapsed={launcher['elapsed_seconds']:.1f}s", flush=True)
    finally:
        if active:
            for key, state in active.items():
                state["log_handle"].close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed v11 four-architecture and four seed-17 shuffle control driver.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    schedule(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DRIVER_ERROR {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
