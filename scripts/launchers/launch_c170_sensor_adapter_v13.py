from __future__ import annotations

import ctypes
import hashlib
import json
import os
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
RUNNER = PROJECT / 'scripts/runners/run_c170_sensor_adapter_v13.py'
OUTPUT_ROOT = PROJECT / "outputs/c170_sensor_adapter_v13_formal_20260930"
BASE_CG_ROOT = PROJECT / "outputs/c170_v11_complex_orbit_formal_20260929"
BASE_POWER_ROOT = PROJECT / "outputs/c170_v4v6_power_extensions_formal_20260929"
BASE_FUSION = PROJECT / "outputs/c170_v11_complex_orbit_formal_20260929/fixed_hard_fusion_predictions.csv"
SEEDS = (17, 29, 43, 61, 89)
ADAPTERS = ("per_sensor_gain", "per_sensor_fir_gain", "shared_fir")
CONCURRENCY = 6
CPU_THREADS = 1
MAX_EPOCHS = 160
PATIENCE = 25
MEMORY_LIMIT_GIB = 24.0


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_state() -> dict[str, float]:
    state = MEMORYSTATUSEX()
    state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise OSError("GlobalMemoryStatusEx failed")
    gib = float(1024**3)
    return {
        "total_gib": state.ullTotalPhys / gib,
        "available_gib": state.ullAvailPhys / gib,
        "used_gib": (state.ullTotalPhys - state.ullAvailPhys) / gib,
        "load_percent": float(state.dwMemoryLoad),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def task_id(spec: dict[str, Any]) -> str:
    suffix = "_shuffle" if spec["shuffle_labels"] else ""
    return f"{spec['adapter']}{suffix}__seed-{spec['seed']}"


def task_specs() -> list[dict[str, Any]]:
    result = []
    for adapter in ADAPTERS:
        result.extend({"adapter": adapter, "seed": seed, "shuffle_labels": False} for seed in SEEDS)
        result.append({"adapter": adapter, "seed": 17, "shuffle_labels": True})
    for spec in result:
        spec.update({"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "cpu_threads": CPU_THREADS, "device": "cuda"})
    return result


def source_files() -> dict[str, Path]:
    return {
        'scripts/launchers/launch_c170_sensor_adapter_v13.py': Path(__file__).resolve(),
        'scripts/runners/run_c170_sensor_adapter_v13.py': RUNNER,
        "src/c170_sensor_adapter_v13.py": PROJECT / "src/c170_sensor_adapter_v13.py",
        'scripts/runners/run_c170_complex_orbit_experiment_v11.py': PROJECT / 'scripts/runners/run_c170_complex_orbit_experiment_v11.py',
        "src/c170_complex_orbit_model_v11.py": PROJECT / "src/c170_complex_orbit_model_v11.py",
        "src/c170_harmonic_phase_model_v4.py": PROJECT / "src/c170_harmonic_phase_model_v4.py",
        'scripts/runners/run_c170_ring_cuda_experiment_v3.py': PROJECT / 'scripts/runners/run_c170_ring_cuda_experiment_v3.py',
        "src/c170_shared_scale_preprocessing_v3.py": PROJECT / "src/c170_shared_scale_preprocessing_v3.py",
        "src/c170_ring_representation.py": PROJECT / "src/c170_ring_representation.py",
        'scripts/runners/run_c170_ring_cuda_experiment_v2.py': PROJECT / 'scripts/runners/run_c170_ring_cuda_experiment_v2.py',
        'scripts/runners/run_c170_experiment.py': PROJECT / 'scripts/runners/run_c170_experiment.py',
        "src/rotation_poc.py": PROJECT / "src/rotation_poc.py",
    }


def snapshot(root: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    destination = root / "source_snapshot"
    code_root = destination / "code"
    code_root.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for rel, source in source_files().items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = code_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[rel.replace("/", "\\")] = sha256(target)
    baseline = {
        "complex_gram_runs": [str(BASE_CG_ROOT / f"complex_gram__seed-{seed}") for seed in SEEDS],
        "complex_gram_shuffle17": str(BASE_CG_ROOT / "complex_gram_shuffle__seed-17"),
        "v6_power_extension_root": str(BASE_POWER_ROOT),
        "fixed_v6_v7_validation_votes": str(BASE_FUSION),
    }
    data = {
        "format": "c170-sensor-adapter-v13-source-snapshot",
        "source_sha256": hashes,
        "dataset_path": str(DATASET.resolve()),
        "dataset_sha256": sha256(DATASET),
        "tasks": tasks,
        "reused_baseline_paths": baseline,
        "fixed_protocol": {
            "train_count": 879, "validation_count": 320, "test_policy": "no test preprocessing/prediction/scoring",
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "concurrency": CONCURRENCY,
            "cpu_threads_per_run": CPU_THREADS, "memory_acceptance_limit_used_gib": MEMORY_LIMIT_GIB,
            "optimizer": "existing v11 AdamW/CE/sampler/weights unchanged",
            "adapter_regularization_weight": 1e-3,
        },
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
        "test_metrics_computed": False,
    }
    write_json(destination / "snapshot_manifest.json", data)
    return data


def verify_run(spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    manifest_path, marker_path = run_dir / "run_manifest.json", run_dir / "RUN_COMPLETE.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"missing completion artifacts: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    config = manifest.get("config", {})
    digest = canonical_sha(config)
    if manifest.get("status") != "completed" or marker.get("status") != "completed":
        raise RuntimeError(f"run did not complete: {run_dir}")
    if manifest.get("format") != "c170-sensor-adapter-v13-run" or marker.get("format") != "c170-sensor-adapter-v13-complete":
        raise RuntimeError(f"unexpected v13 output format: {run_dir}")
    if digest != manifest.get("config_sha256") or digest != marker.get("config_sha256"):
        raise RuntimeError(f"config hash mismatch: {run_dir}")
    if sha256(manifest_path) != marker.get("manifest_sha256"):
        raise RuntimeError(f"manifest hash mismatch: {run_dir}")
    if manifest.get("test_metrics_computed") is not False or marker.get("test_metrics_computed") is not False:
        raise RuntimeError(f"test metrics were unexpectedly computed: {run_dir}")
    if config.get("seed") != spec["seed"] or config.get("shuffle_labels") is not spec["shuffle_labels"]:
        raise RuntimeError(f"wrong seed/shuffle identity: {run_dir}")
    adapter = config.get("sensor_adapter", {})
    if adapter.get("kind") != spec["adapter"] or adapter.get("regularization", {}).get("weight") != 1e-3:
        raise RuntimeError(f"adapter config mismatch: {run_dir}")
    for name, expected_hash in marker.get("artifact_sha256", {}).items():
        entry = manifest.get("artifacts", {}).get(name, {})
        path = run_dir / entry.get("path", "")
        if entry.get("sha256") != expected_hash or not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"artifact hash mismatch: {run_dir}/{name}")
    return {"config_sha256": digest, "manifest_sha256": marker["manifest_sha256"], "artifact_sha256": marker["artifact_sha256"]}


def worker_command(spec: dict[str, Any], run_dir: Path) -> list[str]:
    command = [str(PYTHON), str(RUNNER), "--output-dir", str(run_dir), "--adapter", spec["adapter"],
        "--seed", str(spec["seed"]), "--max-epochs", str(MAX_EPOCHS), "--patience", str(PATIENCE),
        "--cpu-threads", str(CPU_THREADS), "--device", "cuda", "--dataset", str(DATASET),
        "--confirm-physical-order"]
    if spec["shuffle_labels"]:
        command.append("--shuffle-labels")
    return command


def hidden_process(command: list[str], stdout, stderr) -> subprocess.Popen:
    options: dict[str, Any] = {}
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        options = {"startupinfo": startup, "creationflags": subprocess.CREATE_NO_WINDOW}
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    return subprocess.Popen(command, cwd=PROJECT, stdout=stdout, stderr=stderr, env=env, stdin=subprocess.DEVNULL, **options)


def schedule() -> None:
    if not PYTHON.is_file() or not DATASET.is_file() or not RUNNER.is_file():
        raise FileNotFoundError("required Python310, dataset, or v13 runner is missing")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite output root: {OUTPUT_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    tasks = task_specs()
    if len(tasks) != 18 or len({task_id(task) for task in tasks}) != 18:
        raise RuntimeError("the fixed v13 matrix must contain 18 unique tasks")
    root = OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=False)
    snapshot(root, tasks)
    jobs = {task_id(task): {"spec": task, "status": "pending", "run_dir": str(root / task_id(task)),
            "stdout_log": str(root / (task_id(task) + ".stdout.log")), "pid": None, "returncode": None}
        for task in tasks}
    memory = memory_state()
    manifest = {
        "format": "c170-sensor-adapter-v13-driver", "format_version": 1, "status": "running",
        "root": str(root), "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": str(PYTHON), "dataset_sha256": sha256(DATASET), "concurrency": CONCURRENCY,
        "cpu_threads_per_run": CPU_THREADS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "memory_limit_used_gib": MEMORY_LIMIT_GIB, "memory_start": memory,
        "test_metrics_computed": False, "jobs": jobs,
    }
    manifest_path = root / "launcher_manifest.json"
    write_json(manifest_path, manifest)
    pending = list(tasks)
    active: dict[str, tuple[subprocess.Popen, Any, Any, float]] = {}
    failures: list[str] = []
    last_start = 0.0
    while pending or active:
        while pending and len(active) < CONCURRENCY and time.monotonic() - last_start >= 1.5:
            memory = memory_state()
            if memory["used_gib"] >= MEMORY_LIMIT_GIB or memory["available_gib"] < 6.0:
                manifest["memory_last"] = memory
                write_json(manifest_path, manifest)
                break
            spec = pending.pop(0)
            key = task_id(spec)
            run_dir = Path(jobs[key]["run_dir"])
            if run_dir.exists():
                jobs[key]["status"] = "failed_preexisting_output"
                jobs[key]["error"] = "run directory exists; refusing overwrite"
                failures.append(key)
                continue
            out_handle = Path(jobs[key]["stdout_log"]).open("xb")
            err_handle = (root / (key + ".stderr.log")).open("xb")
            proc = hidden_process(worker_command(spec, run_dir), out_handle, err_handle)
            active[key] = (proc, out_handle, err_handle, time.monotonic())
            jobs[key].update({"status": "running", "pid": proc.pid, "started_utc": datetime.now(timezone.utc).isoformat(), "memory_at_start": memory})
            last_start = time.monotonic()
            write_json(manifest_path, manifest)
        finished = []
        for key, (proc, out_handle, err_handle, started) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            out_handle.close(); err_handle.close()
            spec = jobs[key]["spec"]
            jobs[key].update({"returncode": int(code), "elapsed_seconds": time.monotonic() - started})
            try:
                if code != 0:
                    raise RuntimeError(f"worker exit code {code}")
                jobs[key]["verified"] = verify_run(spec, Path(jobs[key]["run_dir"]))
                jobs[key]["status"] = "completed"
            except Exception as error:
                jobs[key]["status"] = "failed"
                jobs[key]["error"] = f"{type(error).__name__}: {error}"
                failures.append(key)
            finished.append(key)
        for key in finished:
            del active[key]
        memory = memory_state()
        manifest["memory_last"] = memory
        manifest["completed_count"] = sum(job["status"] == "completed" for job in jobs.values())
        manifest["failed_count"] = sum(job["status"].startswith("failed") for job in jobs.values())
        manifest["pending_count"] = len(pending)
        manifest["active_count"] = len(active)
        write_json(manifest_path, manifest)
        if pending or active:
            time.sleep(1.0)
    manifest["status"] = "completed" if not failures and all(job["status"] == "completed" for job in jobs.values()) else "failed"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_count"] = sum(job["status"] == "completed" for job in jobs.values())
    manifest["failed_count"] = sum(job["status"].startswith("failed") for job in jobs.values())
    manifest["pending_count"] = len(pending)
    manifest["active_count"] = 0
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    schedule()
