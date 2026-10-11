"""Fixed 24-run v14 C170 sensor-normalization queue (stdout-only workers)."""

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
RUNNER = PROJECT / 'scripts/runners/run_c170_sensor_normalization_v14.py'
OUTPUT_ROOT = PROJECT / "outputs/c170_sensor_normalization_v14_formal_20260930"
CONFIGS = ("sensor_median", "sensor_rms", "window_sensor_rms", "bounded_sensor_gain")
SEEDS = (17, 29, 43, 61, 89)
CONCURRENCY = 6
CPU_THREADS = 1
MAX_EPOCHS = 160
PATIENCE = 25
MEMORY_LIMIT_GIB = 24.0
ESTIMATED_NEXT_WORKER_GIB = 2.0


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


def tasks() -> list[dict[str, Any]]:
    result = []
    for normalization in CONFIGS:
        result.extend({"normalization": normalization, "seed": seed, "shuffle_labels": False} for seed in SEEDS)
        result.append({"normalization": normalization, "seed": 17, "shuffle_labels": True})
    for item in result:
        item.update({"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "cpu_threads": CPU_THREADS, "device": "cuda"})
    return result


def task_id(spec: dict[str, Any]) -> str:
    suffix = "_shuffle" if spec["shuffle_labels"] else ""
    return f"{spec['normalization']}{suffix}__seed-{spec['seed']}"


def source_files() -> dict[str, Path]:
    rels = (
        'scripts/launchers/launch_c170_sensor_normalization_v14.py',
        'scripts/runners/run_c170_sensor_normalization_v14.py',
        "src/c170_sensor_normalization_v14.py",
        'scripts/runners/run_c170_complex_orbit_experiment_v11.py',
        "src/c170_complex_orbit_model_v11.py",
        "src/c170_harmonic_phase_model_v4.py",
        'scripts/runners/run_c170_ring_cuda_experiment_v3.py',
        "src/c170_shared_scale_preprocessing_v3.py",
        "src/c170_ring_representation.py",
        'scripts/runners/run_c170_ring_cuda_experiment_v2.py',
        'scripts/runners/run_c170_experiment.py',
        "src/rotation_poc.py",
    )
    return {name: PROJECT / name for name in rels}


def create_snapshot(root: Path, run_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    code_root = root / "source_snapshot" / "code"
    code_root.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for rel, source in source_files().items():
        if not source.is_file():
            raise FileNotFoundError(source)
        target = code_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[rel.replace("/", "\\")] = sha256(target)
    data = {
        "format": "c170-sensor-normalization-v14-source-snapshot",
        "source_sha256": hashes,
        "dataset_path": str(DATASET.resolve()),
        "dataset_sha256": sha256(DATASET),
        "tasks": run_tasks,
        "fixed_protocol": {
            "train_count": 879, "validation_count": 320,
            "test_policy": "No test preprocessing/prediction/scoring; only the shared fixed loader integrity check.",
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "optimizer": "frozen v11 AdamW/CE/sampler/weights",
            "concurrency": CONCURRENCY, "cpu_threads_per_run": CPU_THREADS,
            "memory_gate": "launch next worker only if used GiB + 2.0 <= 24.0",
            "normalizations": {
                "sensor_median": "filtered/demeaned training-only pooled per-sensor median(abs), floor 1e-6",
                "sensor_rms": "filtered/demeaned training-only pooled per-sensor RMS, floor 1e-6",
                "window_sensor_rms": "per-window per-sensor temporal RMS, floor 1e-6; sample-adaptive, not hardware calibration",
                "bounded_sensor_gain": "shared scale then exp(0.1*(tanh(theta)-mean(tanh(theta)))); theta=0 init; 1e-3*mean(log_gain^2)",
            },
            "interpretation": "train-fitted per-sensor statistics may contain hardware and label/position effects; they do not isolate hardware calibration",
        },
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(0),
        "snapshot_created_utc": datetime.now(timezone.utc).isoformat(),
        "test_metrics_computed": False,
    }
    write_json(root / "source_snapshot" / "snapshot_manifest.json", data)
    return data


def verify_run(spec: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    marker_path = run_dir / "RUN_COMPLETE.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"completion artifacts missing: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    config = manifest.get("config", {})
    config_hash = canonical_sha(config)
    if manifest.get("status") != "completed" or marker.get("status") != "completed":
        raise RuntimeError(f"run is not completed: {run_dir}")
    if manifest.get("format") != "c170-sensor-normalization-v14-run" or marker.get("format") != "c170-sensor-normalization-v14-complete":
        raise RuntimeError(f"unexpected v14 format: {run_dir}")
    if config_hash != manifest.get("config_sha256") or config_hash != marker.get("config_sha256"):
        raise RuntimeError(f"config hash mismatch: {run_dir}")
    if sha256(manifest_path) != marker.get("manifest_sha256"):
        raise RuntimeError(f"manifest hash mismatch: {run_dir}")
    if manifest.get("test_metrics_computed") is not False or marker.get("test_metrics_computed") is not False:
        raise RuntimeError(f"unexpected test metrics: {run_dir}")
    if config.get("seed") != spec["seed"] or config.get("shuffle_labels") is not spec["shuffle_labels"]:
        raise RuntimeError(f"run identity mismatch: {run_dir}")
    normalization = config.get("sensor_normalization", {})
    if normalization.get("kind") != spec["normalization"]:
        raise RuntimeError(f"normalization config mismatch: {run_dir}")
    for name, expected in marker.get("artifact_sha256", {}).items():
        entry = manifest.get("artifacts", {}).get(name, {})
        artifact = run_dir / entry.get("path", "")
        if entry.get("sha256") != expected or not artifact.is_file() or sha256(artifact) != expected:
            raise RuntimeError(f"artifact hash mismatch: {run_dir}/{name}")
    return {"config_sha256": config_hash, "manifest_sha256": marker["manifest_sha256"], "artifact_sha256": marker.get("artifact_sha256", {})}


def worker_command(spec: dict[str, Any], run_dir: Path) -> list[str]:
    command = [
        str(PYTHON), str(RUNNER), "--output-dir", str(run_dir), "--normalization", spec["normalization"],
        "--seed", str(spec["seed"]), "--max-epochs", str(MAX_EPOCHS), "--patience", str(PATIENCE),
        "--cpu-threads", str(CPU_THREADS), "--device", "cuda", "--dataset", str(DATASET),
        "--confirm-physical-order",
    ]
    if spec["shuffle_labels"]:
        command.append("--shuffle-labels")
    return command


def launch(command: list[str], out, err) -> subprocess.Popen:
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    return subprocess.Popen(
        command, cwd=PROJECT, stdout=out, stderr=err, env=env, stdin=subprocess.DEVNULL,
        startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW,
    )


def schedule() -> None:
    if not PYTHON.is_file() or not RUNNER.is_file() or not DATASET.is_file():
        raise FileNotFoundError("frozen Python310, v14 runner, or dataset is missing")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite output root: {OUTPUT_ROOT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    run_tasks = tasks()
    if len(run_tasks) != 24 or len({task_id(item) for item in run_tasks}) != 24:
        raise RuntimeError("expected exactly 24 fixed v14 tasks")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    snapshot = create_snapshot(OUTPUT_ROOT, run_tasks)
    jobs = {
        task_id(item): {
            "spec": item, "status": "pending", "run_dir": str(OUTPUT_ROOT / task_id(item)),
            "stdout_log": str(OUTPUT_ROOT / (task_id(item) + ".stdout.log")),
            "stderr_log": str(OUTPUT_ROOT / (task_id(item) + ".stderr.log")), "pid": None,
        }
        for item in run_tasks
    }
    manifest: dict[str, Any] = {
        "format": "c170-sensor-normalization-v14-driver", "format_version": 1, "status": "running",
        "root": str(OUTPUT_ROOT), "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": str(PYTHON), "dataset_sha256": snapshot["dataset_sha256"],
        "source_sha256": snapshot["source_sha256"], "concurrency": CONCURRENCY,
        "cpu_threads_per_run": CPU_THREADS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "memory_gate": {"limit_used_gib": MEMORY_LIMIT_GIB, "estimated_next_worker_gib": ESTIMATED_NEXT_WORKER_GIB},
        "memory_start": memory_state(), "test_metrics_computed": False, "jobs": jobs,
    }
    manifest_path = OUTPUT_ROOT / "launcher_manifest.json"
    write_json(manifest_path, manifest)
    pending = list(run_tasks)
    active: dict[str, tuple[subprocess.Popen, Any, Any, float]] = {}
    last_start = 0.0
    failures: list[str] = []
    while pending or active:
        while pending and len(active) < CONCURRENCY and time.monotonic() - last_start >= 1.5:
            memory = memory_state()
            if memory["used_gib"] + ESTIMATED_NEXT_WORKER_GIB > MEMORY_LIMIT_GIB:
                manifest["memory_last"] = memory
                write_json(manifest_path, manifest)
                break
            spec = pending[0]
            key = task_id(spec)
            run_dir = Path(jobs[key]["run_dir"])
            if run_dir.exists():
                jobs[key].update({"status": "failed_preexisting_output", "error": "refusing overwrite"})
                failures.append(key)
                pending.pop(0)
                continue
            out = Path(jobs[key]["stdout_log"]).open("xb")
            err = Path(jobs[key]["stderr_log"]).open("xb")
            proc = launch(worker_command(spec, run_dir), out, err)
            active[key] = (proc, out, err, time.monotonic())
            jobs[key].update({"status": "running", "pid": proc.pid, "started_utc": datetime.now(timezone.utc).isoformat(), "memory_at_start": memory})
            pending.pop(0)
            last_start = time.monotonic()
            write_json(manifest_path, manifest)
        finished = []
        for key, (proc, out, err, started) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            out.close(); err.close()
            jobs[key]["returncode"] = int(code)
            jobs[key]["elapsed_seconds"] = time.monotonic() - started
            try:
                if code != 0:
                    raise RuntimeError(f"worker exit code {code}")
                jobs[key]["verified"] = verify_run(jobs[key]["spec"], Path(jobs[key]["run_dir"]))
                jobs[key]["status"] = "completed"
            except Exception as error:
                jobs[key].update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                failures.append(key)
            finished.append(key)
        for key in finished:
            del active[key]
        manifest["memory_last"] = memory_state()
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
