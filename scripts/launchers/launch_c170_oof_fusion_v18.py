"""Thin, resumable worker queue for the frozen v18 OOF fold grid."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from scripts.runners.run_c170_oof_fusion_v18 import DEFAULT_DATASET, DEFAULT_ROOT, EPOCHS, EXPERTS, POSITIONS, SEEDS


RUNNER = PROJECT / "scripts/runners/run_c170_oof_fusion_v18.py"
MAX_WORKERS = 6
RAM_LIMIT = 24 * 1024**3
RAM_HEADROOM = 2 * 1024**3


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def memory_used() -> tuple[int, int]:
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return int(status.ullTotalPhys - status.ullAvailPhys), int(status.ullTotalPhys)


def marker_valid(directory: Path, expert: str, position: int, seed: int) -> bool:
    marker_path = directory / "RUN_COMPLETE.json"
    manifest_path = directory / "run_manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("config", {})
        return (
            marker.get("expert") == expert
            and marker.get("heldout_train_position_mm") == position
            and marker.get("seed") == seed
            and marker.get("epochs_completed") == EPOCHS
            and manifest.get("status") == "complete"
            and config.get("expert") == expert
            and config.get("heldout_train_position_mm") == position
            and config.get("seed") == seed
            and config.get("epochs") == EPOCHS
            and (directory / "fold_probabilities.npz").is_file()
            and (directory / "last_epoch_state.npz").is_file()
        )
    except Exception:
        return False


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    root = args.root.resolve()
    dataset = args.dataset.resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "worker_logs"
    logs.mkdir(exist_ok=True)
    manifest_path = root / "queue_manifest.json"
    previous: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    task_states: dict[str, dict[str, object]] = dict(previous.get("tasks", {}))
    tasks = [(expert, position, seed) for expert in EXPERTS for position in POSITIONS for seed in SEEDS]
    pending: list[tuple[str, int, int]] = []
    reused = 0
    for expert, position, seed in tasks:
        key = f"{expert}|{position}|{seed}"
        directory = root / f"{expert}__position-{position:03d}__seed-{seed}"
        if marker_valid(directory, expert, position, seed):
            task_states[key] = {"status": "complete", "attempts": 0, "run_dir": str(directory), "reused": True}
            reused += 1
        else:
            pending.append((expert, position, seed))
    active: dict[str, dict[str, object]] = {}
    attempts: dict[str, int] = {}
    for task in pending:
        key = f"{task[0]}|{task[1]}|{task[2]}"
        prior = task_states.get(key, {})
        directory = root / f"{task[0]}__position-{task[1]:03d}__seed-{task[2]}"
        prior_failure = prior.get("status") in {"retry_pending", "failed"}
        prior_manifest = directory / "run_manifest.json"
        if prior_manifest.is_file():
            try:
                prior_failure = prior_failure or json.loads(prior_manifest.read_text(encoding="utf-8")).get("status") == "failed"
            except Exception:
                pass
        # An orphaned "active" task with a running manifest was interrupted, not
        # failed; resume it as attempt one. Only recorded failures consume retry.
        attempts[key] = 1 if prior_failure else 0

    def save_queue(status: str) -> None:
        completed = sum(1 for state in task_states.values() if state.get("status") == "complete")
        payload = {
            "format": "c170-oof-fusion-v18-queue", "status": status, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "root": str(root), "dataset": str(dataset), "max_workers": MAX_WORKERS,
            "ram_gate": {"used_plus_headroom_max_bytes": RAM_LIMIT, "headroom_bytes": RAM_HEADROOM},
            "grid_count": len(tasks), "complete_count": completed, "active_count": len(active),
            "pending_count": len(pending), "tasks": task_states,
            "active": {key: {"pid": item["process"].pid, "expert": item["expert"], "position": item["position"], "seed": item["seed"], "attempt": item["attempt"]} for key, item in active.items()},
        }
        write_json(manifest_path, payload)

    print(f"V18_QUEUE_START grid={len(tasks)} reused={reused} pending={len(pending)}", flush=True)
    save_queue("running")
    while pending or active:
        launched = False
        while pending and len(active) < MAX_WORKERS:
            try:
                used, total = memory_used()
            except Exception as error:
                print(f"V18_RAM_GATE_ERROR {type(error).__name__}: {error}", flush=True)
                save_queue("waiting_memory_query")
                break
            if used + RAM_HEADROOM > RAM_LIMIT:
                print(f"V18_RAM_GATE_WAIT used_gib={used/1024**3:.2f} total_gib={total/1024**3:.2f} active={len(active)}", flush=True)
                break
            expert, position, seed = pending.pop(0)
            key = f"{expert}|{position}|{seed}"
            attempt = attempts[key] + 1
            run_dir = root / f"{expert}__position-{position:03d}__seed-{seed}"
            log_path = logs / f"{expert}__position-{position:03d}__seed-{seed}__attempt-{attempt}.log"
            command = [sys.executable, str(RUNNER), "run-one", "--root", str(root), "--dataset", str(dataset),
                       "--expert", expert, "--position", str(position), "--seed", str(seed), "--device", "cuda"]
            log_handle = log_path.open("ab", buffering=0)
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(command, cwd=str(PROJECT), stdout=log_handle, stderr=subprocess.STDOUT,
                                       creationflags=flags)
            active[key] = {"process": process, "log": log_handle, "expert": expert, "position": position,
                           "seed": seed, "attempt": attempt, "run_dir": run_dir, "log_path": log_path}
            attempts[key] = attempt
            task_states[key] = {"status": "active", "attempts": attempt, "pid": process.pid,
                                "run_dir": str(run_dir), "log": str(log_path)}
            print(f"V18_START expert={expert} position={position} seed={seed} pid={process.pid} attempt={attempt}", flush=True)
            launched = True
            save_queue("running")
        completed_keys: list[str] = []
        for key, item in list(active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            expert, position, seed = item["expert"], item["position"], item["seed"]
            if marker_valid(item["run_dir"], expert, position, seed):
                task_states[key] = {"status": "complete", "attempts": item["attempt"], "exit_code": code,
                                    "run_dir": str(item["run_dir"]), "log": str(item["log_path"])}
                print(f"V18_DONE expert={expert} position={position} seed={seed} rc={code}", flush=True)
            elif item["attempt"] < 2:
                attempts[key] = item["attempt"]
                pending.append((expert, position, seed))
                task_states[key] = {"status": "retry_pending", "attempts": item["attempt"], "exit_code": code,
                                    "run_dir": str(item["run_dir"]), "log": str(item["log_path"])}
                print(f"V18_RETRY expert={expert} position={position} seed={seed} rc={code}", flush=True)
            else:
                task_states[key] = {"status": "failed", "attempts": item["attempt"], "exit_code": code,
                                    "run_dir": str(item["run_dir"]), "log": str(item["log_path"])}
                print(f"V18_FAILED expert={expert} position={position} seed={seed} rc={code}", flush=True)
            completed_keys.append(key)
        for key in completed_keys:
            del active[key]
        if completed_keys:
            save_queue("running")
        if active and not completed_keys and not launched:
            time.sleep(2)
        elif not active and pending and not launched:
            time.sleep(10)
    failed = [key for key, state in task_states.items() if state.get("status") == "failed"]
    save_queue("complete" if not failed else "failed")
    completed = sum(1 for state in task_states.values() if state.get("status") == "complete")
    print(f"V18_QUEUE_DONE completed={completed} failed={len(failed)}", flush=True)
    return 0 if not failed and completed == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
