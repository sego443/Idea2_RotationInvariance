"""Thin persistent V19 36-run launcher, adapted from the validated V17 queue."""
from __future__ import annotations

import ctypes, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
PYTHON = Path(r"C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe")
DATASET = PROJECT / "data/processed_c170_20260926/dataset.npz"
RUNNER = PROJECT / 'scripts/runners/run_c170_temporal_gram_v19.py'
ROOT = PROJECT / "outputs/c170_temporal_gram_v19_formal_20260930"
ARCHITECTURES = ("gz", "gzn", "hybrid")
RESOLUTIONS = (64, 128)
SEEDS = (17, 29, 43, 61, 89)
CONCURRENCY, CPU_THREADS, MAX_EPOCHS, PATIENCE = 6, 1, 160, 25
MEMORY_LIMIT_GIB, NEXT_WORKER_GIB = 24.0, 2.0


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_state():
    state = MEMORYSTATUSEX(); state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise OSError("GlobalMemoryStatusEx failed")
    gib = float(1024**3)
    return {"used_gib": (state.ullTotalPhys-state.ullAvailPhys)/gib,
            "available_gib": state.ullAvailPhys/gib, "load_percent": float(state.dwMemoryLoad)}


def write_json(path: Path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def task_id(s):
    return f"{s['architecture']}_L{s['resolution']}{'_shuffle' if s['shuffle_labels'] else ''}__seed-{s['seed']}"


def tasks():
    grid = []
    for arch in ARCHITECTURES:
        for resolution in RESOLUTIONS:
            grid.extend({"architecture": arch, "resolution": resolution, "seed": seed, "shuffle_labels": False}
                        for seed in SEEDS)
            grid.append({"architecture": arch, "resolution": resolution, "seed": 17, "shuffle_labels": True})
    for item in grid:
        item.update({"max_epochs": MAX_EPOCHS, "patience": PATIENCE, "cpu_threads": CPU_THREADS, "device": "cuda"})
    return grid


def is_completed(spec, run):
    marker, manifest, predictions = run/"RUN_COMPLETE.json", run/"run_manifest.json", run/"validation_predictions.csv"
    if not (marker.is_file() and manifest.is_file() and predictions.is_file()):
        return False
    try:
        c = json.loads(marker.read_text(encoding="utf-8-sig")); m = json.loads(manifest.read_text(encoding="utf-8-sig"))
        cfg = m.get("config", {}); arch = cfg.get("architecture", {})
        return (c.get("status") == m.get("status") == "completed"
                and c.get("test_metrics_computed") is False and m.get("test_metrics_computed") is False
                and cfg.get("seed") == spec["seed"] and cfg.get("shuffle_labels") is spec["shuffle_labels"]
                and m.get("metadata", {}).get("v19_architecture", {}).get("name", spec["architecture"]) == spec["architecture"]
                and m.get("metadata", {}).get("v19_architecture", {}).get("temporal_resolution", spec["resolution"]) == spec["resolution"]
                and arch.get("name") == f"temporal_gram_v19_{spec['architecture']}_L{spec['resolution']}")
    except Exception:
        return False


def command(spec, run):
    cmd = [str(PYTHON), str(RUNNER), "--output-dir", str(run), "--architecture", spec["architecture"],
           "--resolution", str(spec["resolution"]), "--seed", str(spec["seed"]), "--max-epochs", str(MAX_EPOCHS),
           "--patience", str(PATIENCE), "--cpu-threads", str(CPU_THREADS), "--device", "cuda",
           "--dataset", str(DATASET), "--confirm-physical-order"]
    if spec["shuffle_labels"]: cmd.append("--shuffle-labels")
    return cmd


def start(cmd, out, err):
    si = subprocess.STARTUPINFO(); si.dwFlags |= subprocess.STARTF_USESHOWWINDOW; si.wShowWindow = subprocess.SW_HIDE
    env = os.environ.copy()
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): env[k] = "1"
    return subprocess.Popen(cmd, cwd=PROJECT, stdout=out, stderr=err, env=env, stdin=subprocess.DEVNULL,
                            startupinfo=si, creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    if not PYTHON.is_file() or not RUNNER.is_file() or not DATASET.is_file():
        raise FileNotFoundError("Python310, V19 runner, or dataset missing")
    grid = tasks()
    if len(grid) != 36 or len({task_id(x) for x in grid}) != 36: raise RuntimeError("V19 grid must contain 36 unique items")
    jobs = {}
    for spec in grid:
        key = task_id(spec); run = ROOT/key
        completed = is_completed(spec, run)
        jobs[key] = {"spec": spec, "status": "completed" if completed else ("pending" if not run.exists() else "preexisting_incomplete"),
                     "run_dir": str(run), "stdout_log": str(ROOT/(key+".stdout.log")),
                     "stderr_log": str(ROOT/(key+".stderr.log")), "pid": None}
    bad = [k for k,v in jobs.items() if v["status"] == "preexisting_incomplete"]
    if bad: raise RuntimeError(f"refusing to overwrite incomplete output: {bad}")
    if jobs["gz_L64__seed-17"]["status"] != "completed": raise RuntimeError("expected official Gz/L64/seed17 run is not reusable")
    manifest = {"format":"c170-temporal-gram-v19-driver", "format_version":1, "status":"running",
                "root":str(ROOT), "started_utc":datetime.now(timezone.utc).isoformat(), "python":str(PYTHON),
                "dataset":str(DATASET), "concurrency":CONCURRENCY, "cpu_threads_per_run":CPU_THREADS,
                "max_epochs":MAX_EPOCHS, "patience":PATIENCE,
                "memory_gate":{"used_limit_gib":MEMORY_LIMIT_GIB,"estimated_next_worker_gib":NEXT_WORKER_GIB},
                "memory_start":memory_state(), "test_metrics_computed":False, "jobs":jobs}
    mpath = ROOT/"launcher_manifest.json"; write_json(mpath, manifest)
    pending = [s for s in grid if jobs[task_id(s)]["status"] == "pending"]
    active = {}; attempts = {task_id(s): 0 for s in grid}; last_start = 0.0
    while pending or active:
        while pending and len(active) < CONCURRENCY and time.monotonic()-last_start >= 1.5:
            mem = memory_state()
            if mem["used_gib"] + NEXT_WORKER_GIB > MEMORY_LIMIT_GIB:
                manifest["memory_last"] = mem; write_json(mpath, manifest); break
            spec = pending.pop(0); key = task_id(spec); run = Path(jobs[key]["run_dir"])
            if run.exists(): raise RuntimeError(f"refusing overwrite: {run}")
            attempts[key] += 1
            out = Path(jobs[key]["stdout_log"]).open("ab"); err = Path(jobs[key]["stderr_log"]).open("ab")
            proc = start(command(spec, run), out, err)
            active[key] = (proc, out, err, time.monotonic())
            jobs[key].update({"status":"running", "attempt":attempts[key], "pid":proc.pid,
                              "started_utc":datetime.now(timezone.utc).isoformat(), "memory_at_start":mem})
            last_start = time.monotonic(); manifest["active_count"] = len(active); manifest["pending_count"] = len(pending)
            write_json(mpath, manifest); print(f"START {key} pid={proc.pid} used_gib={mem['used_gib']:.2f}", flush=True)
        done=[]
        for key,(proc,out,err,started) in list(active.items()):
            rc=proc.poll()
            if rc is None: continue
            out.close(); err.close(); spec=jobs[key]["spec"]; run=Path(jobs[key]["run_dir"])
            jobs[key]["returncode"]=int(rc); jobs[key]["elapsed_seconds"]=time.monotonic()-started
            if rc == 0 and is_completed(spec,run):
                jobs[key]["status"]="completed"; print(f"DONE {key} rc=0",flush=True)
            elif attempts[key] < 2:
                archive=ROOT/"failed_attempts"/f"{key}__attempt-1"
                archive.parent.mkdir(parents=True,exist_ok=True)
                if run.exists(): os.replace(run,archive)
                jobs[key].update({"status":"retry_pending", "error":f"exit={rc} or completion validation failed"})
                pending.append(spec); print(f"RETRY {key} after rc={rc}",flush=True)
            else:
                jobs[key].update({"status":"failed", "error":f"exit={rc} or completion validation failed"})
                print(f"FAIL {key} rc={rc}",flush=True)
            done.append(key)
        for key in done: del active[key]
        manifest["completed_count"]=sum(j["status"]=="completed" for j in jobs.values())
        manifest["failed_count"]=sum(j["status"]=="failed" for j in jobs.values())
        manifest["pending_count"]=len(pending); manifest["active_count"]=len(active); manifest["memory_last"]=memory_state()
        write_json(mpath,manifest)
        if pending or active: time.sleep(1.0)
    manifest["status"]="completed" if all(j["status"]=="completed" for j in jobs.values()) else "failed"
    manifest["finished_utc"]=datetime.now(timezone.utc).isoformat(); write_json(mpath,manifest)
    print(f"V19_QUEUE_DONE completed={manifest['completed_count']} failed={manifest['failed_count']} root={ROOT}",flush=True)
    return 0 if manifest["status"]=="completed" else 1


if __name__ == "__main__": raise SystemExit(main())
