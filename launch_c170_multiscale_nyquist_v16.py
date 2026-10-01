"""Fixed 18-run v16 C170 architecture queue; stdout-only worker monitoring."""

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

PROJECT = Path(__file__).resolve().parent
PYTHON = Path(r"C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe")
DATASET = PROJECT / "data/processed_c170_20260926/dataset.npz"
RUNNER = PROJECT / "run_c170_multiscale_nyquist_experiment_v16.py"
OUTPUT_ROOT = PROJECT / "outputs/c170_multiscale_nyquist_v16_formal_20260930"
ARCHITECTURES = ("multiscale", "nyquist", "multiscale_nyquist")
SEEDS = (17, 29, 43, 61, 89)
CONCURRENCY, CPU_THREADS, MAX_EPOCHS, PATIENCE = 6, 1, 160, 25
MEMORY_LIMIT_GIB, ESTIMATED_NEXT_WORKER_GIB = 24.0, 2.0


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory_state() -> dict[str, float]:
    state = MEMORYSTATUSEX(); state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise OSError("GlobalMemoryStatusEx failed")
    gib = float(1024**3)
    return {"total_gib": state.ullTotalPhys/gib, "available_gib": state.ullAvailPhys/gib, "used_gib": (state.ullTotalPhys-state.ullAvailPhys)/gib, "load_percent": float(state.dwMemoryLoad)}


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1<<20),b""): digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,indent=2,sort_keys=True,ensure_ascii=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,path)


def tasks() -> list[dict[str, Any]]:
    result=[]
    for architecture in ARCHITECTURES:
        result.extend({"architecture":architecture,"seed":seed,"shuffle_labels":False} for seed in SEEDS)
        result.append({"architecture":architecture,"seed":17,"shuffle_labels":True})
    for item in result: item.update({"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"cpu_threads":CPU_THREADS,"device":"cuda"})
    return result


def task_id(spec: dict[str, Any]) -> str:
    return f"{spec['architecture']}{'_shuffle' if spec['shuffle_labels'] else ''}__seed-{spec['seed']}"


def sources() -> dict[str, Path]:
    names=("launch_c170_multiscale_nyquist_v16.py","run_c170_multiscale_nyquist_experiment_v16.py","src/c170_multiscale_nyquist_v16.py","run_c170_complex_orbit_experiment_v11.py","src/c170_complex_orbit_model_v11.py","src/c170_harmonic_phase_model_v4.py","run_c170_ring_cuda_experiment_v3.py","src/c170_shared_scale_preprocessing_v3.py","src/c170_ring_representation.py","run_c170_ring_cuda_experiment_v2.py","run_c170_experiment.py","src/rotation_poc.py")
    return {name:PROJECT/name for name in names}


def snapshot(root: Path, run_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    code=root/"source_snapshot"/"code"; code.mkdir(parents=True,exist_ok=False); hashes={}
    for rel,source in sources().items():
        if not source.is_file(): raise FileNotFoundError(source)
        dst=code/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dst); hashes[rel.replace("/","\\")]=sha256(dst)
    value={"format":"c170-multiscale-nyquist-v16-source-snapshot","source_sha256":hashes,"dataset_path":str(DATASET.resolve()),"dataset_sha256":sha256(DATASET),"tasks":run_tasks,"protocol":{"train":879,"validation":320,"test_policy":"test is not preprocessed, predicted or scored","seeds":SEEDS,"concurrency":CONCURRENCY,"cpu_threads_per_run":CPU_THREADS,"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"memory_gate":"start only when used GiB + 2 <= 24","fusion":"five-seed per-family one-hot majority, then equal majority among v16 family, frozen v6 power10 and v7 group votes; lowest class ID ties"},"python":sys.version.split()[0],"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda_device":torch.cuda.get_device_name(0),"created_utc":datetime.now(timezone.utc).isoformat(),"test_metrics_computed":False}
    write_json(root/"source_snapshot"/"snapshot_manifest.json",value); return value


def verify(spec: dict[str, Any], run: Path) -> dict[str, Any]:
    mp=run/"run_manifest.json"; cp=run/"RUN_COMPLETE.json"
    if not mp.is_file() or not cp.is_file(): raise RuntimeError(f"completion files missing: {run}")
    m=json.loads(mp.read_text(encoding="utf-8-sig")); c=json.loads(cp.read_text(encoding="utf-8-sig")); cfg=m.get("config",{})
    digest=canonical_sha(cfg); name="complex_gram_v16_"+spec["architecture"]
    if m.get("status")!="completed" or c.get("status")!="completed": raise RuntimeError("run not completed")
    if m.get("format")!="c170-multiscale-nyquist-v16-run" or c.get("format")!="c170-multiscale-nyquist-v16-complete": raise RuntimeError("unexpected marker format")
    if digest!=m.get("config_sha256") or digest!=c.get("config_sha256") or sha256(mp)!=c.get("manifest_sha256"): raise RuntimeError("config/manifest hash mismatch")
    if m.get("test_metrics_computed") is not False or c.get("test_metrics_computed") is not False: raise RuntimeError("unexpected test metrics")
    arch=cfg.get("architecture",{})
    if cfg.get("seed")!=spec["seed"] or cfg.get("shuffle_labels") is not spec["shuffle_labels"] or cfg.get("model_variant")!="c170_complex_gram_v16_"+spec["architecture"] or arch.get("name")!=name: raise RuntimeError("config/task mismatch")
    artifacts={}
    for artifact,expected in c.get("artifact_sha256",{}).items():
        entry=m.get("artifacts",{}).get(artifact,{}); path=run/entry.get("path","")
        artifacts[artifact]=bool(entry.get("sha256")==expected and path.is_file() and sha256(path)==expected)
    if len(artifacts)<4 or not all(artifacts.values()): raise RuntimeError("artifact hash mismatch")
    return {"config_sha256":digest,"manifest_sha256":c["manifest_sha256"],"artifact_sha256":artifacts}


def command(spec: dict[str, Any], run: Path) -> list[str]:
    result=[str(PYTHON),str(RUNNER),"--output-dir",str(run),"--architecture",spec["architecture"],"--seed",str(spec["seed"]),"--max-epochs",str(MAX_EPOCHS),"--patience",str(PATIENCE),"--cpu-threads",str(CPU_THREADS),"--device","cuda","--dataset",str(DATASET),"--confirm-physical-order"]
    if spec["shuffle_labels"]: result.append("--shuffle-labels")
    return result


def start(cmd: list[str], out, err) -> subprocess.Popen:
    si=subprocess.STARTUPINFO(); si.dwFlags|=subprocess.STARTF_USESHOWWINDOW; si.wShowWindow=subprocess.SW_HIDE
    env=os.environ.copy()
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): env[name]="1"
    return subprocess.Popen(cmd,cwd=PROJECT,stdout=out,stderr=err,env=env,stdin=subprocess.DEVNULL,startupinfo=si,creationflags=subprocess.CREATE_NO_WINDOW)


def schedule() -> None:
    if not PYTHON.is_file() or not RUNNER.is_file() or not DATASET.is_file(): raise FileNotFoundError("Python310, runner, or dataset missing")
    if OUTPUT_ROOT.exists(): raise FileExistsError(f"refusing overwrite: {OUTPUT_ROOT}")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    todo=tasks()
    if len(todo)!=18 or len({task_id(x) for x in todo})!=18: raise RuntimeError("v16 matrix must have 18 unique tasks")
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=False); snap=snapshot(OUTPUT_ROOT,todo)
    jobs={task_id(x):{"spec":x,"status":"pending","run_dir":str(OUTPUT_ROOT/task_id(x)),"stdout_log":str(OUTPUT_ROOT/(task_id(x)+".stdout.log")),"stderr_log":str(OUTPUT_ROOT/(task_id(x)+".stderr.log")),"pid":None} for x in todo}
    manifest={"format":"c170-multiscale-nyquist-v16-driver","format_version":1,"status":"running","root":str(OUTPUT_ROOT),"started_utc":datetime.now(timezone.utc).isoformat(),"python":str(PYTHON),"dataset_sha256":snap["dataset_sha256"],"source_sha256":snap["source_sha256"],"concurrency":CONCURRENCY,"cpu_threads_per_run":CPU_THREADS,"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"memory_gate":{"used_limit_gib":MEMORY_LIMIT_GIB,"estimated_next_worker_gib":ESTIMATED_NEXT_WORKER_GIB},"memory_start":memory_state(),"test_metrics_computed":False,"jobs":jobs}
    mpath=OUTPUT_ROOT/"launcher_manifest.json"; write_json(mpath,manifest); pending=list(todo); active={}; failures=[]; last=0.0
    while pending or active:
        while pending and len(active)<CONCURRENCY and time.monotonic()-last>=1.5:
            mem=memory_state()
            if mem["used_gib"]+ESTIMATED_NEXT_WORKER_GIB>MEMORY_LIMIT_GIB:
                manifest["memory_last"]=mem; write_json(mpath,manifest); break
            spec=pending[0]; key=task_id(spec); run=Path(jobs[key]["run_dir"])
            if run.exists(): jobs[key].update({"status":"failed_preexisting_output","error":"refusing overwrite"}); failures.append(key); pending.pop(0); continue
            out=Path(jobs[key]["stdout_log"]).open("xb"); err=Path(jobs[key]["stderr_log"]).open("xb"); proc=start(command(spec,run),out,err)
            active[key]=(proc,out,err,time.monotonic()); jobs[key].update({"status":"running","pid":proc.pid,"started_utc":datetime.now(timezone.utc).isoformat(),"memory_at_start":mem}); pending.pop(0); last=time.monotonic(); write_json(mpath,manifest)
        done=[]
        for key,(proc,out,err,started) in list(active.items()):
            rc=proc.poll()
            if rc is None: continue
            out.close(); err.close(); jobs[key].update({"returncode":int(rc),"elapsed_seconds":time.monotonic()-started})
            try:
                if rc!=0: raise RuntimeError(f"worker exit code {rc}")
                jobs[key]["verified"]=verify(jobs[key]["spec"],Path(jobs[key]["run_dir"])); jobs[key]["status"]="completed"
            except Exception as exc:
                jobs[key].update({"status":"failed","error":f"{type(exc).__name__}: {exc}"}); failures.append(key)
            done.append(key)
        for key in done: del active[key]
        manifest["memory_last"]=memory_state(); manifest["completed_count"]=sum(j["status"]=="completed" for j in jobs.values()); manifest["failed_count"]=sum(j["status"].startswith("failed") for j in jobs.values()); manifest["pending_count"]=len(pending); manifest["active_count"]=len(active); write_json(mpath,manifest)
        if pending or active: time.sleep(1.0)
    manifest["status"]="completed" if not failures and all(j["status"]=="completed" for j in jobs.values()) else "failed"; manifest["finished_utc"]=datetime.now(timezone.utc).isoformat(); manifest["completed_count"]=sum(j["status"]=="completed" for j in jobs.values()); manifest["failed_count"]=sum(j["status"].startswith("failed") for j in jobs.values()); manifest["pending_count"]=len(pending); manifest["active_count"]=0; write_json(mpath,manifest)


if __name__=="__main__": schedule()
