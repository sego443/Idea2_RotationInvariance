"""Fixed 36-run v17 complex_gram regularization grid; stdout-only monitoring."""
from __future__ import annotations
import ctypes, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import torch

PROJECT=Path(__file__).resolve().parent
PYTHON=Path(r"C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe")
DATASET=PROJECT/"data/processed_c170_20260926/dataset.npz"
RUNNER=PROJECT/"run_c170_regularization_v17.py"
OUTPUT_ROOT=PROJECT/"outputs/c170_complex_regularization_v17_recovery2_20260930"
WEIGHT_DECAYS=(0.01,0.1); DROPOUTS=(0.2,0.4,0.6); SEEDS=(17,29,43,61,89)
CONCURRENCY=6; CPU_THREADS=1; MAX_EPOCHS=160; PATIENCE=25
MEMORY_LIMIT_GIB=24.0; ESTIMATED_NEXT_WORKER_GIB=2.0

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("ullAvailExtendedVirtual",ctypes.c_ulonglong)]

def memory_state():
    s=MEMORYSTATUSEX(); s.dwLength=ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)): raise OSError("GlobalMemoryStatusEx failed")
    g=float(1024**3); return {"total_gib":s.ullTotalPhys/g,"available_gib":s.ullAvailPhys/g,"used_gib":(s.ullTotalPhys-s.ullAvailPhys)/g,"load_percent":float(s.dwMemoryLoad)}

def sha256(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): d.update(chunk)
    return d.hexdigest()

def canonical(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()
def write_json(path,value):
    p=Path(path); tmp=p.with_name(p.name+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f: json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,p)
def tok(x): return f"{x:g}".replace(".","p")
def tasks():
    out=[]
    for wd in WEIGHT_DECAYS:
        for drop in DROPOUTS:
            out.extend({"weight_decay":wd,"dropout":drop,"seed":seed,"shuffle_labels":False} for seed in SEEDS)
            out.append({"weight_decay":wd,"dropout":drop,"seed":17,"shuffle_labels":True})
    for x in out: x.update({"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"cpu_threads":CPU_THREADS,"device":"cuda"})
    return out
def task_id(s): return f"wd-{tok(s['weight_decay'])}__drop-{tok(s['dropout'])}__seed-{s['seed']}"+("__shuffle" if s["shuffle_labels"] else "")
def sources():
    names=("launch_c170_regularization_v17.py","launch_c170_regularization_v17_recovery.py","launch_c170_regularization_v17_recovery2.py","run_c170_regularization_v17.py","run_c170_complex_orbit_experiment_v11.py","src/c170_complex_orbit_model_v11.py","src/c170_harmonic_phase_model_v4.py","run_c170_ring_cuda_experiment_v3.py","src/c170_shared_scale_preprocessing_v3.py","src/c170_ring_representation.py","run_c170_ring_cuda_experiment_v2.py","run_c170_experiment.py","src/rotation_poc.py")
    return {n:PROJECT/n for n in names}
def snapshot(root,grid):
    code=root/"source_snapshot"/"code"; code.mkdir(parents=True,exist_ok=False); hashes={}
    for rel,src in sources().items():
        if not src.is_file(): raise FileNotFoundError(src)
        dst=code/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst); hashes[rel.replace("/","\\")]=sha256(dst)
    value={"format":"c170-complex-regularization-v17-source-snapshot","source_sha256":hashes,"dataset_path":str(DATASET.resolve()),"dataset_sha256":sha256(DATASET),"tasks":grid,"protocol":{"train":879,"validation":320,"test_policy":"test not preprocessed, predicted, or scored","backbone":"frozen v11 complex_gram; no adapter","weight_decay_grid":WEIGHT_DECAYS,"dropout_grid":DROPOUTS,"weight_decay":"AdamW decoupled weight decay","dropout":"descriptor dropout before final linear classifier","max_epochs":MAX_EPOCHS,"patience":PATIENCE,"concurrency":CONCURRENCY,"cpu_threads_per_run":CPU_THREADS,"memory_gate":"start only when system used GiB + 2 <= 24","baseline_reused":"v11 complex_gram, wd=.001, dropout=.2, seeds 17/29/43/61/89"},"python":sys.version.split()[0],"torch":torch.__version__,"cuda_build":torch.version.cuda,"cuda_device":torch.cuda.get_device_name(0),"created_utc":datetime.now(timezone.utc).isoformat(),"test_metrics_computed":False}
    write_json(root/"source_snapshot"/"snapshot_manifest.json",value); return value
def verify(spec,run,snapshot_hashes):
    mp=run/"run_manifest.json"; cp=run/"RUN_COMPLETE.json"
    if not mp.is_file() or not cp.is_file(): raise RuntimeError("completion marker/manifest missing")
    m=json.loads(mp.read_text(encoding="utf-8-sig")); c=json.loads(cp.read_text(encoding="utf-8-sig")); cfg=m.get("config",{}); arch=cfg.get("architecture",{}); reg=cfg.get("v17_regularization",{})
    digest=canonical(cfg)
    if m.get("format")!="c170-complex-regularization-v17-run" or c.get("format")!="c170-complex-regularization-v17-complete" or m.get("status")!="completed" or c.get("status")!="completed": raise RuntimeError("run/marker status or format mismatch")
    if digest!=m.get("config_sha256") or digest!=c.get("config_sha256") or sha256(mp)!=c.get("manifest_sha256"): raise RuntimeError("config or manifest hash mismatch")
    if cfg.get("architecture",{}).get("name")!="complex_gram" or cfg.get("seed")!=spec["seed"] or cfg.get("shuffle_labels") is not spec["shuffle_labels"]: raise RuntimeError("variant/seed config mismatch")
    if cfg.get("weight_decay")!=spec["weight_decay"] or arch.get("dropout")!=spec["dropout"] or reg.get("weight_decay")!=spec["weight_decay"] or reg.get("dropout")!=spec["dropout"]: raise RuntimeError("regularization config mismatch")
    expected_variant=f"c170_complex_orbit_classifier_v17_complex_gram_wd{tok(spec['weight_decay'])}_dropout{tok(spec['dropout'])}"
    if cfg.get("model_variant")!=expected_variant: raise RuntimeError("model variant mismatch")
    if cfg.get("max_epochs")!=MAX_EPOCHS or cfg.get("patience")!=PATIENCE or cfg.get("device") not in {"cuda","cuda:0"}: raise RuntimeError("training protocol mismatch")
    if m.get("test_metrics_computed") is not False or c.get("test_metrics_computed") is not False: raise RuntimeError("test metrics must remain false")
    prov=m.get("provenance",{}).get("code_sha256",{})
    if prov.get("v17_regularization_wrapper")!=snapshot_hashes["run_c170_regularization_v17.py"]: raise RuntimeError("v17 wrapper provenance mismatch")
    if prov.get("runner")!=snapshot_hashes["run_c170_complex_orbit_experiment_v11.py"] or prov.get("model")!=snapshot_hashes["src\\c170_complex_orbit_model_v11.py"]: raise RuntimeError("base source provenance mismatch")
    artifacts={}
    for key,expected in c.get("artifact_sha256",{}).items():
        entry=m.get("artifacts",{}).get(key,{}); path=run/entry.get("path",""); artifacts[key]=bool(entry.get("sha256")==expected and path.is_file() and sha256(path)==expected)
    if set(artifacts)!={"best_checkpoint","safe_state_dict","learning_curve","validation_predictions"} or not all(artifacts.values()): raise RuntimeError("artifact hash mismatch")
    return {"config_sha256":digest,"manifest_sha256":c["manifest_sha256"],"artifact_sha256":artifacts}
def command(spec,run):
    variant="complex_gram_shuffle" if spec["shuffle_labels"] else "complex_gram"
    return [str(PYTHON),str(RUNNER),"--v17-weight-decay",str(spec["weight_decay"]),"--v17-dropout",str(spec["dropout"]),"--output-dir",str(run),"--variant",variant,"--seed",str(spec["seed"]),"--max-epochs",str(MAX_EPOCHS),"--patience",str(PATIENCE),"--cpu-threads",str(CPU_THREADS),"--device","cuda","--dataset",str(DATASET),"--confirm-physical-order"]
def start(cmd,out,err):
    si=subprocess.STARTUPINFO(); si.dwFlags|=subprocess.STARTF_USESHOWWINDOW; si.wShowWindow=subprocess.SW_HIDE; env=os.environ.copy()
    for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"): env[k]="1"
    return subprocess.Popen(cmd,cwd=PROJECT,stdout=out,stderr=err,env=env,stdin=subprocess.DEVNULL,startupinfo=si,creationflags=subprocess.CREATE_NO_WINDOW)
def schedule():
    if not PYTHON.is_file() or not RUNNER.is_file() or not DATASET.is_file(): raise FileNotFoundError("Python310, v17 runner, or dataset missing")
    if OUTPUT_ROOT.exists(): raise FileExistsError(f"refusing overwrite: {OUTPUT_ROOT}")
    if not torch.cuda.is_available(): raise RuntimeError("launcher CUDA unavailable")
    grid=tasks()
    if len(grid)!=36 or len({task_id(x) for x in grid})!=36: raise RuntimeError("fixed v17 grid is not 36 unique tasks")
    reuse_sources={
        "wd-0p01__drop-0p2__seed-17":"recovery",
        "wd-0p01__drop-0p2__seed-29":"formal",
        "wd-0p01__drop-0p2__seed-43":"recovery",
        "wd-0p01__drop-0p2__seed-61":"recovery",
        "wd-0p01__drop-0p2__seed-89":"recovery",
        "wd-0p01__drop-0p2__seed-17__shuffle":"recovery",
        "wd-0p01__drop-0p4__seed-17":"recovery",
        "wd-0p01__drop-0p4__seed-29":"recovery",
        "wd-0p01__drop-0p4__seed-43":"recovery",
        "wd-0p01__drop-0p4__seed-61":"recovery",
        "wd-0p01__drop-0p4__seed-89":"recovery",
        "wd-0p01__drop-0p4__seed-17__shuffle":"recovery",
        "wd-0p01__drop-0p6__seed-17":"recovery",
        "wd-0p01__drop-0p6__seed-29":"recovery",
        "wd-0p01__drop-0p6__seed-43":"recovery",
        "wd-0p01__drop-0p6__seed-61":"recovery",
        "wd-0p01__drop-0p6__seed-89":"recovery",
        "wd-0p01__drop-0p6__seed-17__shuffle":"recovery",
        "wd-0p1__drop-0p2__seed-29":"recovery",
    }
    prior_roots={"formal":PROJECT/"outputs/c170_complex_regularization_v17_formal_20260930","recovery":PROJECT/"outputs/c170_complex_regularization_v17_recovery_20260930"}
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=False); snap=snapshot(OUTPUT_ROOT,grid)
    jobs={task_id(x):{"spec":x,"status":"pending","run_dir":str(OUTPUT_ROOT/task_id(x)),"stdout_log":str(OUTPUT_ROOT/(task_id(x)+".stdout.log")),"stderr_log":str(OUTPUT_ROOT/(task_id(x)+".stderr.log")),"pid":None} for x in grid}
    reused=[]
    for key,source_name in reuse_sources.items():
        run=prior_roots[source_name]/key
        old_driver=json.loads((prior_roots[source_name]/"launcher_manifest.json").read_text(encoding="utf-8-sig")); old_job=old_driver.get("jobs",{}).get(key,{})
        if not (run/"RUN_COMPLETE.json").is_file() or not (run/"run_manifest.json").is_file(): raise RuntimeError(f"pre-screened reusable run disappeared: {run}")
        jobs[key].update({"status":"completed","run_dir":str(run),"stdout_log":old_job.get("stdout_log"),"stderr_log":old_job.get("stderr_log"),"pid":None,"reused_from":str(run),"reuse_source_status":old_job.get("status"),"reuse_source_returncode":old_job.get("returncode")})
        reused.append({"task_id":key,"run_dir":str(run),"source_root":source_name,"previous_driver_status":old_job.get("status"),"previous_driver_returncode":old_job.get("returncode")})
    new_jobs=[task_id(x) for x in grid if task_id(x) not in reuse_sources]
    recovery_index={"format":"c170-complex-regularization-v17-recovery-index","format_version":1,"source_roots":{k:str(v) for k,v in prior_roots.items()},"reused_runs":reused,"new_runs":[{"task_id":k,"run_dir":str(OUTPUT_ROOT/k)} for k in new_jobs],"pre_screen":"RUN_COMPLETE.json and run_manifest.json present; v17 config and completion status matched the fixed 36-task grid; no artifact rehash performed here.","test_metrics_computed":False}
    write_json(OUTPUT_ROOT/"recovery_index.json",recovery_index)
    manifest={"format":"c170-complex-regularization-v17-driver","format_version":1,"status":"running","root":str(OUTPUT_ROOT),"started_utc":datetime.now(timezone.utc).isoformat(),"python":str(PYTHON),"dataset_sha256":snap["dataset_sha256"],"source_sha256":snap["source_sha256"],"concurrency":CONCURRENCY,"cpu_threads_per_run":CPU_THREADS,"max_epochs":MAX_EPOCHS,"patience":PATIENCE,"memory_gate":{"used_limit_gib":MEMORY_LIMIT_GIB,"estimated_next_worker_gib":ESTIMATED_NEXT_WORKER_GIB},"memory_start":memory_state(),"test_metrics_computed":False,"completed_count":len(reused),"reused_count":len(reused),"pending_count":len(new_jobs),"recovery_index":"recovery_index.json","jobs":jobs}
    mpath=OUTPUT_ROOT/"launcher_manifest.json"; write_json(mpath,manifest); pending=[x for x in grid if jobs[task_id(x)]["status"]!="completed"]; active={}; failures=[]; last=0.0
    while pending or active:
        while pending and len(active)<CONCURRENCY and time.monotonic()-last>=1.5:
            mem=memory_state()
            if mem["used_gib"]+ESTIMATED_NEXT_WORKER_GIB>MEMORY_LIMIT_GIB: manifest["memory_last"]=mem; write_json(mpath,manifest); break
            spec=pending[0]; key=task_id(spec); run=Path(jobs[key]["run_dir"])
            if run.exists(): jobs[key].update({"status":"failed_preexisting_output","error":"refusing overwrite"}); failures.append(key); pending.pop(0); continue
            out=Path(jobs[key]["stdout_log"]).open("xb"); err=Path(jobs[key]["stderr_log"]).open("xb"); proc=start(command(spec,run),out,err)
            active[key]=(proc,out,err,time.monotonic()); jobs[key].update({"status":"running","pid":proc.pid,"started_utc":datetime.now(timezone.utc).isoformat(),"memory_at_start":mem}); pending.pop(0); last=time.monotonic(); write_json(mpath,manifest)
            print(f"START {key} pid={proc.pid} used_gib={mem['used_gib']:.2f} completed={manifest.get('completed_count',0)}",flush=True)
        done=[]
        for key,(proc,out,err,started) in list(active.items()):
            rc=proc.poll()
            if rc is None: continue
            out.close(); err.close(); jobs[key].update({"returncode":int(rc),"elapsed_seconds":time.monotonic()-started})
            try:
                if rc!=0: raise RuntimeError(f"worker exit code {rc}")
                jobs[key]["verified"]=verify(jobs[key]["spec"],Path(jobs[key]["run_dir"]),snap["source_sha256"]); jobs[key]["status"]="completed"
            except Exception as exc: jobs[key].update({"status":"failed","error":f"{type(exc).__name__}: {exc}"}); failures.append(key)
            done.append(key); print(f"DONE {key} rc={rc} status={jobs[key]['status']}",flush=True)
        for key in done: del active[key]
        manifest["memory_last"]=memory_state(); manifest["completed_count"]=sum(j["status"]=="completed" for j in jobs.values()); manifest["failed_count"]=sum(j["status"].startswith("failed") for j in jobs.values()); manifest["pending_count"]=len(pending); manifest["active_count"]=len(active); write_json(mpath,manifest)
        if pending or active: time.sleep(1.0)
    manifest["status"]="completed" if not failures and all(j["status"]=="completed" for j in jobs.values()) else "failed"; manifest["finished_utc"]=datetime.now(timezone.utc).isoformat(); manifest["completed_count"]=sum(j["status"]=="completed" for j in jobs.values()); manifest["failed_count"]=sum(j["status"].startswith("failed") for j in jobs.values()); manifest["pending_count"]=0; manifest["active_count"]=0; write_json(mpath,manifest)
    print(f"V17_QUEUE_DONE completed={manifest['completed_count']} failed={manifest['failed_count']} root={OUTPUT_ROOT}",flush=True)
if __name__=="__main__": schedule()
