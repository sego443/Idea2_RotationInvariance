"""Fixed five-replicate matched-null queue for the frozen F2 candidate."""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUTS = PROJECT_DIR / "outputs"
ROOT = OUTPUTS / "c170_f2_matched_nulls_20260929"
PYTHON = Path(r"C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe")
DATASET = PROJECT_DIR / "data" / "processed_c170_20260926" / "dataset.npz"
REPS = (17, 29, 43, 61, 89)
CG_SEEDS = (17, 29, 43, 61, 89)
POWER_SEEDS = (17, 29, 43, 61, 89, 101, 131, 173, 211, 257)
MAX_WORKERS = 6
MEMORY_ADMISSION_GIB = 24.0
POLL_SECONDS = 5

V11_ROOT = OUTPUTS / "c170_v11_complex_orbit_formal_20260929"
POWER_EXT_ROOT = OUTPUTS / "c170_v4v6_power_extensions_formal_20260929"
V7_ROOT = OUTPUTS / "c170_c4_kernel_ridge_v7_formal_20260929"

SOURCE_FILES = (
    Path("run_c170_f2_matched_null_adapter.py"),
    Path("launch_c170_frozen_f2_matched_nulls.py"),
    Path("run_c170_complex_orbit_experiment_v11.py"),
    Path("run_c170_temporal_phase_invariant_experiment_v6_power_shuffle.py"),
    Path("run_c170_ring_cuda_experiment_v3.py"),
    Path("run_c170_ring_cuda_experiment_v2.py"),
    Path("run_c170_experiment.py"),
    Path("run_c170_c4_kernel_ridge_v7.py"),
    Path("src/c170_complex_orbit_model_v11.py"),
    Path("src/c170_temporal_phase_invariant_model_v6_power_shuffle.py"),
    Path("src/c170_shared_scale_preprocessing_v3.py"),
    Path("src/c170_ring_representation.py"),
    Path("src/c170_c4_kernel_ridge_v7.py"),
    Path("src/rotation_poc.py"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_nn_run(run_dir: Path, expected_variant: str, expected_seed: int, expected_shuffle_seed: int) -> dict:
    manifest_path = run_dir / "run_manifest.json"
    marker_path = run_dir / "RUN_COMPLETE.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise RuntimeError(f"missing completed marker/manifest: {run_dir}")
    manifest, marker = read_json(manifest_path), read_json(marker_path)
    if marker.get("status") != "completed" or marker.get("test_metrics_computed") is not False:
        raise RuntimeError(f"not a valid test-sealed completed run: {run_dir}")
    if marker.get("manifest_sha256") != sha256(manifest_path):
        raise RuntimeError(f"manifest hash mismatch: {run_dir}")
    if marker.get("config_sha256") != manifest.get("config_sha256"):
        raise RuntimeError(f"config hash mismatch: {run_dir}")
    if manifest.get("status") != "completed":
        raise RuntimeError(f"manifest not completed: {run_dir}")
    config = manifest["config"]
    if config.get("variant") != expected_variant or int(config.get("seed", -1)) != expected_seed:
        raise RuntimeError(f"wrong run identity: {run_dir}")
    if manifest.get("split", {}).get("label_shuffle_seed") != expected_shuffle_seed:
        raise RuntimeError(f"wrong shuffle seed: {run_dir}")
    if config.get("shuffle_labels") is not True:
        raise RuntimeError(f"run is not a shuffled-label control: {run_dir}")
    if expected_variant == "learned_power_shuffle" and config.get("active_feature_indices") != list(range(9)):
        raise RuntimeError(f"power-only feature mask mismatch: {run_dir}")
    artifacts = marker.get("artifact_sha256", {})
    for key, expected_hash in artifacts.items():
        item = manifest.get("artifacts", {}).get(key)
        if not item:
            raise RuntimeError(f"artifact entry missing for {key}: {run_dir}")
        artifact_path = run_dir / item["path"]
        if not artifact_path.is_file() or sha256(artifact_path) != expected_hash or item.get("sha256") != expected_hash:
            raise RuntimeError(f"artifact hash mismatch for {key}: {run_dir}")
    return {
        "run_dir": str(run_dir.resolve()),
        "variant": expected_variant,
        "seed": expected_seed,
        "shuffle_seed": expected_shuffle_seed,
        "config_sha256": marker["config_sha256"],
        "manifest_sha256": marker["manifest_sha256"],
        "artifact_sha256": artifacts,
        "selected_epoch_one_based": marker.get("selected_epoch_one_based"),
    }


def system_used_gib() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return (status.ullTotalPhys - status.ullAvailPhys) / float(1024**3)


def shuffle_map_from_nn(run_dir: Path) -> dict[tuple[int, int], int]:
    manifest = read_json(run_dir / "run_manifest.json")
    names = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    return {
        (int(row["distance_mm"]), int(row["block"])): names[row["training_target_direction"]]
        for row in manifest["split"]["label_shuffle_mapping"]
    }


def verify_and_prepare_snapshot() -> tuple[list[dict], list[dict], list[dict]]:
    if not PYTHON.is_file() or not DATASET.is_file():
        raise FileNotFoundError(f"missing frozen Python or dataset: {PYTHON}, {DATASET}")
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite output root: {ROOT}")

    v7_manifest_path = V7_ROOT / "run_manifest.json"
    v7_manifest = read_json(v7_manifest_path)
    v7_config = v7_manifest["config"]
    if v7_config.get("dataset_sha256") != sha256(DATASET):
        raise RuntimeError("dataset hash differs from frozen v7 split")
    if v7_config.get("candidate_count_per_arm") != 108:
        raise RuntimeError("v7 did not use the frozen 108-candidate budget")
    v7_maps = v7_manifest["split_metadata"]["shuffle_mapping"]
    v7_candidates = list(csv.DictReader((V7_ROOT / "selected_validation_summary.csv").open(encoding="utf-8-sig", newline="")))
    if len(v7_candidates) != 7:
        raise RuntimeError(f"expected seven v7 selected model rows (one per fixed null/real arm), got {len(v7_candidates)}")

    label_maps: list[dict] = []
    reused: list[dict] = []
    tasks: list[dict] = []
    v7_reused: list[dict] = []
    direction_to_int = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
    for rep in REPS:
        v7_map = {
            (int(row["position_mm"]), int(row["block"])): int(row["pseudo_label"])
            for row in v7_maps[str(rep)]
        }
        if len(v7_map) != 44:
            raise RuntimeError(f"rep {rep}: v7 shuffle map does not contain 44 training blocks")
        rep_rows = [{"distance_mm": pos, "block": block, "target_direction_id": target} for (pos, block), target in sorted(v7_map.items())]
        label_maps.append({
            "rep": rep,
            "shuffle_seed": rep + 10_000,
            "mapping_sha256": canonical_sha256(rep_rows),
            "training_block_targets": rep_rows,
        })

        # v7 has a separate 108-candidate LOPO selection for each fixed null map.
        selected_rows = [row for row in v7_candidates if row["mode"] == f"group_shuffle_{rep}"]
        if len(selected_rows) != 1:
            raise RuntimeError(f"rep {rep}: expected one selected v7 model, got {len(selected_rows)}")
        selected = selected_rows[0]
        model_path = V7_ROOT / selected["state_npz"]
        expected_model_hash = v7_manifest["artifacts"].get(selected["state_npz"])
        if not model_path.is_file() or sha256(model_path) != expected_model_hash:
            raise RuntimeError(f"rep {rep}: v7 selected model hash mismatch")
        prediction_rows = []
        with (V7_ROOT / "validation_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["mode"] == f"group_shuffle_{rep}":
                    prediction_rows.append(row)
        if len(prediction_rows) != 320:
            raise RuntimeError(f"rep {rep}: expected 320 v7 null validation predictions, got {len(prediction_rows)}")
        v7_reused.append({
            "rep": rep,
            "mode": selected["mode"],
            "selected_preprocessor": selected["preprocessor"],
            "selected_view": selected["view"],
            "selected_sigma_factor": float(selected["selected_sigma_factor"]),
            "selected_alpha": float(selected["selected_alpha"]),
            "candidate_count": 108,
            "model_path": str(model_path.resolve()),
            "model_sha256": expected_model_hash,
            "validation_prediction_rows": 320,
            "validation_csv_path": str((V7_ROOT / "validation_predictions.csv").resolve()),
            "validation_csv_sha256": sha256(V7_ROOT / "validation_predictions.csv"),
            "label_mapping_sha256": canonical_sha256(rep_rows),
        })

        cg_source = V11_ROOT / "complex_gram_shuffle__seed-17"
        power_source = POWER_EXT_ROOT / f"v6_learned_power_shuffle__seed-{rep}"
        # Validate existing candidates before deciding whether they are reusable.
        power_info = validate_nn_run(power_source, "learned_power_shuffle", rep, rep + 10_000)
        power_map = shuffle_map_from_nn(power_source)
        if power_map != v7_map:
            raise RuntimeError(f"rep {rep}: power existing permutation differs from v7 mapping")
        power_info["family"] = "learned_power"
        power_info["rep"] = rep
        power_info["mapping_sha256"] = canonical_sha256(rep_rows)
        reused.append(power_info)

        if rep == 17:
            cg_info = validate_nn_run(cg_source, "complex_gram_shuffle", 17, 17 + 10_000)
            cg_map = shuffle_map_from_nn(cg_source)
            if cg_map != v7_map:
                raise RuntimeError("rep 17: complex_gram existing permutation differs from v7 mapping")
            cg_info["family"] = "complex_gram"
            cg_info["rep"] = rep
            cg_info["mapping_sha256"] = canonical_sha256(rep_rows)
            reused.append(cg_info)

        for seed in CG_SEEDS:
            if not (rep == 17 and seed == 17):
                tasks.append({"rep": rep, "family": "complex_gram", "variant": "complex_gram_shuffle", "seed": seed})
        for seed in POWER_SEEDS:
            if seed != rep:
                tasks.append({"rep": rep, "family": "learned_power", "variant": "learned_power_shuffle", "seed": seed})

    if len(tasks) != 69 or len(reused) != 6 or len(v7_reused) != 5:
        raise RuntimeError(f"unexpected queue size: tasks={len(tasks)}, reused neural={len(reused)}, v7={len(v7_reused)}")

    ROOT.mkdir(parents=True, exist_ok=False)
    snapshot = ROOT / "source_snapshot"
    for relative in SOURCE_FILES:
        source = PROJECT_DIR / relative
        if not source.is_file():
            raise FileNotFoundError(f"source snapshot input missing: {source}")
        target = snapshot / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"source snapshot copy hash mismatch: {relative}")

    v7_summary = V7_ROOT / "selected_validation_summary.csv"
    v7_predictions = V7_ROOT / "validation_predictions.csv"
    source_rows = []
    for relative in SOURCE_FILES:
        source_rows.append({"path": relative.as_posix(), "sha256": sha256(PROJECT_DIR / relative)})
    atomic_json(snapshot / "snapshot_manifest.json", {
        "format": "c170-frozen-f2-matched-null-source-snapshot",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_files": source_rows,
        "dataset": {"path": str(DATASET.resolve()), "sha256": sha256(DATASET)},
        "v7_selected_summary_sha256": sha256(v7_summary),
        "v7_validation_predictions_sha256": sha256(v7_predictions),
    })
    atomic_json(ROOT / "matched_null_mapping.json", {
        "format": "c170-f2-shared-whole-block-null-mappings",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rep_seed_rule": "label_shuffle_seed = rep + 10000; same mapping across all 15 neural members and v7 member",
        "maps": label_maps,
        "reused_neural_members": reused,
        "reused_v7_members": v7_reused,
    })
    experiment = {
        "format": "c170-frozen-f2-matched-null-experiment",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frozen_real_candidate": {
            "complex_gram_v11_seeds": list(CG_SEEDS),
            "learned_power_v6_seeds": list(POWER_SEEDS),
            "v7_mode": "group_true",
            "fusion": "one-hot average within each family; then equal 1/3 family average; argmax lowest class id on ties",
            "validation_macro_f1_exploration_only": 0.8268715072,
            "test_metrics_computed": False,
        },
        "null_protocol": {
            "reps": list(REPS),
            "per_rep_neural_members": {"complex_gram": list(CG_SEEDS), "learned_power": list(POWER_SEEDS)},
            "v7_member": "same rep's group_shuffle selected from the original 108-candidate LOPO search",
            "one_mapping_per_rep_shared_across_all_families": True,
            "initialization_and_sampler_seed": "unchanged per model; independent from shuffle rep",
            "max_workers": MAX_WORKERS,
            "cpu_threads_per_worker": 1,
            "memory_admission_system_used_gib_below": MEMORY_ADMISSION_GIB,
            "test_metrics_computed": False,
        },
        "new_training_tasks": tasks,
        "new_training_task_count": len(tasks),
        "reused_neural_member_count": len(reused),
        "reused_v7_member_count": len(v7_reused),
        "source_snapshot": "source_snapshot/snapshot_manifest.json",
        "dataset_sha256": sha256(DATASET),
        "frozen_v7_sources": {
            "run_manifest_sha256": sha256(v7_manifest_path),
            "summary_csv_sha256": sha256(v7_summary),
            "validation_predictions_sha256": sha256(v7_predictions),
        },
    }
    atomic_json(ROOT / "experiment_manifest.json", experiment)
    return tasks, reused, v7_reused


def verify_new_run(run_dir: Path, variant: str, seed: int, rep: int) -> dict:
    result = validate_nn_run(run_dir, variant, seed, rep + 10_000)
    manifest = read_json(run_dir / "run_manifest.json")
    if int(manifest["config"].get("matched_null_shuffle_rep", -1)) != rep:
        raise RuntimeError(f"wrong shared null rep in config: {run_dir}")
    if int(manifest["config"].get("matched_null_shuffle_seed", -1)) != rep + 10_000:
        raise RuntimeError(f"wrong target shuffle seed in config: {run_dir}")
    if manifest.get("split", {}).get("matched_null_shuffle_rep") != rep:
        raise RuntimeError(f"wrong split null rep: {run_dir}")
    return result


def main() -> int:
    tasks, reused, v7_reused = verify_and_prepare_snapshot()
    print(f"NULL_QUEUE_READY root={ROOT} new_tasks={len(tasks)} reuse_neural={len(reused)} reuse_v7={len(v7_reused)} workers={MAX_WORKERS}", flush=True)
    state = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "root": str(ROOT.resolve()), "tasks": []}
    pending = list(tasks)
    active: dict[int, tuple[dict, subprocess.Popen, object, Path]] = {}
    completed: list[dict] = []
    failed: list[dict] = []
    while pending or active:
        # Reap first, then admit only while Windows reports total system use below the cap.
        for pid, (task, process, log_handle, run_dir) in list(active.items()):
            rc = process.poll()
            if rc is None:
                continue
            log_handle.close()
            del active[pid]
            try:
                if rc != 0:
                    raise RuntimeError(f"worker exit code {rc}")
                result = verify_new_run(run_dir, task["variant"], task["seed"], task["rep"])
                result.update({"family": task["family"], "rep": task["rep"], "worker_pid": pid})
                completed.append(result)
                print(f"RUN_OK rep={task['rep']} family={task['family']} seed={task['seed']} completed={len(completed)}/{len(tasks)}", flush=True)
            except Exception as error:
                failure = {"task": task, "worker_pid": pid, "returncode": rc, "error": str(error), "run_dir": str(run_dir.resolve())}
                failed.append(failure)
                print(f"RUN_FAIL rep={task['rep']} family={task['family']} seed={task['seed']} rc={rc} error={error}", flush=True)
            state["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            state["completed"] = completed
            state["failed"] = failed
            state["running"] = [
                {"task": value[0], "pid": key, "run_dir": str(value[3].resolve())} for key, value in active.items()
            ]
            state["pending_count"] = len(pending)
            atomic_json(ROOT / "driver_state.json", state)

        while pending and len(active) < MAX_WORKERS:
            used = system_used_gib()
            if used >= MEMORY_ADMISSION_GIB:
                if not active:
                    print(f"MEMORY_WAIT used_gib={used:.2f} cap_gib={MEMORY_ADMISSION_GIB:.2f}", flush=True)
                break
            task = pending.pop(0)
            run_dir = ROOT / f"rep-{task['rep']}" / f"{task['family']}__seed-{task['seed']}"
            if run_dir.exists():
                raise FileExistsError(f"refusing to overwrite run directory: {run_dir}")
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            stdout_path = ROOT / "logs" / f"rep-{task['rep']}_{task['family']}_seed-{task['seed']}.stdout.log"
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = stdout_path.open("wb")
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
            command = [
                str(PYTHON), "-u", str(PROJECT_DIR / "run_c170_f2_matched_null_adapter.py"),
                "--output-dir", str(run_dir), "--variant", task["variant"], "--seed", str(task["seed"]),
                "--shuffle-seed", str(task["rep"]), "--max-epochs", "160", "--patience", "25",
                "--cpu-threads", "1", "--device", "cuda", "--dataset", str(DATASET),
                "--confirm-physical-order",
            ]
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_DIR),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            active[process.pid] = (task, process, log_handle, run_dir)
            print(f"RUN_START pid={process.pid} rep={task['rep']} family={task['family']} seed={task['seed']} workers={len(active)} system_used_gib={used:.2f} stdout={stdout_path}", flush=True)

        state["updated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        state["completed"] = completed
        state["failed"] = failed
        state["running"] = [
            {"task": value[0], "pid": key, "run_dir": str(value[3].resolve())} for key, value in active.items()
        ]
        state["pending_count"] = len(pending)
        atomic_json(ROOT / "driver_state.json", state)
        if pending or active:
            time.sleep(POLL_SECONDS)

    result = {
        "format": "c170-frozen-f2-matched-null-complete",
        "status": "completed" if not failed and len(completed) == len(tasks) else "failed",
        "created_utc": state["created_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "new_tasks_expected": len(tasks),
        "new_tasks_completed": len(completed),
        "new_tasks_failed": failed,
        "reused_neural_members": reused,
        "reused_v7_members": v7_reused,
        "test_metrics_computed": False,
    }
    atomic_json(ROOT / "NULL_RUNS_COMPLETE.json", result)
    print(f"NULL_QUEUE_FINISHED status={result['status']} completed={len(completed)}/{len(tasks)} failed={len(failed)}", flush=True)
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
