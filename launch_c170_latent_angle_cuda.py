"""Run the frozen C170 latent-angle matrix on CUDA with safe resume and GPU sampling."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = PROJECT_DIR / "run_c170_latent_angle_experiment.py"
MODEL_PATH = PROJECT_DIR / "src" / "c170_latent_angle.py"
DEFAULT_DATASET_PATH = PROJECT_DIR / "data" / "processed_c170_20260926" / "dataset.npz"
DEFAULT_PYTHON = Path(r"C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe")
FORMAL_SEEDS = (17, 29, 43, 61, 89)
PROBE_SEEDS = (17, 29, 43, 61, 89, 101, 113, 127)
FORMAL_VARIANTS = ("mlp", "no_angle", "full", "no_rms", "shuffle")
MANIFEST_NAME = "launcher_manifest.json"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_COMPLETE_NAME = "RUN_COMPLETE.json"
RESOURCE_SAMPLES_NAME = "gpu_samples.csv"
FORMAT = "c170-latent-angle-launcher"
FORMAT_VERSION = 1
RESOURCE_SAMPLE_SECONDS = 2.0
GPU_QUERY = "timestamp,index,uuid,utilization.gpu,memory.used,memory.total"
_ACTIVE_CHILDREN: dict[str, dict[str, Any]] = {}
CODE_FILES = {
    "dataset_helpers": PROJECT_DIR / "run_c170_experiment.py",
    "preprocessing_helpers": PROJECT_DIR / "run_c170_ring_experiment.py",
    "encoder_helpers": PROJECT_DIR / "src" / "rotation_poc.py",
}
ANGLE_WEIGHTS = {"mlp": 0.0, "no_angle": 0.0, "full": 0.2, "no_rms": 0.2, "shuffle": 0.2}
SMOOTHNESS_WEIGHTS = {"mlp": 0.0, "no_angle": 0.001, "full": 0.001, "no_rms": 0.001, "shuffle": 0.001}


class LauncherError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSpec:
    variant: str
    seed: int
    max_epochs: int
    patience: int

    @property
    def job_id(self) -> str:
        return f"{self.variant}__seed-{self.seed}"


def build_run_specs(suite: str) -> list[RunSpec]:
    if suite == "formal":
        return [
            RunSpec(variant, seed, max_epochs=160, patience=25)
            for variant in FORMAL_VARIANTS
            for seed in FORMAL_SEEDS
        ]
    if suite == "probe":
        return [RunSpec("full", seed, max_epochs=5, patience=5) for seed in PROBE_SEEDS]
    raise ValueError(f"unknown suite: {suite}")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LauncherError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LauncherError(f"expected a JSON object in {path}")
    return value


def _read_learning_curve(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise LauncherError(f"cannot read learning curve {path}: {exc}") from exc
    if not rows:
        raise LauncherError(f"learning curve is empty: {path}")
    required = {"epoch", "optimizer_updates", "epoch_seconds"}
    if not required.issubset(rows[0]):
        raise LauncherError(f"learning curve is missing columns {sorted(required - set(rows[0]))}: {path}")
    for row in rows:
        try:
            epoch = int(row["epoch"])
            updates = int(row["optimizer_updates"])
            seconds = float(row["epoch_seconds"])
        except (TypeError, ValueError) as exc:
            raise LauncherError(f"invalid epoch timing row in {path}: {row}") from exc
        if epoch <= 0 or updates <= 0 or not math.isfinite(seconds) or seconds < 0:
            raise LauncherError(f"invalid epoch timing values in {path}: {row}")
    return rows


def validate_completed_run(
    run_dir: Path,
    expected: Mapping[str, Any],
    expected_config_sha256: str | None = None,
    expected_code_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the runner's completion marker, requested config, outputs, and checkpoint hash."""
    manifest_path = run_dir / RUN_MANIFEST_NAME
    complete_path = run_dir / RUN_COMPLETE_NAME
    manifest = _read_json(manifest_path)
    complete = _read_json(complete_path)
    if manifest.get("format") != "c170-latent-angle-run" or manifest.get("format_version") != 1:
        raise LauncherError(f"unexpected runner manifest format in {manifest_path}")
    if manifest.get("status") != "completed":
        raise LauncherError(f"run is not marked completed: {run_dir}")
    if manifest.get("test_metrics_computed") is not False:
        raise LauncherError(f"runner manifest reports test metrics were computed: {run_dir}")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise LauncherError(f"runner config is missing or malformed: {manifest_path}")
    digest = config_sha256(config)
    if manifest.get("config_sha256") != digest:
        raise LauncherError(f"runner config hash does not match its manifest: {run_dir}")
    for key, expected_value in expected.items():
        if config.get(key) != expected_value:
            raise LauncherError(
                f"run config mismatch for {key} in {run_dir}: expected {expected_value!r}, got {config.get(key)!r}"
            )
    if expected_config_sha256 is not None and digest != expected_config_sha256:
        raise LauncherError(f"run config changed since the prior launcher record: {run_dir}")
    if expected_code_sha256 is not None:
        provenance = manifest.get("provenance")
        actual_code_sha256 = provenance.get("code_sha256") if isinstance(provenance, dict) else None
        if actual_code_sha256 != dict(expected_code_sha256):
            raise LauncherError(f"runner provenance code hashes do not match current sources: {run_dir}")

    if complete.get("format") != "c170-latent-angle-complete" or complete.get("format_version") != 1:
        raise LauncherError(f"unexpected runner completion format in {complete_path}")
    if complete.get("status") != "completed" or complete.get("test_metrics_computed") is not False:
        raise LauncherError(f"invalid runner completion state in {complete_path}")
    if complete.get("config_sha256") != digest:
        raise LauncherError(f"completion marker config hash does not match: {run_dir}")

    checkpoint_path = run_dir / "best_checkpoint.pt"
    required_artifacts = (
        checkpoint_path,
        run_dir / "learning_curve.csv",
        run_dir / "validation_predictions.csv",
    )
    for artifact in required_artifacts:
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise LauncherError(f"required run artifact is missing or empty: {artifact}")
    checkpoint_digest = sha256_file(checkpoint_path)
    if complete.get("checkpoint_sha256") != checkpoint_digest:
        raise LauncherError(f"checkpoint checksum does not match completion marker: {run_dir}")
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise LauncherError(f"run manifest checksum does not match completion marker: {run_dir}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise LauncherError(f"runner artifact summaries are missing: {manifest_path}")
    for key, fallback_name in (
        ("checkpoint", "best_checkpoint.pt"),
        ("learning_curve", "learning_curve.csv"),
        ("validation_predictions", "validation_predictions.csv"),
    ):
        summary = artifacts.get(key)
        if not isinstance(summary, dict):
            raise LauncherError(f"runner artifact summary is missing for {key}: {manifest_path}")
        artifact_path = run_dir / str(summary.get("path", fallback_name))
        if not artifact_path.is_file() or summary.get("sha256") != sha256_file(artifact_path):
            raise LauncherError(f"runner artifact checksum does not match for {key}: {artifact_path}")

    selection = manifest.get("selection")
    if not isinstance(selection, dict) or complete.get("selected_epoch") != selection.get("best_epoch_zero_based"):
        raise LauncherError(f"selected epoch does not match the completed run manifest: {run_dir}")

    timing = manifest.get("timing")
    if not isinstance(timing, dict):
        raise LauncherError(f"runner timing is missing: {manifest_path}")
    for key in ("preprocessing_seconds", "training_seconds", "total_elapsed_seconds"):
        value = timing.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise LauncherError(f"invalid timing.{key} in {manifest_path}")
    curve = _read_learning_curve(run_dir / "learning_curve.csv")
    return {
        "config": config,
        "config_sha256": digest,
        "checkpoint_sha256": checkpoint_digest,
        "timing": timing,
        "learning_curve": curve,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.replace(path)


def _runtime_probe(python_exe: Path) -> dict[str, Any]:
    code = (
        "import json,platform,sys,torch; "
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    try:
        result = subprocess.run(
            [str(python_exe), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LauncherError(f"cannot inspect CUDA runtime with {python_exe}: {exc}") from exc
    if result.returncode != 0:
        raise LauncherError(f"CUDA runtime check failed:\n{result.stdout}\n{result.stderr}")
    try:
        runtime = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise LauncherError(f"CUDA runtime check returned invalid output: {result.stdout!r}") from exc
    if not runtime.get("cuda_available"):
        raise LauncherError(f"CUDA is unavailable in {python_exe}: {runtime}")
    return runtime


def _nvidia_smi_path(requested: str | None) -> str:
    candidate = requested or shutil.which("nvidia-smi")
    if not candidate:
        raise LauncherError("nvidia-smi was not found; GPU resource sampling is required")
    result = subprocess.run(
        [candidate, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise LauncherError(f"nvidia-smi resource query failed: {result.stderr.strip()}")
    return str(Path(candidate).resolve()) if Path(candidate).exists() else candidate


def _runner_expected_fields(
    spec: RunSpec,
    dataset_path: Path,
    dataset_sha: str,
    cpu_threads: int,
    device: str,
) -> dict[str, Any]:
    expected = {
        "variant": spec.variant,
        "seed": spec.seed,
        "max_epochs": spec.max_epochs,
        "patience": spec.patience,
        "cpu_threads": cpu_threads,
        "device": device,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "train_distance_mm": [0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160],
        "val_distance_mm": [5, 40, 90, 130],
        "registered_test_distance_mm": [15, 25, 110, 150],
        "excluded_distance_mm": [70],
        "input_shape": [500, 4],
        "num_actions": 4,
        "num_angles": 64,
        "updates_per_epoch": 20,
        "batch_sizes_per_epoch": [44] * 19 + [43],
        "batch_loss_weight_multiplier": 20,
        "sampler_group_key": "distance_mm_and_original_direction_label",
        "sampler_rng": "numpy_seedsequence_seed_epoch",
        "sample_weight_rule": "position_voronoi_weight/(4*original_group_size)",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "angle_weight": ANGLE_WEIGHTS[spec.variant],
        "smoothness_weight": SMOOTHNESS_WEIGHTS[spec.variant],
        "angle_target_bandwidth": 8.0,
        "include_rms": spec.variant != "no_rms",
        "sampler": "original_distance_label_groups_no_replacement_19x44_plus_43",
        "label_shuffle_seed": spec.seed + 10_000 if spec.variant == "shuffle" else None,
    }
    return expected


def _code_hashes(runner_path: Path) -> dict[str, str]:
    files = {"runner": runner_path, "model": MODEL_PATH, **CODE_FILES}
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise LauncherError(f"required runner source files are missing: {missing}")
    return {name: sha256_file(path) for name, path in files.items()}


def _make_command(
    python_exe: Path,
    runner_path: Path,
    run_dir: Path,
    dataset_path: Path,
    spec: RunSpec,
    cpu_threads: int,
    device: str,
) -> list[str]:
    return [
        str(python_exe),
        str(runner_path),
        "--output-dir",
        str(run_dir),
        "--variant",
        spec.variant,
        "--seed",
        str(spec.seed),
        "--max-epochs",
        str(spec.max_epochs),
        "--patience",
        str(spec.patience),
        "--cpu-threads",
        str(cpu_threads),
        "--device",
        device,
        "--dataset",
        str(dataset_path),
        "--confirm-physical-order",
    ]


def _launcher_config(args: argparse.Namespace, specs: Sequence[RunSpec]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    python_exe = Path(args.python_executable).expanduser().resolve()
    runner_path = Path(args.runner).expanduser().resolve()
    runtime = _runtime_probe(python_exe)
    if tuple(int(part) for part in runtime["python"].split(".")[:2]) != (3, 10):
        raise LauncherError(f"the experiment requires Python 3.10, got {runtime['python']}")
    if not str(runtime["torch"]).startswith("1.13.1") or runtime["cuda_build"] != "11.7":
        raise LauncherError(
            "the approved runtime is PyTorch 1.13.1+cu117; "
            f"found torch={runtime['torch']} cuda_build={runtime['cuda_build']}"
        )
    if not dataset_path.is_file():
        raise LauncherError(f"dataset does not exist: {dataset_path}")
    if not runner_path.is_file():
        raise LauncherError(f"runner does not exist: {runner_path}")
    if not python_exe.is_file():
        raise LauncherError(f"Python executable does not exist: {python_exe}")
    dataset_sha = sha256_file(dataset_path)
    code_hashes = _code_hashes(runner_path)
    nvidia_smi = _nvidia_smi_path(args.nvidia_smi)
    return {
        "suite": args.suite,
        "output_dir": str(output_dir),
        "concurrency": args.concurrency,
        "device": "cuda",
        "cpu_threads": args.cpu_threads,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "python_executable": str(python_exe),
        "runtime": runtime,
        "runner_path": str(runner_path),
        "code_sha256": code_hashes,
        "nvidia_smi": nvidia_smi,
        "nvidia_smi_query": GPU_QUERY,
        "resource_sample_seconds": RESOURCE_SAMPLE_SECONDS,
        "run_specs": [asdict(spec) | {"job_id": spec.job_id} for spec in specs],
    }


def _new_job_record(spec: RunSpec, output_dir: Path) -> dict[str, Any]:
    return {
        "variant": spec.variant,
        "seed": spec.seed,
        "max_epochs": spec.max_epochs,
        "patience": spec.patience,
        "run_dir": str(output_dir / spec.job_id),
        "stdout_log": str(output_dir / f"{spec.job_id}.stdout.log"),
        "status": "pending",
    }


def _prepare_output_root(
    output_dir: Path,
    config: Mapping[str, Any],
    specs: Sequence[RunSpec],
    resume: bool,
) -> dict[str, Any]:
    manifest_path = output_dir / MANIFEST_NAME
    config_digest = config_sha256(config)
    if resume:
        if not output_dir.is_dir() or not manifest_path.is_file():
            raise LauncherError("--resume requires an existing output root with launcher_manifest.json")
        manifest = _read_json(manifest_path)
        if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
            raise LauncherError(f"unsupported launcher manifest in {manifest_path}")
        if manifest.get("config_sha256") != config_digest or manifest.get("config") != config:
            raise LauncherError("resume configuration differs from the existing experiment root")
        if manifest.get("status") not in {"running", "failed", "interrupted"}:
            if manifest.get("status") != "completed":
                raise LauncherError(f"invalid launcher state: {manifest.get('status')}")
        jobs = manifest.get("jobs")
        if not isinstance(jobs, dict):
            raise LauncherError("launcher manifest has no jobs object")
        expected_ids = {spec.job_id for spec in specs}
        if set(jobs) != expected_ids:
            raise LauncherError("launcher manifest run matrix does not match the requested suite")
        return manifest

    if output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries:
            raise LauncherError(f"output directory is not empty; use --resume only for a prior launcher run: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    jobs = {spec.job_id: _new_job_record(spec, output_dir) for spec in specs}
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "status": "running",
        "config": dict(config),
        "config_sha256": config_digest,
        "started_utc": _utc_now(),
        "jobs": jobs,
        "resource_samples_file": RESOURCE_SAMPLES_NAME,
    }
    resource_path = output_dir / RESOURCE_SAMPLES_NAME
    with resource_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sampled_at_utc",
                "elapsed_seconds",
                "suite",
                "active_jobs",
                "active_pids",
                "gpu_timestamp",
                "gpu_index",
                "gpu_uuid",
                "gpu_utilization_percent",
                "memory_used_mib",
                "memory_total_mib",
                "sample_error",
            ),
        )
        writer.writeheader()
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _preflight_existing_jobs(
    manifest: dict[str, Any],
    specs: Sequence[RunSpec],
    output_dir: Path,
    dataset_path: Path,
    dataset_sha: str,
    cpu_threads: int,
    device: str,
    expected_code_sha256: Mapping[str, str],
) -> list[RunSpec]:
    pending: list[RunSpec] = []
    jobs: dict[str, Any] = manifest["jobs"]
    for spec in specs:
        record = jobs[spec.job_id]
        run_dir = output_dir / spec.job_id
        log_path = output_dir / f"{spec.job_id}.stdout.log"
        if run_dir.exists():
            expected = _runner_expected_fields(spec, dataset_path, dataset_sha, cpu_threads, device)
            prior_digest = record.get("runner_config_sha256")
            completed = validate_completed_run(
                run_dir,
                expected,
                prior_digest,
                expected_code_sha256,
            )
            record.update(
                {
                    "status": "skipped_completed",
                    "returncode": 0,
                    "runner_config_sha256": completed["config_sha256"],
                    "checkpoint_sha256": completed["checkpoint_sha256"],
                    "timing": completed["timing"],
                    "learning_curve": completed["learning_curve"],
                    "completed_utc": record.get("completed_utc") or _utc_now(),
                }
            )
            continue
        if log_path.exists():
            raise LauncherError(f"run log exists but its run directory is missing; refusing to overwrite: {log_path}")
        if record.get("status") not in {"pending", "running", "failed", "cancelled"}:
            raise LauncherError(f"launcher run state has no output directory: {spec.job_id} ({record.get('status')})")
        record["status"] = "pending"
        pending.append(spec)
    return pending


def _append_gpu_sample(
    path: Path,
    nvidia_smi: str,
    suite: str,
    elapsed: float,
    active: Mapping[str, subprocess.Popen[Any]],
) -> None:
    command = [nvidia_smi, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        sample_stdout = result.stdout
        sample_error = result.stderr.strip() if result.returncode else ""
        sample_code = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        sample_stdout = ""
        sample_error = str(exc)
        sample_code = -1
    active_jobs = ";".join(active)
    active_pids = ";".join(str(proc.pid) for proc in active.values())
    rows: list[dict[str, Any]] = []
    if sample_code == 0:
        for raw in csv.reader(sample_stdout.splitlines()):
            if len(raw) != 6:
                continue
            rows.append(
                {
                    "gpu_timestamp": raw[0].strip(),
                    "gpu_index": raw[1].strip(),
                    "gpu_uuid": raw[2].strip(),
                    "gpu_utilization_percent": raw[3].strip(),
                    "memory_used_mib": raw[4].strip(),
                    "memory_total_mib": raw[5].strip(),
                    "sample_error": "",
                }
            )
    if not rows:
        rows = [
            {
                "gpu_timestamp": "",
                "gpu_index": "",
                "gpu_uuid": "",
                "gpu_utilization_percent": "",
                "memory_used_mib": "",
                "memory_total_mib": "",
                "sample_error": sample_error or f"nvidia-smi exit={sample_code}",
            }
        ]
    fieldnames = (
        "sampled_at_utc",
        "elapsed_seconds",
        "suite",
        "active_jobs",
        "active_pids",
        "gpu_timestamp",
        "gpu_index",
        "gpu_uuid",
        "gpu_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
        "sample_error",
    )
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(
                {
                    "sampled_at_utc": _utc_now(),
                    "elapsed_seconds": f"{elapsed:.3f}",
                    "suite": suite,
                    "active_jobs": active_jobs,
                    "active_pids": active_pids,
                    **row,
                }
            )
        stream.flush()
        os.fsync(stream.fileno())


def _aggregate(manifest: Mapping[str, Any], started_monotonic: float) -> dict[str, Any]:
    jobs = manifest["jobs"]
    completed = [record for record in jobs.values() if record.get("status") in {"completed", "skipped_completed"}]
    wall_seconds = max(0.0, time.monotonic() - started_monotonic)
    epoch_seconds: list[float] = []
    total_updates = 0
    preprocessing_seconds = 0.0
    runner_other_seconds = 0.0
    for record in completed:
        curve = record.get("learning_curve", [])
        total_updates += sum(int(row["optimizer_updates"]) for row in curve)
        epoch_values = [float(row["epoch_seconds"]) for row in curve]
        if len(epoch_values) > 1:
            epoch_seconds.extend(epoch_values[-3:])
        timing = record.get("timing", {})
        prep = float(timing.get("preprocessing_seconds", 0.0))
        training = float(timing.get("training_seconds", sum(epoch_values)))
        total = float(timing.get("total_elapsed_seconds", prep + training))
        preprocessing_seconds += prep
        runner_other_seconds += max(0.0, total - prep - training)
    return {
        "completed_jobs": len(completed),
        "total_jobs": len(jobs),
        "launcher_wall_seconds": wall_seconds,
        "total_optimizer_updates": total_updates,
        "updates_per_second_overall": total_updates / wall_seconds if wall_seconds > 0 else None,
        "seconds_per_update_overall": wall_seconds / total_updates if total_updates > 0 else None,
        "steady_epoch_seconds_last_three": epoch_seconds,
        "steady_epoch_seconds_mean": sum(epoch_seconds) / len(epoch_seconds) if epoch_seconds else None,
        "runner_preprocessing_seconds_sum": preprocessing_seconds,
        "runner_other_setup_and_finalize_seconds_sum": runner_other_seconds,
        "runner_training_seconds_sum": sum(
            float(record.get("timing", {}).get("training_seconds", 0.0)) for record in completed
        ),
    }


def preserve_completed_aggregate(
    manifest: dict[str, Any], verification_seconds: float
) -> None:
    """Record a resume check without replacing the original experiment wall time."""
    aggregate = manifest.get("aggregate")
    if not isinstance(aggregate, dict):
        raise LauncherError("completed launcher manifest has no prior aggregate to preserve")
    wall_seconds = aggregate.get("launcher_wall_seconds")
    valid_wall_seconds = (
        isinstance(wall_seconds, (int, float))
        and math.isfinite(float(wall_seconds))
        and wall_seconds > 0
    )
    recovered_unknown_wall_time = (
        wall_seconds is None and aggregate.get("wall_time_unavailable_on_recovery") is True
    )
    if not valid_wall_seconds and not recovered_unknown_wall_time:
        raise LauncherError("completed launcher manifest has no valid original wall time")
    manifest["aggregate"] = {
        **aggregate,
        "resume_verification_seconds": max(0.0, float(verification_seconds)),
    }


def _gpu_resource_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        return {"sample_count": 0, "read_error": str(exc)}
    valid: list[dict[str, Any]] = []
    for row in rows:
        try:
            utilization = float(row["gpu_utilization_percent"])
            memory_used = float(row["memory_used_mib"])
            memory_total = float(row["memory_total_mib"])
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(value) for value in (utilization, memory_used, memory_total)):
            valid.append(
                {
                    "gpu_index": row["gpu_index"],
                    "gpu_utilization_percent": utilization,
                    "memory_used_mib": memory_used,
                    "memory_total_mib": memory_total,
                }
            )
    if not valid:
        return {"sample_count": 0, "error_rows": len(rows)}
    utilizations = [row["gpu_utilization_percent"] for row in valid]
    memories = [row["memory_used_mib"] for row in valid]
    return {
        "sample_count": len(valid),
        "error_rows": len(rows) - len(valid),
        "gpu_indices": sorted({row["gpu_index"] for row in valid}),
        "mean_utilization_percent": sum(utilizations) / len(utilizations),
        "peak_utilization_percent": max(utilizations),
        "peak_memory_used_mib": max(memories),
        "reported_memory_total_mib": max(row["memory_total_mib"] for row in valid),
    }


def _terminate_active(active: dict[str, dict[str, Any]], manifest: dict[str, Any], reason: str) -> None:
    for job_id, state in active.items():
        proc: subprocess.Popen[Any] = state["process"]
        if proc.poll() is None:
            proc.terminate()
    for job_id, state in active.items():
        proc = state["process"]
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_stream = state.get("log_stream")
        if log_stream is not None:
            log_stream.close()
        record = manifest["jobs"][job_id]
        record.update(
            {
                "status": "cancelled",
                "returncode": proc.returncode,
                "ended_utc": _utc_now(),
                "failure_reason": reason,
            }
        )
    active.clear()


def _run_scheduler(
    manifest: dict[str, Any],
    pending: Sequence[RunSpec],
    *,
    output_dir: Path,
    runner_path: Path,
    python_exe: Path,
    dataset_path: Path,
    dataset_sha: str,
    concurrency: int,
    cpu_threads: int,
    device: str,
    nvidia_smi: str,
    poll_seconds: float,
) -> None:
    manifest_path = output_dir / MANIFEST_NAME
    resource_path = output_dir / RESOURCE_SAMPLES_NAME
    active = _ACTIVE_CHILDREN
    active.clear()
    waiting = list(pending)
    started = time.monotonic()
    next_sample = started
    failure: str | None = None

    while waiting or active:
        while waiting and len(active) < concurrency and failure is None:
            spec = waiting.pop(0)
            run_dir = output_dir / spec.job_id
            log_path = output_dir / f"{spec.job_id}.stdout.log"
            expected = _runner_expected_fields(spec, dataset_path, dataset_sha, cpu_threads, device)
            command = _make_command(python_exe, runner_path, run_dir, dataset_path, spec, cpu_threads, device)
            record = manifest["jobs"][spec.job_id]
            record.update(
                {
                    "status": "starting",
                    "command": command,
                    "command_display": subprocess.list2cmdline(command),
                    "started_utc": _utc_now(),
                }
            )
            _write_json_atomic(manifest_path, manifest)
            log_stream = None
            try:
                log_stream = log_path.open("x", encoding="utf-8", newline="\n")
                proc = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, ValueError) as exc:
                if log_stream is not None:
                    log_stream.close()
                record.update(
                    {
                        "status": "failed",
                        "returncode": None,
                        "failure_reason": f"could not start runner: {exc}",
                        "ended_utc": _utc_now(),
                    }
                )
                failure = f"failed to start {spec.job_id}: {exc}"
                _write_json_atomic(manifest_path, manifest)
                print(failure, file=sys.stderr, flush=True)
                break
            active[spec.job_id] = {
                "process": proc,
                "log_stream": log_stream,
                "started_monotonic": time.monotonic(),
                "expected": expected,
            }
            record.update({"status": "running", "pid": proc.pid})
            _write_json_atomic(manifest_path, manifest)
            print(f"started {spec.job_id} pid={proc.pid}", flush=True)

        now = time.monotonic()
        for job_id, state in list(active.items()):
            proc = state["process"]
            returncode = proc.poll()
            if returncode is None:
                continue
            state["log_stream"].close()
            record = manifest["jobs"][job_id]
            process_wall_seconds = max(0.0, time.monotonic() - state["started_monotonic"])
            record.update(
                {
                    "returncode": returncode,
                    "process_wall_seconds": process_wall_seconds,
                    "ended_utc": _utc_now(),
                }
            )
            if returncode != 0:
                record.update({"status": "failed", "failure_reason": f"runner exited with {returncode}"})
                failure = f"{job_id} exited with code {returncode}; see {record['stdout_log']}"
            else:
                run_dir = output_dir / job_id
                try:
                    completed = validate_completed_run(
                        run_dir,
                        state["expected"],
                        expected_code_sha256=manifest["config"]["code_sha256"],
                    )
                    curve = completed["learning_curve"]
                    epoch_values = [float(row["epoch_seconds"]) for row in curve]
                    steady = epoch_values[-3:] if len(epoch_values) > 1 else []
                    record.update(
                        {
                            "status": "completed",
                            "runner_config_sha256": completed["config_sha256"],
                            "checkpoint_sha256": completed["checkpoint_sha256"],
                            "timing": completed["timing"],
                            "learning_curve": curve,
                            "total_optimizer_updates": sum(int(row["optimizer_updates"]) for row in curve),
                            "steady_epoch_seconds_last_three": steady,
                        }
                    )
                    print(
                        f"completed {job_id} wall={process_wall_seconds:.1f}s "
                        f"epochs={len(curve)} updates={record['total_optimizer_updates']}",
                        flush=True,
                    )
                except LauncherError as exc:
                    record.update({"status": "failed", "failure_reason": str(exc)})
                    failure = f"{job_id} exited 0 but did not produce a valid completed run: {exc}"
            del active[job_id]
            _write_json_atomic(manifest_path, manifest)
            if failure:
                print(failure, file=sys.stderr, flush=True)
                break

        if failure is not None:
            _terminate_active(active, manifest, failure)
            waiting.clear()
            _write_json_atomic(manifest_path, manifest)
            break

        now = time.monotonic()
        if now >= next_sample:
            _append_gpu_sample(
                resource_path,
                nvidia_smi,
                manifest["config"]["suite"],
                now - started,
                {job_id: state["process"] for job_id, state in active.items()},
            )
            next_sample = now + RESOURCE_SAMPLE_SECONDS
        if waiting or active:
            time.sleep(poll_seconds)

    manifest["aggregate"] = _aggregate(manifest, started)
    manifest["aggregate"]["gpu_resources"] = _gpu_resource_summary(resource_path)
    manifest["ended_utc"] = _utc_now()
    if failure is None and all(
        manifest["jobs"][spec.job_id].get("status") in {"completed", "skipped_completed"}
        for spec in build_run_specs(manifest["config"]["suite"])
    ):
        manifest["status"] = "completed"
    else:
        manifest["status"] = "failed"
        manifest["failure_reason"] = failure or "one or more runs were not completed"
    _write_json_atomic(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise LauncherError(manifest["failure_reason"])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the approved C170 CUDA probe or formal matrix. Probe results are throughput-only; "
            "this launcher never computes final-test metrics."
        )
    )
    parser.add_argument("--suite", choices=("probe", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new experiment root; use --resume to reuse it")
    parser.add_argument("--concurrency", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--runner", type=Path, default=RUNNER_PATH)
    parser.add_argument("--nvidia-smi", default=None)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="skip verified completed runs in this exact configuration")
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = build_run_specs(args.suite)
    config = _launcher_config(args, specs)
    output_dir = Path(config["output_dir"])
    manifest = _prepare_output_root(output_dir, config, specs, args.resume)
    resume_started = time.monotonic()
    prior_aggregate = manifest.get("aggregate")
    dataset_path = Path(config["dataset_path"])
    pending = _preflight_existing_jobs(
        manifest,
        specs,
        output_dir,
        dataset_path,
        config["dataset_sha256"],
        config["cpu_threads"],
        config["device"],
        config["code_sha256"],
    )
    _write_json_atomic(output_dir / MANIFEST_NAME, manifest)
    if not pending:
        if isinstance(prior_aggregate, dict):
            manifest["aggregate"] = prior_aggregate
            preserve_completed_aggregate(manifest, time.monotonic() - resume_started)
        else:
            aggregate = _aggregate(manifest, time.monotonic())
            aggregate.update(
                {
                    "launcher_wall_seconds": None,
                    "updates_per_second_overall": None,
                    "seconds_per_update_overall": None,
                    "wall_time_unavailable_on_recovery": True,
                    "resume_verification_seconds": max(0.0, time.monotonic() - resume_started),
                }
            )
            manifest["aggregate"] = aggregate
        manifest["aggregate"]["gpu_resources"] = _gpu_resource_summary(output_dir / RESOURCE_SAMPLES_NAME)
        manifest["status"] = "completed"
        manifest["ended_utc"] = _utc_now()
        _write_json_atomic(output_dir / MANIFEST_NAME, manifest)
        print(f"all {len(specs)} runs were already complete and verified in {output_dir}", flush=True)
        return 0

    print(
        f"suite={args.suite} jobs={len(specs)} pending={len(pending)} "
        f"concurrency={args.concurrency} gpu={config['runtime']['gpu_name']} output={output_dir}",
        flush=True,
    )
    try:
        _run_scheduler(
            manifest,
            pending,
            output_dir=output_dir,
            runner_path=Path(config["runner_path"]),
            python_exe=Path(config["python_executable"]),
            dataset_path=dataset_path,
            dataset_sha=config["dataset_sha256"],
            concurrency=args.concurrency,
            cpu_threads=args.cpu_threads,
            device=config["device"],
            nvidia_smi=config["nvidia_smi"],
            poll_seconds=args.poll_seconds,
        )
    except KeyboardInterrupt:
        _terminate_active(_ACTIVE_CHILDREN, manifest, "launcher interrupted by user")
        for record in manifest["jobs"].values():
            if record.get("status") in {"running", "starting"}:
                record["status"] = "cancelled"
        manifest["status"] = "interrupted"
        manifest["ended_utc"] = _utc_now()
        manifest["aggregate"] = _aggregate(manifest, time.monotonic())
        manifest["aggregate"]["gpu_resources"] = _gpu_resource_summary(
            output_dir / RESOURCE_SAMPLES_NAME
        )
        _write_json_atomic(output_dir / MANIFEST_NAME, manifest)
        raise
    aggregate = manifest.get("aggregate", {})
    print(json.dumps({"status": manifest["status"], "aggregate": aggregate}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LauncherError as exc:
        print(f"launcher error: {exc}", file=sys.stderr)
        raise SystemExit(2)
