"""Schedule the frozen C170 ring-v2 CUDA experiment or a throughput-only probe."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
RUNNER_PATH = PROJECT_DIR / "run_c170_ring_cuda_experiment_v2.py"
DEFAULT_DATASET = PROJECT_DIR / "data" / "processed_c170_20260926" / "dataset.npz"
DEFAULT_PYTHON = Path(r"C:\Users\56569\AppData\Local\Programs\Python\Python310\python.exe")
FORMAL_SEEDS = (17, 29, 43, 61, 89)
# The frozen runner accepts these five seed values. Repeats provide eight equal-cost
# independent processes for load measurement without changing that runner contract.
PROBE_SEEDS = (17, 29, 43, 61, 89, 17, 29, 43)
FORMAL_VARIANTS = ("ring_main", "ring_ce", "slot_supcon", "ring_shuffle")
VARIANT_CONFIG = {
    "ring_main": {"model_variant": "main", "use_supcon": True, "shuffle_labels": False},
    "ring_ce": {
        "model_variant": "classification_only",
        "use_supcon": False,
        "shuffle_labels": False,
    },
    "slot_supcon": {"model_variant": "slot_readout", "use_supcon": True, "shuffle_labels": False},
    "ring_shuffle": {"model_variant": "main", "use_supcon": True, "shuffle_labels": True},
}
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_COMPLETE_NAME = "RUN_COMPLETE.json"
LAUNCHER_MANIFEST_NAME = "launcher_manifest.json"
RESOURCE_SAMPLES_NAME = "gpu_samples.csv"
LAUNCHER_FORMAT = "c170-ring-cuda-v2-launcher"
LAUNCHER_FORMAT_VERSION = 1
RUN_FORMAT = "c170-ring-cuda-v2-run"
RUN_FORMAT_VERSION = 1
COMPLETE_FORMAT = "c170-ring-cuda-v2-complete"
COMPLETE_FORMAT_VERSION = 1
UPDATES_PER_EPOCH = 20
RESOURCE_SAMPLE_SECONDS = 2.0
GPU_QUERY = "timestamp,index,uuid,utilization.gpu,memory.used,memory.total"
SOURCE_PATHS = {
    "ring_representation": PROJECT_DIR / "src" / "c170_ring_representation.py",
    "dataset_preprocessing_helpers": PROJECT_DIR / "run_c170_experiment.py",
    "preprocessing_transitive_signal_processing": PROJECT_DIR / "src" / "rotation_poc.py",
}
RUNNER_SOURCE_KEYS = tuple(SOURCE_PATHS) + ("runner",)


class LauncherError(RuntimeError):
    """An invalid run configuration, launch, or completed artifact."""


@dataclass(frozen=True)
class RunSpec:
    variant: str
    seed: int
    max_epochs: int
    patience: int
    repeat_index: int = 1

    @property
    def job_id(self) -> str:
        base = f"{self.variant}__seed-{self.seed}"
        return base if self.repeat_index == 1 else f"{base}__repeat-{self.repeat_index}"


def build_run_specs(suite: str) -> list[RunSpec]:
    if suite == "formal":
        return [
            RunSpec(variant, seed, max_epochs=160, patience=25)
            for variant in FORMAL_VARIANTS
            for seed in FORMAL_SEEDS
        ]
    if suite == "probe":
        seen: dict[int, int] = {}
        specs: list[RunSpec] = []
        for seed in PROBE_SEEDS:
            seen[seed] = seen.get(seed, 0) + 1
            specs.append(RunSpec("ring_main", seed, max_epochs=3, patience=3, repeat_index=seen[seed]))
        return specs
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
    for expected_epoch, row in enumerate(rows, start=1):
        try:
            epoch = int(row["epoch"])
            updates = int(row["optimizer_updates"])
            seconds = float(row["epoch_seconds"])
        except (TypeError, ValueError) as exc:
            raise LauncherError(f"invalid epoch timing row in {path}: {row}") from exc
        if epoch != expected_epoch or updates != UPDATES_PER_EPOCH:
            raise LauncherError(f"invalid epoch/update sequence in {path}: {row}")
        if not math.isfinite(seconds) or seconds < 0:
            raise LauncherError(f"invalid epoch timing values in {path}: {row}")
    return rows


def _runner_expected_fields(
    spec: RunSpec,
    dataset_path: Path,
    dataset_sha: str,
    cpu_threads: int,
) -> dict[str, Any]:
    variant = VARIANT_CONFIG[spec.variant]
    expected = {
        "variant": spec.variant,
        "model_variant": variant["model_variant"],
        "use_supcon": variant["use_supcon"],
        "shuffle_labels": variant["shuffle_labels"],
        "seed": spec.seed,
        "max_epochs": spec.max_epochs,
        "patience": spec.patience,
        "cpu_threads": cpu_threads,
        "device": "cuda:0",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha,
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
        "supcon_temperature": 0.1,
        "supcon_weight": 0.1 if variant["use_supcon"] else 0.0,
        "supcon_batch_reduction": "existing_helper_scalar_mean_over_anchors",
        "checkpoint_selection": "highest_validation_macro_f1_then_lowest_validation_cross_entropy",
        "test_policy": "registered_only_no_preprocessing_no_predictions_no_metrics",
        "physical_order_confirmed_by_operator": True,
    }
    return expected


def _make_command(
    python_exe: Path,
    runner_path: Path,
    run_dir: Path,
    dataset_path: Path,
    spec: RunSpec,
    cpu_threads: int,
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
        "cuda",
        "--dataset",
        str(dataset_path),
        "--confirm-physical-order",
    ]


def validate_completed_run(
    run_dir: Path,
    expected: Mapping[str, Any],
    expected_code_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the runner marker, canonical config, provenance, and all three artifacts."""
    manifest_path = run_dir / RUN_MANIFEST_NAME
    complete_path = run_dir / RUN_COMPLETE_NAME
    manifest = _read_json(manifest_path)
    complete = _read_json(complete_path)

    if manifest.get("format") != RUN_FORMAT or manifest.get("format_version") != RUN_FORMAT_VERSION:
        raise LauncherError(f"unexpected runner manifest format in {manifest_path}")
    if manifest.get("status") != "completed" or manifest.get("test_metrics_computed") is not False:
        raise LauncherError(f"invalid runner completion state in {manifest_path}")
    test_access = manifest.get("test_access")
    if not isinstance(test_access, dict) or test_access.get("predicted_or_scored") is not False:
        raise LauncherError(f"runner manifest does not confirm test data remained unscored: {run_dir}")

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

    split = manifest.get("split")
    if not isinstance(split, dict):
        raise LauncherError(f"runner split metadata is missing: {manifest_path}")
    if split.get("split_counts") != {"train": 879, "val": 320, "test": 320}:
        raise LauncherError(f"runner split counts do not match the frozen protocol: {run_dir}")
    shuffled = bool(expected.get("shuffle_labels"))
    expected_shuffle_seed = int(expected["seed"]) + 10_000 if shuffled else None
    if split.get("label_shuffle_seed") != expected_shuffle_seed:
        raise LauncherError(f"runner split label-shuffle seed does not match the requested branch: {run_dir}")
    mapping = split.get("label_shuffle_mapping")
    if not isinstance(mapping, list):
        raise LauncherError(f"runner split label-shuffle mapping is malformed: {run_dir}")
    if shuffled and not mapping:
        raise LauncherError(f"shuffled run has no recorded block mapping: {run_dir}")
    if not shuffled and mapping:
        raise LauncherError(f"unshuffled run unexpectedly records a label mapping: {run_dir}")

    provenance = manifest.get("provenance")
    actual_code_sha = provenance.get("code_sha256") if isinstance(provenance, dict) else None
    if not isinstance(actual_code_sha, dict) or actual_code_sha != dict(expected_code_sha256):
        raise LauncherError(f"runner provenance hashes do not match the current sources: {run_dir}")

    if complete.get("format") != COMPLETE_FORMAT or complete.get("format_version") != COMPLETE_FORMAT_VERSION:
        raise LauncherError(f"unexpected runner completion marker format in {complete_path}")
    if complete.get("status") != "completed" or complete.get("test_metrics_computed") is not False:
        raise LauncherError(f"invalid runner completion marker state in {complete_path}")
    if complete.get("config_sha256") != digest:
        raise LauncherError(f"completion marker config hash does not match: {run_dir}")
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise LauncherError(f"run manifest checksum does not match completion marker: {run_dir}")

    artifact_paths = {
        "best_checkpoint": run_dir / "best_checkpoint.pt",
        "learning_curve": run_dir / "learning_curve.csv",
        "validation_predictions": run_dir / "validation_predictions.csv",
    }
    marker_hashes = complete.get("artifact_sha256")
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(marker_hashes, dict) or not isinstance(manifest_artifacts, dict):
        raise LauncherError(f"runner artifact summaries are missing: {run_dir}")
    artifact_hashes: dict[str, str] = {}
    for name, path in artifact_paths.items():
        summary = manifest_artifacts.get(name)
        if not isinstance(summary, dict) or summary.get("path") != path.name:
            raise LauncherError(f"invalid artifact summary for {name}: {run_dir}")
        if not path.is_file() or path.stat().st_size == 0:
            raise LauncherError(f"required run artifact is missing or empty: {path}")
        digest_value = sha256_file(path)
        if summary.get("sha256") != digest_value or marker_hashes.get(name) != digest_value:
            raise LauncherError(f"artifact checksum does not match for {name}: {path}")
        artifact_hashes[name] = digest_value

    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise LauncherError(f"runner checkpoint selection is missing: {manifest_path}")
    selected_epoch = selection.get("best_epoch_one_based")
    if not isinstance(selected_epoch, int) or selected_epoch < 1:
        raise LauncherError(f"runner selected epoch is invalid: {run_dir}")
    if complete.get("selected_epoch_one_based") != selected_epoch:
        raise LauncherError(f"selected epoch does not match completion marker: {run_dir}")

    timing = manifest.get("timing")
    if not isinstance(timing, dict):
        raise LauncherError(f"runner timing is missing: {manifest_path}")
    for key in ("preprocessing_seconds", "training_seconds", "total_elapsed_seconds"):
        value = timing.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise LauncherError(f"invalid timing.{key} in {manifest_path}")
    curve = _read_learning_curve(artifact_paths["learning_curve"])
    if selected_epoch > len(curve):
        raise LauncherError(f"selected epoch is outside the learning curve: {run_dir}")
    return {
        "config": config,
        "config_sha256": digest,
        "artifact_sha256": artifact_hashes,
        "timing": timing,
        "learning_curve": curve,
        "selection": selection,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
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
    if tuple(int(part) for part in runtime["python"].split(".")[:2]) != (3, 10):
        raise LauncherError(f"the experiment requires Python 3.10, got {runtime['python']}")
    if not str(runtime["torch"]).startswith("1.13.1") or runtime["cuda_build"] != "11.7":
        raise LauncherError(
            "the approved runtime is PyTorch 1.13.1+cu117; "
            f"found torch={runtime['torch']} cuda_build={runtime['cuda_build']}"
        )
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


def _source_hashes(runner_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    paths = {"runner": runner_path, **SOURCE_PATHS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if not Path(__file__).is_file():
        missing.append(str(Path(__file__)))
    if missing:
        raise LauncherError(f"required source files are missing: {missing}")
    runner_hashes = {name: sha256_file(path) for name, path in paths.items()}
    launcher_hashes = {"launcher": sha256_file(Path(__file__).resolve()), **runner_hashes}
    return launcher_hashes, runner_hashes


def _launcher_config(args: argparse.Namespace, specs: Sequence[RunSpec]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    python_exe = Path(args.python_executable).expanduser().resolve()
    runner_path = Path(args.runner).expanduser().resolve()
    if args.cpu_threads != 2:
        raise LauncherError("the approved experiment fixes --cpu-threads to 2")
    if not dataset_path.is_file():
        raise LauncherError(f"dataset does not exist: {dataset_path}")
    if not runner_path.is_file():
        raise LauncherError(f"runner does not exist: {runner_path}")
    if not python_exe.is_file():
        raise LauncherError(f"Python executable does not exist: {python_exe}")
    runtime = _runtime_probe(python_exe)
    nvidia_smi = _nvidia_smi_path(args.nvidia_smi)
    dataset_sha = sha256_file(dataset_path)
    code_hashes, runner_code_hashes = _source_hashes(runner_path)
    return {
        "suite": args.suite,
        "output_dir": str(output_dir),
        "concurrency": args.concurrency,
        "device": "cuda",
        "cpu_threads": 2,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "python_executable": str(python_exe),
        "runner_path": str(runner_path),
        "nvidia_smi": nvidia_smi,
        "code_sha256": code_hashes,
        "runner_code_sha256": runner_code_hashes,
        "runtime": runtime,
        "run_matrix": [
            {
                "job_id": spec.job_id,
                "variant": spec.variant,
                "seed": spec.seed,
                "max_epochs": spec.max_epochs,
                "patience": spec.patience,
                "repeat_index": spec.repeat_index,
            }
            for spec in specs
        ],
    }


def _new_job_record(spec: RunSpec, output_dir: Path) -> dict[str, Any]:
    return {
        "variant": spec.variant,
        "seed": spec.seed,
        "repeat_index": spec.repeat_index,
        "max_epochs": spec.max_epochs,
        "patience": spec.patience,
        "status": "pending",
        "run_dir": str(output_dir / spec.job_id),
        "stdout_log": str(output_dir / f"{spec.job_id}.stdout.log"),
        "latest_epoch": 0,
        "last_progress_utc": None,
        "pid": None,
    }


def _prepare_output_root(output_dir: Path, config: Mapping[str, Any], specs: Sequence[RunSpec]) -> dict[str, Any]:
    if output_dir.exists():
        entries = list(output_dir.iterdir())
        has_only_prepared_snapshot = (
            config.get("suite") == "formal"
            and len(entries) == 1
            and entries[0].name == "source_snapshot"
            and entries[0].is_dir()
            and (entries[0] / "snapshot_manifest.json").is_file()
            and (entries[0] / "environment.json").is_file()
        )
        if entries and not has_only_prepared_snapshot:
            raise LauncherError(
                f"output directory is not empty except for the approved source_snapshot: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        run_dir = output_dir / spec.job_id
        log_path = output_dir / f"{spec.job_id}.stdout.log"
        if run_dir.exists() or log_path.exists():
            raise LauncherError(f"task output already exists; refusing to overwrite: {spec.job_id}")
    resource_path = output_dir / RESOURCE_SAMPLES_NAME
    with resource_path.open("x", newline="", encoding="utf-8") as stream:
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
    jobs = {spec.job_id: _new_job_record(spec, output_dir) for spec in specs}
    manifest = {
        "format": LAUNCHER_FORMAT,
        "format_version": LAUNCHER_FORMAT_VERSION,
        "status": "running",
        "config": dict(config),
        "config_sha256": config_sha256(config),
        "started_utc": _utc_now(),
        "jobs": jobs,
        "resource_samples_file": RESOURCE_SAMPLES_NAME,
        "test_metrics_computed": False,
    }
    _write_json_atomic(output_dir / LAUNCHER_MANIFEST_NAME, manifest)
    return manifest


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
    fields = (
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
        writer = csv.DictWriter(stream, fieldnames=fields)
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


def _resource_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        return {"sample_count": 0, "read_error": str(exc)}
    valid: list[dict[str, float]] = []
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
                    "utilization": utilization,
                    "memory_used": memory_used,
                    "memory_total": memory_total,
                }
            )
    if not valid:
        return {"sample_count": 0, "error_rows": len(rows)}
    return {
        "sample_count": len(valid),
        "error_rows": len(rows) - len(valid),
        "mean_utilization_percent": sum(row["utilization"] for row in valid) / len(valid),
        "peak_utilization_percent": max(row["utilization"] for row in valid),
        "peak_memory_used_mib": max(row["memory_used"] for row in valid),
        "reported_memory_total_mib": max(row["memory_total"] for row in valid),
    }


def _aggregate(manifest: Mapping[str, Any], wall_seconds: float) -> dict[str, Any]:
    jobs = manifest["jobs"]
    completed = [record for record in jobs.values() if record.get("status") == "completed"]
    all_updates = 0
    steady_rows: list[dict[str, float]] = []
    preprocessing_seconds = 0.0
    runner_training_seconds = 0.0
    steady_window_epochs = 2 if manifest["config"]["suite"] == "probe" else 3
    for record in completed:
        curve = record.get("learning_curve", [])
        all_updates += sum(int(row["optimizer_updates"]) for row in curve)
        # Exclude epoch one, which includes CUDA initialization, from probe steady timing.
        for row in curve[-steady_window_epochs:] if len(curve) > 1 else []:
            steady_rows.append(
                {"seconds": float(row["epoch_seconds"]), "updates": float(row["optimizer_updates"])}
            )
        timing = record.get("timing", {})
        preprocessing_seconds += float(timing.get("preprocessing_seconds", 0.0))
        runner_training_seconds += float(timing.get("training_seconds", 0.0))
    steady_seconds = [row["seconds"] for row in steady_rows]
    mean_epoch_seconds = sum(steady_seconds) / len(steady_seconds) if steady_seconds else None
    steady_updates = sum(row["updates"] for row in steady_rows)
    steady_time_sum = sum(steady_seconds)
    concurrency = int(manifest["config"]["concurrency"])
    return {
        "completed_jobs": len(completed),
        "total_jobs": len(jobs),
        "launcher_wall_seconds": max(0.0, wall_seconds),
        "total_optimizer_updates": all_updates,
        "updates_per_second_overall": all_updates / wall_seconds if wall_seconds > 0 else None,
        "steady_epoch_seconds_in_window": steady_seconds,
        "steady_window_epochs_per_run": steady_window_epochs,
        "steady_epoch_seconds_mean": mean_epoch_seconds,
        "steady_single_worker_updates_per_second": (
            UPDATES_PER_EPOCH / mean_epoch_seconds if mean_epoch_seconds else None
        ),
        "steady_aggregate_updates_per_second_estimate": (
            concurrency * UPDATES_PER_EPOCH / mean_epoch_seconds if mean_epoch_seconds else None
        ),
        "steady_updates_per_epoch_second_pooled": (
            steady_updates / steady_time_sum if steady_time_sum > 0 else None
        ),
        "runner_preprocessing_seconds_sum": preprocessing_seconds,
        "runner_training_seconds_sum": runner_training_seconds,
    }


def _refresh_progress(job_id: str, record: dict[str, Any]) -> bool:
    # The runner atomically replaces learning_curve.csv every epoch. Reading that file
    # while it is being replaced can prevent os.replace on Windows, so progress comes
    # from the append-only worker log and only the epoch/update tokens are inspected.
    log_path = Path(record["stdout_log"])
    if not log_path.is_file():
        return False
    try:
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 32_768), os.SEEK_SET)
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return False
    latest_epoch = 0
    latest_updates = 0
    for line in reversed(lines):
        if not line.startswith("variant="):
            continue
        fields: dict[str, str] = {}
        for token in line.split():
            if token.startswith("epoch="):
                fields["epoch"] = token[6:]
            elif token.startswith("updates="):
                fields["updates"] = token[8:]
        try:
            latest_epoch = int(fields["epoch"])
            latest_updates = int(fields["updates"])
        except (KeyError, ValueError):
            continue
        if latest_epoch > 0 and latest_updates > 0:
            break
        latest_epoch = 0
        latest_updates = 0
    if latest_epoch == 0:
        return False
    if latest_epoch <= int(record.get("latest_epoch", 0)):
        return False
    record.update(
        {
            "latest_epoch": latest_epoch,
            "latest_epoch_updates": latest_updates,
            "last_progress_utc": _utc_now(),
        }
    )
    return True


def _hidden_process_options() -> dict[str, Any]:
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {"startupinfo": startupinfo, "creationflags": subprocess.CREATE_NO_WINDOW}


def _terminate_active(active: dict[str, dict[str, Any]], manifest: dict[str, Any], reason: str) -> None:
    for state in active.values():
        process: subprocess.Popen[Any] = state["process"]
        if process.poll() is None:
            process.terminate()
    for job_id, state in active.items():
        process = state["process"]
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        state["log_stream"].close()
        manifest["jobs"][job_id].update(
            {
                "status": "cancelled",
                "returncode": process.returncode,
                "ended_utc": _utc_now(),
                "failure_reason": reason,
            }
        )
    active.clear()


def _run_scheduler(
    manifest: dict[str, Any],
    specs: Sequence[RunSpec],
    *,
    output_dir: Path,
    runner_path: Path,
    python_exe: Path,
    dataset_path: Path,
    concurrency: int,
    nvidia_smi: str,
    poll_seconds: float,
) -> None:
    manifest_path = output_dir / LAUNCHER_MANIFEST_NAME
    resource_path = output_dir / RESOURCE_SAMPLES_NAME
    waiting = list(specs)
    active: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    next_sample = started
    next_manifest_update = started
    failure: str | None = None
    runner_code_sha = manifest["config"]["runner_code_sha256"]

    while waiting or active:
        while waiting and len(active) < concurrency and failure is None:
            spec = waiting.pop(0)
            job_id = spec.job_id
            record = manifest["jobs"][job_id]
            run_dir = Path(record["run_dir"])
            log_path = Path(record["stdout_log"])
            if run_dir.exists() or log_path.exists():
                failure = f"refusing to overwrite existing task output: {job_id}"
                record.update({"status": "failed", "failure_reason": failure})
                break
            command = _make_command(
                python_exe,
                runner_path,
                run_dir,
                dataset_path,
                spec,
                cpu_threads=2,
            )
            record.update(
                {
                    "status": "starting",
                    "command": command,
                    "command_display": subprocess.list2cmdline(command),
                    "started_utc": _utc_now(),
                }
            )
            _write_json_atomic(manifest_path, manifest)
            try:
                log_stream = log_path.open("x", encoding="utf-8", newline="\n")
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_DIR,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    **_hidden_process_options(),
                )
            except (OSError, ValueError) as exc:
                if "log_stream" in locals() and not log_stream.closed:
                    log_stream.close()
                failure = f"could not start {job_id}: {exc}"
                record.update(
                    {"status": "failed", "returncode": None, "failure_reason": failure, "ended_utc": _utc_now()}
                )
                _write_json_atomic(manifest_path, manifest)
                break
            active[job_id] = {
                "process": process,
                "log_stream": log_stream,
                "started_monotonic": time.monotonic(),
                "expected": _runner_expected_fields(
                    spec,
                    dataset_path,
                    manifest["config"]["dataset_sha256"],
                    cpu_threads=2,
                ),
            }
            record.update({"status": "running", "pid": process.pid, "last_progress_utc": _utc_now()})
            _write_json_atomic(manifest_path, manifest)
            print(f"started {job_id} pid={process.pid}", flush=True)

        now = time.monotonic()
        if failure is not None:
            _terminate_active(active, manifest, failure)
            waiting.clear()
            _write_json_atomic(manifest_path, manifest)
            break

        changed = False
        for job_id, state in list(active.items()):
            record = manifest["jobs"][job_id]
            changed = _refresh_progress(job_id, record) or changed
            process: subprocess.Popen[Any] = state["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            state["log_stream"].close()
            process_wall = max(0.0, time.monotonic() - state["started_monotonic"])
            record.update({"returncode": returncode, "process_wall_seconds": process_wall, "ended_utc": _utc_now()})
            if returncode != 0:
                failure = f"{job_id} exited with code {returncode}; see {record['stdout_log']}"
                record.update({"status": "failed", "failure_reason": failure})
            else:
                try:
                    completed = validate_completed_run(
                        Path(record["run_dir"]),
                        state["expected"],
                        expected_code_sha256=runner_code_sha,
                    )
                    curve = completed["learning_curve"]
                    record.update(
                        {
                            "status": "completed",
                            "runner_config_sha256": completed["config_sha256"],
                            "artifact_sha256": completed["artifact_sha256"],
                            "timing": completed["timing"],
                            "selection": completed["selection"],
                            "learning_curve": curve,
                            "total_optimizer_updates": sum(int(row["optimizer_updates"]) for row in curve),
                            "steady_epoch_seconds_in_window": [
                                float(row["epoch_seconds"])
                                for row in curve[-(2 if manifest["config"]["suite"] == "probe" else 3):]
                            ],
                        }
                    )
                    print(
                        f"completed {job_id} wall={process_wall:.1f}s "
                        f"epochs={len(curve)} updates={record['total_optimizer_updates']}",
                        flush=True,
                    )
                except LauncherError as exc:
                    failure = f"{job_id} exited 0 but failed artifact verification: {exc}"
                    record.update({"status": "failed", "failure_reason": failure})
            del active[job_id]
            changed = True
            _write_json_atomic(manifest_path, manifest)
            if failure is not None:
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
            changed = True
        if changed or now >= next_manifest_update:
            _write_json_atomic(manifest_path, manifest)
            next_manifest_update = now + 2.0
        if waiting or active:
            time.sleep(poll_seconds)

    wall_seconds = time.monotonic() - started
    manifest["aggregate"] = _aggregate(manifest, wall_seconds)
    manifest["aggregate"]["gpu_resources"] = _resource_summary(resource_path)
    manifest["ended_utc"] = _utc_now()
    if failure is None and all(manifest["jobs"][spec.job_id].get("status") == "completed" for spec in specs):
        manifest["status"] = "completed"
        print(json.dumps({"status": "completed", "aggregate": manifest["aggregate"]}, indent=2), flush=True)
    else:
        manifest["status"] = "failed"
        manifest["failure_reason"] = failure or "one or more runs did not complete"
    _write_json_atomic(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise LauncherError(manifest["failure_reason"])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schedule the fixed C170 ring-v2 CUDA probe or 20-run formal matrix."
    )
    parser.add_argument("--suite", choices=("probe", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new experiment root; existing data is never overwritten")
    parser.add_argument("--concurrency", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--python-executable", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--runner", type=Path, default=RUNNER_PATH)
    parser.add_argument("--nvidia-smi", default=None)
    parser.add_argument("--cpu-threads", type=int, choices=(2,), default=2)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = build_run_specs(args.suite)
    config = _launcher_config(args, specs)
    output_dir = Path(config["output_dir"])
    manifest = _prepare_output_root(output_dir, config, specs)
    print(
        f"suite={args.suite} jobs={len(specs)} concurrency={args.concurrency} "
        f"gpu={config['runtime']['gpu_name']} output={output_dir}",
        flush=True,
    )
    _run_scheduler(
        manifest,
        specs,
        output_dir=output_dir,
        runner_path=Path(config["runner_path"]),
        python_exe=Path(config["python_executable"]),
        dataset_path=Path(config["dataset_path"]),
        concurrency=args.concurrency,
        nvidia_smi=config["nvidia_smi"],
        poll_seconds=args.poll_seconds,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LauncherError as error:
        print(f"launcher error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
