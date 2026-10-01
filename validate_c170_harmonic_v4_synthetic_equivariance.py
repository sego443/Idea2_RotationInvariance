"""Check v4 C4 and ideal first-harmonic phase behavior on synthetic inputs only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_experiment as c170
import run_c170_harmonic_phase_experiment_v4 as runner
from src.c170_harmonic_phase_model_v4 import HarmonicPhaseClassifierV4
from src.c170_shared_scale_preprocessing_v3 import calibrate_windows_shared_scale_v3


EXPECTED_SCALE = 11123.037109375
DEFAULT_REPORT = PROJECT_DIR / "docs" / "results" / "2026-09-29-c170-harmonic-v4-synthetic-equivariance.md"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_completed_run(run_dir: Path) -> tuple[dict, dict, Path]:
    manifest_path = run_dir / "run_manifest.json"
    marker_path = run_dir / "RUN_COMPLETE.json"
    weights_path = run_dir / "best_weights.npz"
    if not marker_path.is_file():
        raise RuntimeError(f"RUN_COMPLETE marker is missing: {marker_path}")
    if not manifest_path.is_file() or not weights_path.is_file():
        raise RuntimeError("completed-run manifest or safe NPZ weights are missing")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if marker.get("format") != "c170-ring-cuda-v4-complete" or marker.get("status") != "completed":
        raise RuntimeError("RUN_COMPLETE marker has an unexpected format or status")
    if marker.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise RuntimeError("RUN_COMPLETE manifest hash does not match run_manifest.json")
    if manifest.get("status") != "completed" or manifest.get("test_metrics_computed") is not False:
        raise RuntimeError("run manifest is incomplete or claims test metrics were computed")
    if manifest.get("config", {}).get("variant") != "harmonic_gram":
        raise RuntimeError("the checkpoint is not from the harmonic_gram arm")
    if manifest.get("config", {}).get("seed") != 17:
        raise RuntimeError("the checkpoint is not from seed 17")
    if manifest.get("config", {}).get("max_epochs") != 160:
        raise RuntimeError("diagnostic requires the formal 160-epoch-budget run, not a smoke")
    if manifest.get("config", {}).get("patience") != 25:
        raise RuntimeError("diagnostic requires the formal patience-25 run, not a smoke")
    if manifest.get("config", {}).get("device") != "cuda:0":
        raise RuntimeError("diagnostic requires the frozen CUDA training run")
    if manifest.get("preprocessing", {}).get("shared_sensor_scale") != EXPECTED_SCALE:
        raise RuntimeError("checkpoint preprocessing scale differs from the frozen train-only scalar")
    if marker.get("config_sha256") != manifest.get("config_sha256"):
        raise RuntimeError("completion marker config hash differs from manifest")

    for name, expected_hash in marker.get("artifact_sha256", {}).items():
        artifact = run_dir / manifest.get("artifacts", {}).get(name, {}).get("path", "")
        if not artifact.is_file() or _sha256(artifact) != expected_hash:
            raise RuntimeError(f"completed artifact hash check failed: {name}")
    if marker.get("artifact_sha256", {}).get("safe_state_dict") != _sha256(weights_path):
        raise RuntimeError("safe NPZ weights are not covered by the completion marker")
    return manifest, marker, weights_path


def _synthetic_raw_like() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make two deterministic raw-scale windows from s, a, b without labels."""
    rng = np.random.default_rng(20260929)
    time = np.arange(500, dtype=np.float64) / 250.0
    common: list[np.ndarray] = []
    first: list[np.ndarray] = []
    quadrature: list[np.ndarray] = []
    for sample in range(2):
        offset = 0.19 * sample
        s = (
            16000.0
            + 5500.0 * np.sin(2.0 * np.pi * 2.2 * time + offset)
            + 1400.0 * np.sin(2.0 * np.pi * 7.0 * time - 0.2)
            + rng.normal(0.0, 350.0, size=time.shape)
        )
        a = (
            10000.0 * np.cos(2.0 * np.pi * 3.3 * time + 0.13 + offset)
            + 2300.0 * np.sin(2.0 * np.pi * 11.0 * time)
            + rng.normal(0.0, 250.0, size=time.shape)
        )
        b = (
            9200.0 * np.sin(2.0 * np.pi * 3.3 * time + 0.13 + offset)
            + 1900.0 * np.cos(2.0 * np.pi * 8.5 * time)
            + rng.normal(0.0, 250.0, size=time.shape)
        )
        common.append(s)
        first.append(a)
        quadrature.append(b)
    return (
        np.stack(common).astype(np.float32),
        np.stack(first).astype(np.float32),
        np.stack(quadrature).astype(np.float32),
    )


def _reconstruct_channels(s: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack((s + a, s + b, s - a, s - b), axis=-1).astype(np.float32)


def _v3_preprocess(raw: np.ndarray, scale: float, device: torch.device) -> torch.Tensor:
    filtered = c170.filter_windows(raw, sample_rate=250)
    normalized, used_scale = calibrate_windows_shared_scale_v3(filtered, scale=scale)
    if used_scale != EXPECTED_SCALE:
        raise RuntimeError("synthetic input did not use the fixed train-derived scale")
    return torch.from_numpy(normalized.copy()).to(device=device, dtype=torch.float32)


def _max_abs_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(torch.max(torch.abs(reference - candidate)).detach().cpu())


def _compare(
    model: HarmonicPhaseClassifierV4,
    candidate_raw: np.ndarray,
    scale: float,
    device: torch.device,
    base_logits: torch.Tensor,
    base_embedding: torch.Tensor,
) -> dict[str, float]:
    candidate_inputs = _v3_preprocess(candidate_raw, scale, device)
    with torch.inference_mode():
        logits, embedding = model(candidate_inputs, return_embedding=True)
    cosine_gap = 1.0 - torch.sum(base_embedding * embedding, dim=1)
    return {
        "max_abs_logit_error": _max_abs_error(base_logits, logits),
        "max_abs_embedding_error": _max_abs_error(base_embedding, embedding),
        "max_embedding_cosine_gap": float(torch.max(torch.abs(cosine_gap)).detach().cpu()),
    }


def _run(args: argparse.Namespace) -> dict:
    run_dir = args.run_dir.resolve()
    manifest, marker, weights_path = _verify_completed_run(run_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("the requested frozen validation environment has no usable CUDA device")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    model = HarmonicPhaseClassifierV4("harmonic_gram", n_classes=4).to(device).eval()
    safe_state = runner._load_numpy_state_dict(weights_path, model.state_dict(), device)
    model.load_state_dict(safe_state, strict=True)

    s, a, b = _synthetic_raw_like()
    raw = _reconstruct_channels(s, a, b)
    scale = float(manifest["preprocessing"]["shared_sensor_scale"])
    base_inputs = _v3_preprocess(raw, scale, device)
    with torch.inference_mode():
        base_logits, base_embedding = model(base_inputs, return_embedding=True)

    roll_errors = {}
    for shift in range(4):
        rolled = np.roll(raw, shift=shift, axis=-1).copy()
        roll_errors[str(shift)] = _compare(
            model, rolled, scale, device, base_logits, base_embedding
        )

    phase_errors = {}
    for angle in (0.37, 1.13, 2.41):
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated_a = a * cosine - b * sine
        rotated_b = a * sine + b * cosine
        phase_raw = _reconstruct_channels(s, rotated_a, rotated_b)
        phase_errors[f"{angle:.2f}"] = _compare(
            model, phase_raw, scale, device, base_logits, base_embedding
        )

    return {
        "run_dir": str(run_dir),
        "variant": manifest["config"]["variant"],
        "seed": manifest["config"]["seed"],
        "selected_epoch_one_based": manifest["selection"]["best_epoch_one_based"],
        "config_sha256": manifest["config_sha256"],
        "manifest_sha256": hashlib.sha256((run_dir / "run_manifest.json").read_bytes()).hexdigest(),
        "run_complete_sha256": _sha256(run_dir / "RUN_COMPLETE.json"),
        "safe_weights_sha256": _sha256(weights_path),
        "shared_train_scale": scale,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "batch_size": int(raw.shape[0]),
        "synthetic_input_only": True,
        "test_metrics_computed": False,
        "channel_roll_errors": roll_errors,
        "ideal_first_harmonic_common_phase_errors": phase_errors,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def _write_report(report_path: Path, result: dict) -> None:
    roll_rows = []
    for shift, errors in result["channel_roll_errors"].items():
        roll_rows.append(
            f"| {shift} | {errors['max_abs_logit_error']:.3e} | "
            f"{errors['max_abs_embedding_error']:.3e} | {errors['max_embedding_cosine_gap']:.3e} |"
        )
    phase_rows = []
    for angle, errors in result["ideal_first_harmonic_common_phase_errors"].items():
        phase_rows.append(
            f"| {angle} rad | {errors['max_abs_logit_error']:.3e} | "
            f"{errors['max_abs_embedding_error']:.3e} | {errors['max_embedding_cosine_gap']:.3e} |"
        )
    text = "\n".join(
        [
            "# Synthetic-only v4 symmetry check",
            "",
            f"Run: `{result['run_dir']}` (harmonic_gram, seed {result['seed']}); "
            f"selected epoch: {result['selected_epoch_one_based']}; config SHA-256: "
            f"`{result['config_sha256']}`; manifest SHA-256: `{result['manifest_sha256']}`; "
            f"RUN_COMPLETE SHA-256: `{result['run_complete_sha256']}`; safe NPZ SHA-256: "
            f"`{result['safe_weights_sha256']}`.",
            f"Environment: {result['device']}, PyTorch {result['torch_version']} "
            f"(CUDA {result['torch_cuda_build']}), {result['cpu_threads']} CPU threads, "
            f"batch size {result['batch_size']}.",
            f"Input preprocessing: v3 0.5–40 Hz order-3 zero-phase filter, per-window/per-channel "
            f"demean, and fixed train-derived shared scalar {result['shared_train_scale']:.9f}.",
            "",
            "All inputs below are deterministic synthetic raw-scale waveforms with no labels. "
            "The script loads only the completed run manifest and numeric `best_weights.npz`; "
            "it does not open the dataset or calculate test metrics.",
            "",
            "| Cyclic channel roll | Max abs logit error | Max abs normalized embedding error | Max cosine gap |",
            "|---:|---:|---:|---:|",
            *roll_rows,
            "",
            "| Synthetic first-harmonic common phase | Max abs logit error | Max abs normalized embedding error | Max cosine gap |",
            "|---:|---:|---:|---:|",
            *phase_rows,
            "",
            "The channel-roll check tests the model's discrete C4 behavior on these constructed inputs. "
            "The phase check rotates only the synthetic first-harmonic `(a,b)` pair; it is an idealized "
            "mathematical action, not a physical wrist rotation or evidence of general SO(2) invariance. "
            "Continuous-angle invariance requires the strict spatial bandlimit `|m|<=1`; four-sensor Nyquist "
            "and aliased higher harmonics are not covered. This check is diagnostic only and does not alter "
            "model selection or the validation/test boundary.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    result = _run(args)
    report_path = args.report.resolve()
    _write_report(report_path, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
