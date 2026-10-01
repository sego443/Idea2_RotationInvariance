"""Compare formal v5 safe checkpoints on synthetic C4 and ideal-phase inputs only."""

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
import run_c170_harmonic_nyquist_experiment_v5 as runner
from src.c170_harmonic_nyquist_model_v5 import HarmonicNyquistClassifierV5
from src.c170_shared_scale_preprocessing_v3 import calibrate_windows_shared_scale_v3


EXPECTED_SCALE = 11123.037109375
DEFAULT_REPORT = (
    PROJECT_DIR
    / "docs"
    / "results"
    / "2026-09-29-c170-harmonic-nyquist-v5-synthetic-equivariance.md"
)
PHASE_ANGLES = (0.37, 1.13, 2.41)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupled-run-dir", type=Path, required=True)
    parser.add_argument("--uncoupled-run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_dict_sha256(values: dict) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_formal_run(run_dir: Path, expected_variant: str) -> tuple[dict, dict, Path, dict]:
    run_dir = run_dir.resolve()
    marker_path = run_dir / "RUN_COMPLETE.json"
    manifest_path = run_dir / "run_manifest.json"
    weights_path = run_dir / "best_weights.npz"
    if not marker_path.is_file() or not manifest_path.is_file() or not weights_path.is_file():
        raise RuntimeError(f"formal run files are incomplete under {run_dir}")

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    config = manifest.get("config", {})
    if marker.get("format") != "c170-harmonic-nyquist-v5-complete" or marker.get("status") != "completed":
        raise RuntimeError(f"unexpected completion marker in {run_dir}")
    if manifest.get("format") != "c170-harmonic-nyquist-v5-run":
        raise RuntimeError(f"unexpected manifest format in {run_dir}")
    if marker.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise RuntimeError(f"manifest hash mismatch in {run_dir}")
    if manifest.get("status") != "completed" or manifest.get("test_metrics_computed") is not False:
        raise RuntimeError(f"incomplete run or test metrics are marked in {run_dir}")
    if config.get("variant") != expected_variant or config.get("seed") != 17:
        raise RuntimeError(f"expected {expected_variant} seed 17, got {config.get('variant')} seed {config.get('seed')}")
    if config.get("max_epochs") != 160 or config.get("patience") != 25 or config.get("device") != "cuda:0":
        raise RuntimeError(f"{run_dir} is not a formal CUDA 160/patience-25 run")
    if manifest.get("preprocessing", {}).get("shared_sensor_scale") != EXPECTED_SCALE:
        raise RuntimeError(f"shared scale differs from the frozen value in {run_dir}")
    if marker.get("config_sha256") != manifest.get("config_sha256"):
        raise RuntimeError(f"config hash mismatch in {run_dir}")
    test_access = manifest.get("test_access", {})
    if any(
        test_access.get(field) is not False
        for field in ("used_as_model_input", "preprocessed", "used_to_estimate_statistics", "predicted_or_scored")
    ):
        raise RuntimeError(f"test access metadata is not sealed in {run_dir}")

    expected_sources = manifest.get("provenance", {}).get("code_sha256", {})
    actual_sources = runner._collect_code_hashes()
    if actual_sources != expected_sources:
        raise RuntimeError(f"current computation source hashes do not match {run_dir}")
    required_artifacts = {"best_checkpoint", "safe_state_dict", "learning_curve", "validation_predictions"}
    if set(marker.get("artifact_sha256", {})) != required_artifacts:
        raise RuntimeError(f"completion marker artifact set is incomplete in {run_dir}")
    if set(manifest.get("artifacts", {})) != required_artifacts:
        raise RuntimeError(f"manifest artifact set is incomplete in {run_dir}")
    for name, expected_hash in marker["artifact_sha256"].items():
        artifact_path = run_dir / manifest.get("artifacts", {}).get(name, {}).get("path", "")
        if not artifact_path.is_file() or _sha256(artifact_path) != expected_hash:
            raise RuntimeError(f"artifact hash mismatch for {name} in {run_dir}")
    if marker.get("artifact_sha256", {}).get("safe_state_dict") != _sha256(weights_path):
        raise RuntimeError(f"numeric NPZ is not covered by the completion marker in {run_dir}")

    source_bundle_sha256 = _canonical_dict_sha256(expected_sources)
    return manifest, marker, weights_path, {
        "run_dir": str(run_dir),
        "selected_epoch_one_based": manifest["selection"]["best_epoch_one_based"],
        "config_sha256": manifest["config_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "run_complete_sha256": _sha256(marker_path),
        "source_bundle_sha256": source_bundle_sha256,
        "runner_sha256": expected_sources["runner"],
        "model_sha256": expected_sources["model"],
        "source_hashes": expected_sources,
        "safe_weights_sha256": _sha256(weights_path),
    }


def _synthetic_components() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create two unlabeled raw-scale windows with nonzero first and Nyquist modes."""
    rng = np.random.default_rng(20260929)
    time = np.arange(500, dtype=np.float64) / 250.0
    common: list[np.ndarray] = []
    first: list[np.ndarray] = []
    quadrature: list[np.ndarray] = []
    nyquist: list[np.ndarray] = []
    for sample in range(2):
        phase = 0.19 * sample
        s = (
            16000.0
            + 5500.0 * np.sin(2.0 * np.pi * 2.2 * time + phase)
            + 1400.0 * np.sin(2.0 * np.pi * 7.0 * time - 0.2)
            + rng.normal(0.0, 350.0, size=time.shape)
        )
        a = (
            10000.0 * np.cos(2.0 * np.pi * 3.3 * time + 0.13 + phase)
            + 2300.0 * np.sin(2.0 * np.pi * 11.0 * time)
            + rng.normal(0.0, 250.0, size=time.shape)
        )
        b = (
            9200.0 * np.sin(2.0 * np.pi * 3.3 * time + 0.13 + phase)
            + 1900.0 * np.cos(2.0 * np.pi * 8.5 * time)
            + rng.normal(0.0, 250.0, size=time.shape)
        )
        n = (
            7200.0 * np.sin(2.0 * np.pi * 4.7 * time - 0.31 + phase)
            + 1700.0 * np.cos(2.0 * np.pi * 13.0 * time)
            + rng.normal(0.0, 220.0, size=time.shape)
        )
        common.append(s)
        first.append(a)
        quadrature.append(b)
        nyquist.append(n)
    return tuple(
        np.stack(values).astype(np.float32)
        for values in (common, first, quadrature, nyquist)
    )  # type: ignore[return-value]


def _reconstruct_channels(s: np.ndarray, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    return np.stack((s + a + n, s + b - n, s - a + n, s - b - n), axis=-1).astype(np.float32)


def _preprocess_v3(raw: np.ndarray, scale: float, device: torch.device) -> torch.Tensor:
    filtered = c170.filter_windows(raw, sample_rate=250)
    normalized, used_scale = calibrate_windows_shared_scale_v3(filtered, scale=scale)
    if used_scale != EXPECTED_SCALE:
        raise RuntimeError("synthetic input did not use the frozen train-derived scalar")
    return torch.from_numpy(normalized.copy()).to(device=device, dtype=torch.float32)


def _errors(
    model: HarmonicNyquistClassifierV5,
    raw: np.ndarray,
    scale: float,
    device: torch.device,
    base_logits: torch.Tensor,
    base_embedding: torch.Tensor,
) -> dict[str, float]:
    values = _preprocess_v3(raw, scale, device)
    with torch.inference_mode():
        logits, embedding = model(values, return_embedding=True)
    cosine_gap = 1.0 - torch.sum(base_embedding * embedding, dim=1)
    return {
        "max_abs_logit_error": float((base_logits - logits).abs().max().cpu()),
        "max_abs_embedding_error": float((base_embedding - embedding).abs().max().cpu()),
        "max_embedding_cosine_gap": float(cosine_gap.abs().max().cpu()),
    }


def _evaluate_run(
    manifest: dict,
    weights_path: Path,
    run_identity: dict,
    variant: str,
    components: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
) -> dict:
    s, a, b, n = components
    scale = float(manifest["preprocessing"]["shared_sensor_scale"])
    model = HarmonicNyquistClassifierV5(variant, n_classes=4).to(device).eval()
    state_dict = runner._load_numpy_state_dict(weights_path, model.state_dict(), device)
    model.load_state_dict(state_dict, strict=True)

    raw = _reconstruct_channels(s, a, b, n)
    base_inputs = _preprocess_v3(raw, scale, device)
    with torch.inference_mode():
        base_logits, base_embedding = model(base_inputs, return_embedding=True)

    rolls = {}
    for shift in range(4):
        rolled = np.roll(raw, shift=shift, axis=-1).copy()
        rolls[str(shift)] = _errors(model, rolled, scale, device, base_logits, base_embedding)

    phases = {}
    for angle in PHASE_ANGLES:
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated_a = a * cosine - b * sine
        rotated_b = a * sine + b * cosine
        phase_raw = _reconstruct_channels(s, rotated_a, rotated_b, n)
        phases[f"{angle:.2f}"] = _errors(
            model, phase_raw, scale, device, base_logits, base_embedding
        )
    return {
        **run_identity,
        "variant": variant,
        "shared_train_scale": scale,
        "channel_roll_errors": rolls,
        "fixed_n_common_z_phase_errors": phases,
    }


def _write_report(report_path: Path, result: dict) -> None:
    rows = []
    for variant in ("nyquist_coupled", "nyquist_uncoupled"):
        arm = result["runs"][variant]
        for shift, errors in arm["channel_roll_errors"].items():
            rows.append(
                f"| {variant} | C4 roll {shift} | {errors['max_abs_logit_error']:.3e} | "
                f"{errors['max_abs_embedding_error']:.3e} | {errors['max_embedding_cosine_gap']:.3e} |"
            )
        for angle, errors in arm["fixed_n_common_z_phase_errors"].items():
            rows.append(
                f"| {variant} | fixed-n z phase {angle} rad | {errors['max_abs_logit_error']:.3e} | "
                f"{errors['max_abs_embedding_error']:.3e} | {errors['max_embedding_cosine_gap']:.3e} |"
            )

    run_rows = []
    for variant in ("nyquist_coupled", "nyquist_uncoupled"):
        arm = result["runs"][variant]
        run_rows.append(
            f"| {variant} | `{arm['run_dir']}` | {arm['selected_epoch_one_based']} | "
            f"`{arm['runner_sha256']}` | `{arm['model_sha256']}` | `{arm['source_bundle_sha256']}` | "
            f"`{arm['run_complete_sha256']}` | `{arm['manifest_sha256']}` | `{arm['safe_weights_sha256']}` |"
        )
    text = "\n".join(
        [
            "# Synthetic-only v5 Nyquist symmetry check",
            "",
            f"Environment: {result['device']}, PyTorch {result['torch_version']} "
            f"(CUDA {result['torch_cuda_build']}), 2 CPU threads, batch size 2.",
            "",
            "| Arm | Formal run path | Selected epoch | Runner SHA-256 | Model SHA-256 | Source bundle SHA-256 | RUN_COMPLETE SHA-256 | Manifest SHA-256 | Safe NPZ SHA-256 |",
            "|---|---|---:|---|---|---|---|---|---|",
            *run_rows,
            "",
            f"The script loaded only the completed-run manifests and numeric `best_weights.npz` files, "
            f"then passed two deterministic unlabeled synthetic raw-scale windows through the v3 filter, "
            f"per-window/per-channel demean, and each checkpoint's fixed train-derived scalar "
            f"{EXPECTED_SCALE:.9f}. It did not open the dataset or calculate test metrics.",
            "",
            "| Arm | Transform | Max abs logit error | Max abs normalized embedding error | Max embedding cosine gap |",
            "|---|---|---:|---:|---:|",
            *rows,
            "",
            "The C4 rows measure discrete cyclic channel rolls on these constructed inputs. The phase rows "
            "rotate only the ideal first-harmonic `(a,b)` pair while holding the synthetic Nyquist component "
            "`n` fixed. The uncoupled arm is expected to be invariant up to float roundoff; the coupled arm "
            "contains `n*z²`, which rotates by `exp(i*2*phi)`, so its observed change is reported rather than "
            "treated as a failure. This is a mathematical negative control, not a physical wrist-rotation or "
            "general SO(2) test. Four-sensor Nyquist and aliasing remain limitations.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen formal validation environment requires CUDA")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    runs = {}
    for variant, run_dir in (
        ("nyquist_coupled", args.coupled_run_dir),
        ("nyquist_uncoupled", args.uncoupled_run_dir),
    ):
        manifest, _, weights_path, identity = _load_formal_run(run_dir, variant)
        runs[variant] = _evaluate_run(
            manifest,
            weights_path,
            identity,
            variant,
            _synthetic_components(),
            device,
        )
    result = {
        "runs": runs,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "batch_size": 2,
        "synthetic_input_only": True,
        "test_metrics_computed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    report_path = args.report.resolve()
    _write_report(report_path, result)
    print(json.dumps(result, sort_keys=True, indent=2))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
