"""One C170 sensor-normalization v14 run using the frozen complex_gram trainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from src.c170_sensor_normalization_v14 import (
    GAIN_BOUND, LEGACY_SHARED_SCALE_SENTINEL, NORMALIZATIONS, REGULARIZATION_WEIGHT,
    prepare_train_val,
)
from scripts.runners import run_c170_complex_orbit_experiment_v11 as v11
def _gain_model_class():
    base_class = v11.C170ComplexOrbitClassifierV11
    active: dict[str, nn.Module | None] = {"model": None}

    class BoundedGainComplexOrbitClassifierV14(base_class):
        def __init__(self, variant: str, n_classes: int = 4) -> None:
            super().__init__(variant, n_classes)
            self.sensor_theta = nn.Parameter(torch.zeros(4))
            active["model"] = self

        def sensor_log_gain(self) -> torch.Tensor:
            bounded = GAIN_BOUND * torch.tanh(self.sensor_theta)
            return bounded - bounded.mean()

        def forward(self, samples: torch.Tensor) -> torch.Tensor:
            calibrated = samples * torch.exp(self.sensor_log_gain()).view(1, 1, 4)
            return super().forward(calibrated)

        def regularization_loss(self) -> torch.Tensor:
            return REGULARIZATION_WEIGHT * self.sensor_log_gain().square().mean()

    return BoundedGainComplexOrbitClassifierV14, active


def install_v14(normalization: str) -> None:
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"unknown normalization: {normalization}")
    original_config = v11._variant_config
    original_hashes = v11._collect_code_hashes
    original_canonical_hash = v11.v2._canonical_json_sha256
    original_write_json = v11.v2._atomic_write_json
    original_validate = v11._validate_and_prepare_v11
    original_parser = v11._parser
    original_prepare = v11.v3.prepare_model_inputs_shared_scale_v3
    original_class = v11.C170ComplexOrbitClassifierV11

    active_info: dict[str, object] = {}
    gain_active: dict[str, nn.Module | None] | None = None

    def variant_config(variant: str):
        config = original_config(variant)
        config["model_variant"] = f"{config['model_variant']}_sensor_normalization_v14_{normalization}"
        config["preprocessing"] = {
            "sensor_median": "v3 SOS filter + per-window channel demean + train-only per-sensor pooled median(abs) normalization",
            "sensor_rms": "v3 SOS filter + per-window channel demean + train-only per-sensor pooled RMS normalization",
            "window_sensor_rms": "v3 SOS filter + per-window channel demean + per-window per-sensor temporal RMS normalization",
            "bounded_sensor_gain": "v3 SOS filter + per-window channel demean + existing train-only shared median(abs) scale + bounded learned sensor gain",
        }[normalization]
        config["preprocessing_scale_required"] = (
            LEGACY_SHARED_SCALE_SENTINEL if normalization == "bounded_sensor_gain" else None
        )
        config["preprocessing_scale_fit_split"] = "train" if normalization != "window_sensor_rms" else None
        config["sensor_normalization"] = {
            "kind": normalization,
            "sensor_order": ["S0", "S1", "S2", "S3"],
            "scope": "fixed physical sensor identity; no label/position input",
            "definition": {
                "sensor_median": "filtered per-window demeaned train-only pooled per-sensor median(abs(x)); floor 1e-6",
                "sensor_rms": "filtered per-window demeaned train-only pooled per-sensor sqrt(mean(x^2)); floor 1e-6",
                "window_sensor_rms": "each window and sensor divided by its own temporal RMS; floor 1e-6; sample-adaptive amplitude normalization, not hardware calibration",
                "bounded_sensor_gain": "existing shared train-only scale then exp(0.1*(tanh(theta)-mean(tanh(theta)))); theta zero-initialized; no FIR",
            }[normalization],
            "regularization": {
                "weight": REGULARIZATION_WEIGHT if normalization == "bounded_sensor_gain" else 0.0,
                "definition": "1e-3*mean(centered_log_gain^2)" if normalization == "bounded_sensor_gain" else None,
            },
        }
        return config

    def collect_hashes():
        hashes = original_hashes()
        hashes["sensor_normalization_runner_v14"] = v11.v2._file_sha256(Path(__file__).resolve())
        hashes["sensor_normalization_module_v14"] = v11.v2._file_sha256(PROJECT / "src" / "c170_sensor_normalization_v14.py")
        return hashes

    def canonical_hash(config):
        # The upstream runner builds a mutable config dictionary; include the full exact v14 discriminator.
        return original_canonical_hash(config)

    def write_json(path, value):
        if Path(path).name == "run_manifest.json":
            value["format"] = "c170-sensor-normalization-v14-run"
            if active_info:
                value.setdefault("preprocessing", {}).update(active_info)
        elif Path(path).name == "RUN_COMPLETE.json":
            value["format"] = "c170-sensor-normalization-v14-complete"
        return original_write_json(path, value)

    def add_parser_option():
        parser = original_parser()
        parser.add_argument("--normalization", choices=NORMALIZATIONS, required=True)
        return parser

    def validate_v14(args, device):
        if args.normalization != normalization:
            raise ValueError("run normalization differs from installed v14 configuration")
        if normalization == "bounded_sensor_gain":
            result = original_validate(args, device)
            info = dict(result[3]["preprocessing"])
            info.update({
                "sensor_normalization": normalization,
                "sensor_order": ["S0", "S1", "S2", "S3"],
                "fitted_sensor_scales": [float(info["shared_sensor_scale"])] * 4,
                "fit_scope": "existing v11 shared scalar fit on training windows",
                "scale_floor": float(torch.finfo(torch.float32).eps),
                "processed_splits": ["train", "validation"],
                "test_preprocessed": False,
                "interpretation": "bounded trainable gains are fixed to physical sensor identity; no label/position input",
            })
            active_info.update(info)
            result[3]["preprocessing"] = info
            return result

        holder: dict[str, object] = {}

        def prepare(windows, splits):
            processed, info = prepare_train_val(normalization, windows, splits)
            holder.update(info)
            active_info.update(info)
            return processed, LEGACY_SHARED_SCALE_SENTINEL

        v11.v3.prepare_model_inputs_shared_scale_v3 = prepare
        try:
            result = original_validate(args, device)
        finally:
            v11.v3.prepare_model_inputs_shared_scale_v3 = original_prepare
        preprocessing = result[3]["preprocessing"]
        preprocessing.pop("shared_sensor_scale", None)
        preprocessing.pop("shared_sensor_scale_required_by_v11", None)
        preprocessing.update(holder)
        result[2]["v14_sensor_normalization"] = normalization
        return result

    v11._variant_config = variant_config
    v11._collect_code_hashes = collect_hashes
    v11.v2._canonical_json_sha256 = canonical_hash
    v11.v2._atomic_write_json = write_json
    v11._parser = add_parser_option
    v11._validate_and_prepare_v11 = validate_v14

    if normalization == "bounded_sensor_gain":
        model_class, gain_active = _gain_model_class()
        original_objective = v11.v2.batch_objective

        def objective(per_sample_ce, sample_weights, supcon_loss, *, use_supcon):
            base = original_objective(per_sample_ce, sample_weights, supcon_loss, use_supcon=use_supcon)
            model = gain_active["model"] if gain_active is not None else None
            return base if model is None else base + model.regularization_loss()

        v11.v2.batch_objective = objective
        v11.C170ComplexOrbitClassifierV11 = model_class


def sanity_check() -> dict[str, object]:
    rng = np.random.default_rng(1401)
    train = rng.normal(size=(9, 500, 4)).astype(np.float32)
    val = rng.normal(size=(3, 500, 4)).astype(np.float32)
    results: dict[str, object] = {}
    for kind in NORMALIZATIONS[:3]:
        data = np.concatenate((train, val), axis=0)
        prepared, info = prepare_train_val(kind, data, {"train": np.arange(9), "val": np.arange(9, 12)})
        if prepared["train"].shape != train.shape or prepared["val"].shape != val.shape:
            raise RuntimeError(f"{kind} produced wrong shape")
        if not np.isfinite(prepared["train"]).all() or not np.isfinite(prepared["val"]).all():
            raise RuntimeError(f"{kind} produced non-finite values")
        results[kind] = {"finite": True, "train_shape": list(prepared["train"].shape), "metadata": info}
    torch.manual_seed(1402)
    base = v11.C170ComplexOrbitClassifierV11("complex_gram").eval()
    cls, _ = _gain_model_class()
    torch.manual_seed(1402)
    model = cls("complex_gram").eval()
    with torch.no_grad():
        x = torch.from_numpy(train[:2])
        error = float((base(x) - model(x)).abs().max())
    if error != 0.0:
        raise RuntimeError(f"bounded gain identity initialization changed logits: {error}")
    model.train()
    loss = F.cross_entropy(model(torch.from_numpy(train[:2])), torch.tensor([0, 1])) + model.regularization_loss()
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("bounded gain produced non-finite loss or gradients")
    results["bounded_sensor_gain"] = {"identity_error": error, "finite_loss_and_gradients": True}
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--normalization", choices=NORMALIZATIONS)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43, 61, 89))
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--dataset", type=Path, default=v11.DEFAULT_DATASET)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--confirm-physical-order", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sanity_only:
        print(sanity_check(), flush=True)
        return 0
    if args.output_dir is None or args.normalization is None or args.seed is None:
        raise ValueError("output-dir, normalization, and seed are required")
    if not args.confirm_physical_order:
        raise ValueError("pass --confirm-physical-order after confirming S0-S3")
    install_v14(args.normalization)
    base_variant = "complex_gram_shuffle" if args.shuffle_labels else "complex_gram"
    return v11.main([
        "--output-dir", str(args.output_dir), "--variant", base_variant,
        "--seed", str(args.seed), "--dataset", str(args.dataset),
        "--max-epochs", str(args.max_epochs), "--patience", str(args.patience),
        "--cpu-threads", str(args.cpu_threads), "--device", args.device,
        "--normalization", args.normalization, "--confirm-physical-order",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
