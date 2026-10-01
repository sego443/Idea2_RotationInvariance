"""One C170 v13 sensor-adapter run using the frozen v11 complex_gram trainer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

from src.c170_sensor_adapter_v13 import ADAPTERS, REGULARIZATION_WEIGHT, SensorAdapterV13
import run_c170_complex_orbit_experiment_v11 as v11


def _make_adapted_class(adapter_kind: str):
    base_class = v11.C170ComplexOrbitClassifierV11
    active: dict[str, nn.Module | None] = {"model": None}

    class AdaptedComplexOrbitClassifierV13(base_class):
        def __init__(self, variant: str, n_classes: int = 4) -> None:
            super().__init__(variant, n_classes)
            self.sensor_adapter = SensorAdapterV13(adapter_kind)
            active["model"] = self

        def forward(self, samples: torch.Tensor) -> torch.Tensor:
            return super().forward(self.sensor_adapter(samples))

        def regularization_loss(self) -> torch.Tensor:
            return self.sensor_adapter.regularization_loss()

    original_objective = v11.v2.batch_objective

    def objective_with_adapter(per_sample_ce, sample_weights, supcon_loss, *, use_supcon):
        data_objective = original_objective(
            per_sample_ce, sample_weights, supcon_loss, use_supcon=use_supcon
        )
        model = active["model"]
        if model is None:
            return data_objective
        return data_objective + model.regularization_loss()

    v11.v2.batch_objective = objective_with_adapter
    return AdaptedComplexOrbitClassifierV13


def _install_v13_provenance(adapter_kind: str) -> None:
    original_variant_config = v11._variant_config
    original_hashes = v11._collect_code_hashes
    original_canonical_hash = v11.v2._canonical_json_sha256
    original_write_json = v11.v2._atomic_write_json

    def variant_config(variant: str):
        config = original_variant_config(variant)
        config["model_variant"] = f"{config['model_variant']}_sensor_adapter_v13_{adapter_kind}"
        return config

    def collect_hashes():
        hashes = original_hashes()
        hashes["sensor_adapter_runner_v13"] = v11.v2._file_sha256(Path(__file__).resolve())
        hashes["sensor_adapter_module_v13"] = v11.v2._file_sha256(
            PROJECT / "src" / "c170_sensor_adapter_v13.py"
        )
        return hashes

    def canonical_hash(config):
        config["sensor_adapter"] = {
            "kind": adapter_kind,
            "sensor_identity_binding": "fixed physical S0-S3 channel order; no labels or positions",
            "initialization": "identity; zero log-gain and/or zero-sum FIR residual",
            "gain_parameterization": "exp(log_gain - mean(log_gain)); initialized at one",
            "fir_parameterization": "identity plus depthwise residual; residual taps centered to zero sum",
            "fir_taps": 9,
            "shared_fir_gain": 1.0 if adapter_kind == "shared_fir" else None,
            "regularization": {
                "weight": REGULARIZATION_WEIGHT,
                "definition": "weight*(mean(centered_log_gain^2)+mean(centered_fir_residual^2)); omit absent terms",
                "added_to_each_weighted_ce_batch_objective": True,
            },
            "applied_before": "v11_complex_gram_network",
            "claim_scope": "channel-specific adapter is fixed to sensor identity; C4 applies after calibration",
        }
        config["sensor_adapter_regularization_weight"] = REGULARIZATION_WEIGHT
        return original_canonical_hash(config)

    def write_json(path, value):
        if Path(path).name == "run_manifest.json":
            value["format"] = "c170-sensor-adapter-v13-run"
        elif Path(path).name == "RUN_COMPLETE.json":
            value["format"] = "c170-sensor-adapter-v13-complete"
        return original_write_json(path, value)

    v11._variant_config = variant_config
    v11._collect_code_hashes = collect_hashes
    v11.v2._canonical_json_sha256 = canonical_hash
    v11.v2._atomic_write_json = write_json


def identity_and_gradient_sanity() -> dict[str, object]:
    original_class = v11.C170ComplexOrbitClassifierV11
    x = torch.randn((2, 500, 4), generator=torch.Generator().manual_seed(991), dtype=torch.float32)
    results = {}
    for kind in ADAPTERS:
        torch.manual_seed(37)
        baseline = original_class("complex_gram", n_classes=4).eval()
        torch.manual_seed(37)
        adapted_class = _make_adapted_class(kind)
        adapted = adapted_class("complex_gram", n_classes=4).eval()
        with torch.no_grad():
            baseline_logits = baseline(x)
            adapted_logits = adapted(x)
        max_error = float((baseline_logits - adapted_logits).abs().max())
        if not torch.equal(baseline_logits, adapted_logits):
            raise RuntimeError(f"{kind} identity initialization changed baseline logits: {max_error}")
        adapted.train()
        logits = adapted(x)
        labels = torch.tensor([0, 2], dtype=torch.long)
        loss = F.cross_entropy(logits, labels) + adapted.regularization_loss()
        loss.backward()
        grad_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in adapted.parameters()
        )
        if not torch.isfinite(loss) or not grad_finite:
            raise RuntimeError(f"{kind} produced non-finite loss or gradient")
        results[kind] = {"identity_exact": True, "max_logit_error": max_error, "finite_gradients": True}
    v11.C170ComplexOrbitClassifierV11 = original_class
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--adapter", choices=ADAPTERS)
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
        print(identity_and_gradient_sanity(), flush=True)
        return 0
    if args.output_dir is None or args.adapter is None or args.seed is None:
        raise ValueError("output-dir, adapter, and seed are required for training")
    if not args.confirm_physical_order:
        raise ValueError("pass --confirm-physical-order after confirming the S0-S3 order")
    base_variant = "complex_gram_shuffle" if args.shuffle_labels else "complex_gram"
    _install_v13_provenance(args.adapter)
    v11.C170ComplexOrbitClassifierV11 = _make_adapted_class(args.adapter)
    runner_args = [
        "--output-dir", str(args.output_dir), "--variant", base_variant, "--seed", str(args.seed),
        "--dataset", str(args.dataset), "--max-epochs", str(args.max_epochs),
        "--patience", str(args.patience), "--cpu-threads", str(args.cpu_threads),
        "--device", args.device, "--confirm-physical-order",
    ]
    return v11.main(runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
