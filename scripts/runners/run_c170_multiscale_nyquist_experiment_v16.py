"""One C170 v16 run; uses the frozen v11 training, split and selection protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / "src"))

from scripts.runners import run_c170_complex_orbit_experiment_v11 as v11
from src.c170_multiscale_nyquist_v16 import (
    ARCHITECTURES, C170MultiscaleNyquistClassifierV16, DESCRIPTOR_DIM, PARAMETER_COUNT,
)


def install_v16(architecture: str) -> None:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown v16 architecture: {architecture}")
    original_config = v11._variant_config
    original_hashes = v11._collect_code_hashes
    original_hash = v11.v2._canonical_json_sha256
    original_write = v11.v2._atomic_write_json

    def config(variant: str):
        value = original_config(variant)
        value.update({
            "model_variant": f"c170_complex_gram_v16_{architecture}",
            "architecture": "complex_gram_v16_" + architecture,
            "descriptor_dim": DESCRIPTOR_DIM[architecture],
            "total_parameters": PARAMETER_COUNT[architecture],
            "trainable_parameters": PARAMETER_COUNT[architecture],
        })
        return value

    def hashes():
        values = original_hashes()
        values["multiscale_nyquist_runner_v16"] = v11.v2._file_sha256(Path(__file__).resolve())
        values["multiscale_nyquist_model_v16"] = v11.v2._file_sha256(PROJECT / "src" / "c170_multiscale_nyquist_v16.py")
        return values

    def canonical_hash(value):
        value["architecture"]["name"] = "complex_gram_v16_" + architecture
        value["architecture"]["descriptor_dim"] = DESCRIPTOR_DIM[architecture]
        value["architecture"]["total_parameters"] = PARAMETER_COUNT[architecture]
        value["architecture"]["trainable_parameters"] = PARAMETER_COUNT[architecture]
        value["feature_definition"]["complex_convolution"] = (
            "v16 M: two parallel no-bias complex Conv1d branches 1->8, k9/p4 and k65/p32, stride2, concat to16; "
            "otherwise original 1->16 k33/p16; second layer 16->24 k9/p4 stride2"
        )
        value["feature_definition"]["nyquist_branch"] = (
            "N: n=(S0-S1+S2-S3)/2; shared real Conv1d 1->8 k9/s2/p4, GroupNorm4, SiLU applied to n and -n; "
            "average their full 64-d temporal descriptors before concatenating to original 672-d descriptor"
        ) if architecture in {"nyquist", "multiscale_nyquist"} else None
        value["v16_architecture"] = {
            "name": architecture,
            "multiscale_complex_first_layer": architecture in {"multiscale", "multiscale_nyquist"},
            "sign_invariant_nyquist_branch": architecture in {"nyquist", "multiscale_nyquist"},
            "descriptor_dim": DESCRIPTOR_DIM[architecture],
            "parameter_count": PARAMETER_COUNT[architecture],
            "nyquist_sign_invariance": "shared branch on n and -n; average after complete descriptor; no angle supervision",
        }
        return original_hash(value)

    def write(path, value):
        name = Path(path).name
        if name == "run_manifest.json":
            value["format"] = "c170-multiscale-nyquist-v16-run"
        elif name == "RUN_COMPLETE.json":
            value["format"] = "c170-multiscale-nyquist-v16-complete"
        return original_write(path, value)

    class V16Classifier(C170MultiscaleNyquistClassifierV16):
        def __init__(self, variant: str, n_classes: int = 4) -> None:
            super().__init__(variant, architecture, n_classes)

    v11._variant_config = config
    v11._collect_code_hashes = hashes
    v11.v2._canonical_json_sha256 = canonical_hash
    v11.v2._atomic_write_json = write
    v11.C170ComplexOrbitClassifierV11 = V16Classifier


def real_train_sanity(dataset: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v16 smoke check")
    device = torch.device("cuda:0")
    prepared = v11._validate_and_prepare_v11(
        SimpleNamespace(dataset=dataset, confirm_physical_order=True, variant="complex_gram", seed=17), device
    )
    tensors = prepared[3]["tensors"]
    samples = tensors["train_inputs"][:4]
    labels = tensors["train_targets"][:4]
    results: dict[str, object] = {"real_train_windows": int(samples.shape[0]), "device": str(device), "test_metrics_computed": False}
    for index, architecture in enumerate(ARCHITECTURES):
        torch.manual_seed(1600 + index)
        model = C170MultiscaleNyquistClassifierV16("complex_gram", architecture).to(device).eval()
        with torch.no_grad():
            reference = model(samples)
            errors = [float((model(torch.roll(samples, shifts=shift, dims=-1)) - reference).abs().max()) for shift in range(1, 4)]
        model.train()
        loss = F.cross_entropy(model(samples), labels)
        loss.backward()
        finite = bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if not finite or max(errors) > 2e-5:
            raise RuntimeError(f"{architecture} real-data C4/finite sanity failed: errors={errors}, finite={finite}")
        results[architecture] = {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "descriptor_dim": DESCRIPTOR_DIM[architecture],
            "c4_max_logit_errors": errors,
            "finite_loss_and_gradients": finite,
        }
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
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
        import json
        print(json.dumps(real_train_sanity(args.dataset), indent=2), flush=True)
        return 0
    if args.output_dir is None or args.architecture is None or args.seed is None:
        raise ValueError("output-dir, architecture, and seed are required for training")
    if not args.confirm_physical_order:
        raise ValueError("pass --confirm-physical-order after confirming S0-S3")
    install_v16(args.architecture)
    variant = "complex_gram_shuffle" if args.shuffle_labels else "complex_gram"
    return v11.main([
        "--output-dir", str(args.output_dir), "--variant", variant, "--seed", str(args.seed),
        "--dataset", str(args.dataset), "--max-epochs", str(args.max_epochs), "--patience", str(args.patience),
        "--cpu-threads", str(args.cpu_threads), "--device", args.device, "--confirm-physical-order",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
