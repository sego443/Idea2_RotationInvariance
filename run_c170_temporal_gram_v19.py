"""One C170 v19 temporal-Gram training run on the frozen v11 protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "src"))

import run_c170_complex_orbit_experiment_v11 as v11
from src.c170_temporal_gram_v19 import (
    ARCHITECTURES,
    DESCRIPTOR_DIMS,
    RESOLUTIONS,
    C170TemporalGramClassifierV19,
    parameter_count,
)


DEFAULT_OUTPUT_ROOT = PROJECT / "outputs/c170_temporal_gram_v19_formal_20260930"
SEEDS = (17, 29, 43, 61, 89)
_ACTIVE_ARCHITECTURE: str | None = None
_ACTIVE_RESOLUTION: int | None = None


def install_v19(architecture: str, resolution: int) -> None:
    if architecture not in ARCHITECTURES or resolution not in RESOLUTIONS:
        raise ValueError(f"invalid v19 configuration: {architecture}/L{resolution}")
    global _ACTIVE_ARCHITECTURE, _ACTIVE_RESOLUTION
    _ACTIVE_ARCHITECTURE, _ACTIVE_RESOLUTION = architecture, int(resolution)

    old_config = v11._variant_config
    old_hashes = v11._collect_code_hashes
    old_manifest_base = v11._manifest_base
    old_canonical_hash = v11.v2._canonical_json_sha256
    old_write = v11.v2._atomic_write_json
    name = f"temporal_gram_v19_{architecture}_L{resolution}"
    dimensions = DESCRIPTOR_DIMS[architecture]
    count = parameter_count(architecture, resolution)

    def variant_config(variant: str) -> dict[str, Any]:
        value = old_config(variant)
        if variant.removesuffix("_shuffle") != "complex_gram":
            raise ValueError(f"v19 requires the frozen complex_gram protocol, got {variant}")
        value.update({
            "model_variant": f"c170_{name}",
            "architecture": name,
            "descriptor_dim": dimensions,
            "total_parameters": count,
            "trainable_parameters": count,
            "v19_architecture": architecture,
            "temporal_resolution": resolution,
            "initialization_rng": "v11 trainer resets the registered seed immediately before v19 module construction",
        })
        return value

    def collect_hashes() -> dict[str, str]:
        result = old_hashes()
        result["temporal_gram_runner_v19"] = v11.v2._file_sha256(Path(__file__).resolve())
        result["temporal_gram_model_v19"] = v11.v2._file_sha256(PROJECT / "src" / "c170_temporal_gram_v19.py")
        return result

    def canonical_hash(value: dict[str, Any]) -> str:
        if isinstance(value, dict) and value.get("variant") in v11.VARIANTS and value.get("model_variant") == f"c170_{name}":
            value["architecture"]["name"] = name
            value["architecture"]["descriptor_dim"] = dimensions
            value["architecture"]["total_parameters"] = count
            value["architecture"]["trainable_parameters"] = count
            value["architecture"]["dropout"] = 0.2
            value["feature_definition"].update({
                "complex_input": "z=(S0-S2)/2+i*(S1-S3)/2; n=(S0-S1+S2-S3)/2; s=mean(S0..S3)",
                "temporal_resampling": f"adaptive_avg_pool1d real(z), imag(z), and n to L={resolution}; linear and channel-consistent",
                "temporal_gram_image": "Re(u_i*conj(u_j)), Im(u_i*conj(u_j)), Aij=.5*(log1p(p_i)+log1p(p_j)); p=|z|^2; u=z/sqrt(1+p)",
                "nyquist_image": "Gij=g_i*g_j for g=n/sqrt(1+n^2); present for Gzn and Hybrid, sign-invariant under n -> -n",
                "image_cnn": "Conv2d Cin->8 k5s2p2 no-bias GN4 SiLU; 8->16 k3s1p1 GN4 SiLU; 16->24 k3s1p1 GN6 SiLU; adaptive average pools 1x1,2x2,4x4 plus spatial population std",
                "power_descriptors": "log1p(|z|^2), and for Gzn/Hybrid log1p(n^2); adaptive temporal average bins 1,2,4 plus population std",
                "s_path": "Gz/Gzn reuse v11 64D S-branch; Hybrid reuses original 672D pre-dropout complex_gram descriptor",
                "claim_scope": "mathematical discrete C4 invariance; no claim of verified continuous physical rotation performance",
            })
            value["v19_architecture"] = {
                "name": architecture,
                "temporal_resolution": resolution,
                "image_channels": 3 if architecture == "gz" else 4,
                "image_descriptor_dim": 528,
                "descriptor_dim": dimensions,
                "total_parameters": count,
                "trainable_parameters": count,
                "dropout": 0.2,
                "registered_test_touched": False,
            }
        return old_canonical_hash(value)

    def manifest_base(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = old_manifest_base(*args, **kwargs)
        value["format"] = "c170-temporal-gram-v19-run"
        value.setdefault("metadata", {})["v19_architecture"] = {
            "name": architecture,
            "temporal_resolution": resolution,
            "descriptor_dim": dimensions,
            "parameter_count": count,
        }
        value["initialization"]["rule"] = (
            "registered seed is reset immediately before model construction; real-train C4/finite sanity uses "
            "forked, separate RNG states and cannot affect formal initialization"
        )
        return value

    def atomic_write(path: Path, value: dict[str, Any]) -> None:
        if isinstance(value, dict):
            if Path(path).name == "run_manifest.json":
                value["format"] = "c170-temporal-gram-v19-run"
                value.setdefault("metadata", {})["v19_architecture"] = {
                    "name": architecture,
                    "temporal_resolution": resolution,
                    "descriptor_dim": dimensions,
                    "parameter_count": count,
                }
            elif Path(path).name == "RUN_COMPLETE.json":
                value["format"] = "c170-temporal-gram-v19-complete"
                value["v19_architecture"] = {"name": architecture, "temporal_resolution": resolution}
        old_write(path, value)

    class V19Classifier(C170TemporalGramClassifierV19):
        def __init__(self, variant: str, n_classes: int = 4) -> None:
            super().__init__(variant, architecture, resolution, n_classes)

    v11._variant_config = variant_config
    v11._collect_code_hashes = collect_hashes
    v11._manifest_base = manifest_base
    v11.v2._canonical_json_sha256 = canonical_hash
    v11.v2._atomic_write_json = atomic_write
    v11.C170ComplexOrbitClassifierV11 = V19Classifier


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def real_train_sanity(dataset: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the v19 C4/finite sanity check")
    device = torch.device("cuda:0")
    prepared = v11._validate_and_prepare_v11(
        SimpleNamespace(dataset=dataset, confirm_physical_order=True, variant="complex_gram", seed=17), device
    )
    tensors = prepared[3]["tensors"]
    samples = tensors["train_inputs"][:4]
    labels = tensors["train_targets"][:4]
    if samples.shape[0] != 4 or samples.shape[1:] != (500, 4):
        raise RuntimeError(f"unexpected real training batch shape: {tuple(samples.shape)}")

    results: dict[str, Any] = {
        "format": "c170-temporal-gram-v19-real-train-sanity",
        "real_train_windows": int(samples.shape[0]),
        "device": str(device),
        "test_metrics_computed": False,
        "architectures": {},
    }
    index = 0
    for architecture in ARCHITECTURES:
        for resolution in RESOLUTIONS:
            seed = 1900 + index
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                model = C170TemporalGramClassifierV19("complex_gram", architecture, resolution).to(device).eval()
                with torch.no_grad():
                    reference = model(samples)
                    errors = [
                        float((model(torch.roll(samples, shifts=shift, dims=-1)) - reference).abs().max())
                        for shift in (1, 2, 3)
                    ]
                model.train()
                loss = F.cross_entropy(model(samples), labels)
                loss.backward()
                finite = bool(torch.isfinite(loss)) and all(
                    parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                    for parameter in model.parameters()
                )
                if not bool(torch.isfinite(reference).all()) or not finite or max(errors) > 2e-5:
                    raise RuntimeError(
                        f"{architecture}/L{resolution} real-train C4/finite sanity failed: "
                        f"errors={errors}, finite={finite}"
                    )
                results["architectures"][f"{architecture}_L{resolution}"] = {
                    "descriptor_dim": DESCRIPTOR_DIMS[architecture],
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "c4_max_logit_errors": errors,
                    "finite_loss_and_gradients": finite,
                    "sanity_seed_isolated_from_formal_seed": seed,
                }
            index += 1
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--architecture", choices=ARCHITECTURES)
    parser.add_argument("--resolution", type=int, choices=RESOLUTIONS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--dataset", type=Path, default=v11.DEFAULT_DATASET)
    parser.add_argument("--max-epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--confirm-physical-order", action="store_true")
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--sanity-output", type=Path, default=DEFAULT_OUTPUT_ROOT / "real_train_c4_sanity.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sanity_only:
        result = real_train_sanity(args.dataset)
        _atomic_json(args.sanity_output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.architecture is None or args.resolution is None or args.seed is None or args.output_dir is None:
        raise ValueError("output-dir, architecture, resolution, and seed are required for training")
    if args.shuffle_labels and args.seed != 17:
        raise ValueError("the frozen shuffle control uses seed 17 only")
    if not args.confirm_physical_order:
        raise ValueError("pass --confirm-physical-order after confirming S0-S3")
    install_v19(args.architecture, args.resolution)
    variant = "complex_gram_shuffle" if args.shuffle_labels else "complex_gram"
    return v11.main([
        "--output-dir", str(args.output_dir), "--variant", variant, "--seed", str(args.seed),
        "--dataset", str(args.dataset), "--max-epochs", str(args.max_epochs), "--patience", str(args.patience),
        "--cpu-threads", str(args.cpu_threads), "--device", args.device, "--confirm-physical-order",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
