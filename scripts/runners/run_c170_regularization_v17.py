"""Thin v17 wrapper for the frozen v11 complex_gram trainer."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys
import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from scripts.runners import run_c170_complex_orbit_experiment_v11 as base
WEIGHT_DECAY = None
DROPOUT = None

class V17ComplexGram(base.C170ComplexOrbitClassifierV11):
    def __init__(self, variant: str, n_classes: int = 4) -> None:
        super().__init__(variant, n_classes)
        if self.architecture != "complex_gram":
            raise ValueError(f"v17 supports complex_gram only, got {self.architecture}")
        self.core.dropout = nn.Dropout(p=float(DROPOUT))


def _patch(weight_decay: float, dropout: float) -> None:
    global WEIGHT_DECAY, DROPOUT
    WEIGHT_DECAY, DROPOUT = float(weight_decay), float(dropout)
    base.WEIGHT_DECAY = WEIGHT_DECAY
    old_variant_config = base._variant_config
    def variant_config(variant: str):
        value = old_variant_config(variant)
        if value["architecture"] != "complex_gram":
            raise ValueError("v17 requires the complex_gram backbone")
        suffix = f"wd{WEIGHT_DECAY:g}_dropout{DROPOUT:g}".replace(".", "p")
        value["model_variant"] = f"c170_complex_orbit_classifier_v17_complex_gram_{suffix}"
        value["regularization"] = {
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "weight_decay_applied_to": "AdamW",
            "dropout_applied_to": "672D complex_gram descriptor before classifier",
        }
        return value
    base._variant_config = variant_config
    base.C170ComplexOrbitClassifierV11 = V17ComplexGram

    old_hashes = base._collect_code_hashes
    def collect_hashes():
        result = old_hashes()
        result["v17_regularization_wrapper"] = base.v2._file_sha256(Path(__file__).resolve())
        return result
    base._collect_code_hashes = collect_hashes

    old_manifest_base = base._manifest_base
    def manifest_base(*args, **kwargs):
        value = old_manifest_base(*args, **kwargs)
        value["format"] = "c170-complex-regularization-v17-run"
        value["metadata"]["regularization"] = {"weight_decay": WEIGHT_DECAY, "dropout": DROPOUT}
        value["initialization"]["rule"] = "v11_seeded_backbone_initialization_then_parameter_free_dropout_override"
        return value
    base._manifest_base = manifest_base

    old_canonical = base.v2._canonical_json_sha256
    def canonical(value):
        if isinstance(value, dict) and value.get("variant") in base.VARIANTS and value.get("architecture", {}).get("name") == "complex_gram":
            value["architecture"]["dropout"] = DROPOUT
            value["v17_regularization"] = {
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "weight_decay_applied_to": "AdamW",
                "dropout_applied_to": "672D complex_gram descriptor before classifier",
            }
        return old_canonical(value)
    base.v2._canonical_json_sha256 = canonical

    old_write = base.v2._atomic_write_json
    def atomic_write_json(path, value):
        if isinstance(value, dict):
            if Path(path).name == "run_manifest.json":
                value["format"] = "c170-complex-regularization-v17-run"
                value.setdefault("metadata", {})["regularization"] = {"weight_decay": WEIGHT_DECAY, "dropout": DROPOUT}
            elif Path(path).name == "RUN_COMPLETE.json":
                value["format"] = "c170-complex-regularization-v17-complete"
                value["regularization"] = {"weight_decay": WEIGHT_DECAY, "dropout": DROPOUT}
        return old_write(path, value)
    base.v2._atomic_write_json = atomic_write_json


def sanity(weight_decay: float, dropout: float, device_name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for v17 sanity")
    _patch(weight_decay, dropout)
    model = V17ComplexGram("complex_gram").cuda()
    model.train()
    x = torch.randn(4, 500, 4, device="cuda")
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0,1,2,3], device="cuda"))
    loss.backward()
    if model.core.dropout.p != dropout or not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("v17 dropout/config/finite sanity failed")
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise RuntimeError("v17 non-finite gradient")
    print(f"V17_SANITY_OK wd={weight_decay} dropout={dropout} dropout_p={model.core.dropout.p} loss={float(loss.detach()):.6f} cuda={torch.cuda.get_device_name(0)}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--v17-weight-decay", type=float, required=True)
    parser.add_argument("--v17-dropout", type=float, required=True)
    parser.add_argument("--v17-sanity-only", action="store_true")
    options, remainder = parser.parse_known_args(argv)
    if options.v17_weight_decay not in (0.01, 0.1) or options.v17_dropout not in (0.2, 0.4, 0.6):
        raise ValueError("v17 fixed grid is wd={0.01,0.1}, dropout={0.2,0.4,0.6}")
    if options.v17_sanity_only:
        sanity(options.v17_weight_decay, options.v17_dropout, "cuda")
        return 0
    _patch(options.v17_weight_decay, options.v17_dropout)
    return base.main(remainder)

if __name__ == "__main__":
    raise SystemExit(main())