"""Thin adapter for F2 matched-null runs; frozen model runners remain untouched."""

from __future__ import annotations

import argparse
import copy
import importlib
import sys
from pathlib import Path
from typing import Sequence

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

REPS = (17, 29, 43, 61, 89)


def _module_for_variant(variant: str):
    if variant == "complex_gram_shuffle":
        return importlib.import_module("run_c170_complex_orbit_experiment_v11")
    if variant == "learned_power_shuffle":
        return importlib.import_module("run_c170_temporal_phase_invariant_experiment_v6_power_shuffle")
    raise ValueError(f"unsupported matched-null variant: {variant}")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        variant_index = argv.index("--variant")
        variant = argv[variant_index + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit("--variant is required") from error

    module = _module_for_variant(variant)
    original_parser = module._parser

    def parser_with_shuffle_rep():
        parser = original_parser()
        parser.add_argument("--shuffle-seed", type=int, choices=REPS, required=True)
        return parser

    module._parser = parser_with_shuffle_rep
    args = module._parser().parse_args(argv)
    if args.shuffle_seed not in REPS:
        raise ValueError("shuffle seed must be one of the five frozen null reps")

    validate_name = "_validate_and_prepare_v11" if variant == "complex_gram_shuffle" else "_validate_and_prepare_v6"
    original_validate = getattr(module, validate_name)

    def validate_with_shared_shuffle_rep(run_args, device):
        prep_args = copy.copy(run_args)
        # Keep run_args.seed for model initialization and the frozen batch sampler.
        # Only the shared pseudo-label map uses the null replicate seed.
        prep_args.seed = int(run_args.shuffle_seed)
        result = original_validate(prep_args, device)
        result[2]["matched_null_shuffle_rep"] = int(run_args.shuffle_seed)
        result[2]["matched_null_shuffle_seed"] = int(run_args.shuffle_seed) + int(module.v2.LABEL_SHUFFLE_SEED_OFFSET)
        return result

    setattr(module, validate_name, validate_with_shared_shuffle_rep)

    original_config_hash = module.v2._canonical_json_sha256

    def config_hash_with_shuffle_rep(config):
        if config.get("shuffle_labels"):
            config["matched_null_shuffle_rep"] = int(args.shuffle_seed)
            config["matched_null_shuffle_seed"] = int(args.shuffle_seed) + int(module.v2.LABEL_SHUFFLE_SEED_OFFSET)
        return original_config_hash(config)

    module.v2._canonical_json_sha256 = config_hash_with_shuffle_rep

    original_hashes = module._collect_code_hashes

    def hashes_with_adapter():
        result = original_hashes()
        result["matched_null_adapter"] = module.v2._file_sha256(Path(__file__).resolve())
        return result

    module._collect_code_hashes = hashes_with_adapter
    return module._run(args)


if __name__ == "__main__":
    raise SystemExit(main())
