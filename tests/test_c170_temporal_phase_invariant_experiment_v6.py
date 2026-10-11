"""Focused runner and safe state-dict tests for v6."""

from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.runners.run_c170_temporal_phase_invariant_experiment_v6 import (
    EXPECTED_SHARED_SCALE,
    _batches_for_epoch,
    _load_numpy_state_dict,
    _manifest_base,
    _parser,
    _save_numpy_state_dict,
    _variant_config,
)
from src.c170_temporal_phase_invariant_model_v6 import TemporalPhaseInvariantClassifierV6


class TemporalPhaseInvariantRunnerTests(unittest.TestCase):
    def test_cli_variants_masks_and_manifest_contract(self) -> None:
        args = _parser().parse_args([
            "--output-dir", "run",
            "--variant", "learned_gram",
            "--seed", "17",
            "--confirm-physical-order",
        ])
        self.assertEqual(args.max_epochs, 160)
        self.assertEqual(args.patience, 25)
        self.assertEqual(args.cpu_threads, 2)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(EXPECTED_SHARED_SCALE, 11123.037109375)
        self.assertEqual(_variant_config("learned_gram")["active_feature_indices"], list(range(23)))
        self.assertEqual(_variant_config("fixed_gram")["active_feature_indices"], list(range(23)))
        self.assertEqual(_variant_config("learned_power")["active_feature_indices"], list(range(9)))
        self.assertTrue(_variant_config("learned_shuffle")["shuffle_labels"])
        manifest = _manifest_base(
            {"variant": "learned_gram"},
            "unit-config-sha",
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            args,
            torch.device("cpu"),
        )
        self.assertEqual(manifest["format"], "c170-temporal-phase-invariant-v6-run")
        self.assertFalse(manifest["test_metrics_computed"])
        self.assertIn("model", manifest["provenance"]["code_sha256"])

    def test_sampler_and_safe_npz_roundtrip_keep_fixed_buffers_strictly(self) -> None:
        groups = {}
        cursor = 0
        for distance in range(11):
            for action in range(4):
                count = 19 if (distance, action) == (10, 2) else 20
                groups[(distance, action)] = np.arange(cursor, cursor + count, dtype=np.int64)
                cursor += count
        batches = _batches_for_epoch(groups, seed=29, epoch=6)
        flattened = np.concatenate(batches)
        self.assertEqual(len(batches), 20)
        self.assertEqual([len(batch) for batch in batches], [44] * 19 + [43])
        self.assertEqual(len(flattened), 879)
        self.assertEqual(len(np.unique(flattened)), 879)

        for variant in ("learned_gram", "fixed_gram", "learned_power", "learned_shuffle"):
            torch.manual_seed(17)
            model = TemporalPhaseInvariantClassifierV6(variant).eval()
            samples = torch.randn(2, 500, 4)
            expected = model(samples)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "best_weights.npz"
                _save_numpy_state_dict(path, model.state_dict())
                with np.load(path, allow_pickle=False) as archive:
                    self.assertIn("identity_kernel", archive.files)
                    self.assertTrue(all(not archive[name].dtype.hasobject for name in archive.files))
                restored = _load_numpy_state_dict(path, model.state_dict(), torch.device("cpu"))
                model.load_state_dict(restored, strict=True)
            self.assertTrue(torch.equal(expected, model(samples)), variant)


if __name__ == "__main__":
    unittest.main()
