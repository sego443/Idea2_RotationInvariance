"""Focused runner contracts for the frozen C170 harmonic-phase v4 study."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from run_c170_harmonic_phase_experiment_v4 import (
    EXPECTED_SHARED_SCALE,
    _batches_for_epoch,
    _load_numpy_state_dict,
    _parser,
    _save_numpy_state_dict,
    _variant_config,
)
from src.c170_harmonic_phase_model_v4 import HarmonicPhaseClassifierV4


class HarmonicPhaseRunnerContractTests(unittest.TestCase):
    def test_cli_and_variants_freeze_four_arms_and_shared_scale(self) -> None:
        args = _parser().parse_args([
            "--output-dir", "run",
            "--variant", "harmonic_gram",
            "--seed", "17",
            "--confirm-physical-order",
        ])
        self.assertEqual(args.max_epochs, 160)
        self.assertEqual(args.patience, 25)
        self.assertEqual(args.cpu_threads, 2)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(EXPECTED_SHARED_SCALE, 11123.037109375)
        self.assertEqual(_variant_config("harmonic_gram")["active_feature_indices"], list(range(10)))
        self.assertEqual(_variant_config("harmonic_power")["active_feature_indices"], [0, 1])
        self.assertEqual(_variant_config("harmonic_phase")["active_feature_indices"], list(range(12)))
        self.assertEqual(_variant_config("harmonic_shuffle")["shuffle_labels"], True)

    def test_runner_sampler_uses_879_unique_examples_in_20_batches(self) -> None:
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

    def test_safe_npz_roundtrip_restores_model_outputs_without_pickle(self) -> None:
        torch.manual_seed(17)
        model = HarmonicPhaseClassifierV4("harmonic_gram").eval()
        samples = torch.randn(2, 500, 4)
        expected = model(samples).detach()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best_weights.npz"
            _save_numpy_state_dict(path, model.state_dict())
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(archive.files)
                self.assertTrue(all(not archive[name].dtype.hasobject for name in archive.files))
            restored = _load_numpy_state_dict(path, model.state_dict(), torch.device("cpu"))
            model.load_state_dict(restored, strict=True)
        self.assertTrue(torch.equal(expected, model(samples)))


if __name__ == "__main__":
    unittest.main()
