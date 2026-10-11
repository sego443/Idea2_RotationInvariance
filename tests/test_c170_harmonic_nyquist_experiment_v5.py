"""Runner integration and safe checkpoint contracts for v5."""

from __future__ import annotations

import datetime
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.runners.run_c170_harmonic_nyquist_experiment_v5 import (
    EXPECTED_SHARED_SCALE,
    _batches_for_epoch,
    _load_numpy_state_dict,
    _manifest_base,
    _parser,
    _save_numpy_state_dict,
    _variant_config,
)
from src.c170_harmonic_nyquist_model_v5 import HarmonicNyquistClassifierV5


class NyquistRunnerContractTests(unittest.TestCase):
    def test_cli_variant_configs_and_manifest(self) -> None:
        args = _parser().parse_args([
            "--output-dir", "run",
            "--variant", "nyquist_coupled",
            "--seed", "17",
            "--confirm-physical-order",
        ])
        self.assertEqual(args.max_epochs, 160)
        self.assertEqual(args.patience, 25)
        self.assertEqual(args.cpu_threads, 2)
        self.assertEqual(args.device, "cuda")
        self.assertEqual(EXPECTED_SHARED_SCALE, 11123.037109375)
        self.assertEqual(_variant_config("nyquist_coupled")["active_feature_indices"], list(range(17)))
        self.assertEqual(_variant_config("nyquist_uncoupled")["active_feature_indices"], list(range(15)))
        self.assertEqual(_variant_config("gram_control")["active_feature_indices"], list(range(10)))
        self.assertTrue(_variant_config("nyquist_shuffle")["shuffle_labels"])
        manifest = _manifest_base(
            {"variant": "nyquist_coupled"},
            "unit-config-sha",
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            args,
            torch.device("cpu"),
        )
        self.assertEqual(manifest["format"], "c170-harmonic-nyquist-v5-run")
        self.assertFalse(manifest["test_metrics_computed"])
        self.assertIn("model", manifest["provenance"]["code_sha256"])

    def test_exact_sampler_and_safe_npz_forward_roundtrip(self) -> None:
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

        torch.manual_seed(17)
        model = HarmonicNyquistClassifierV5("nyquist_coupled").eval()
        samples = torch.randn(2, 500, 4)
        expected = model(samples)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "best_weights.npz"
            _save_numpy_state_dict(path, model.state_dict())
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(archive.files)
                self.assertTrue(all(not archive[name].dtype.hasobject for name in archive.files))
            state = _load_numpy_state_dict(path, model.state_dict(), torch.device("cpu"))
            model.load_state_dict(state, strict=True)
        self.assertTrue(torch.equal(expected, model(samples)))


if __name__ == "__main__":
    unittest.main()
