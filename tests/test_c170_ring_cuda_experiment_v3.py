import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from run_c170_ring_cuda_experiment_v3 import (
    VARIANT_CONFIG,
    _load_numpy_state_dict,
    _manifest_base_v3,
    _parser,
    _save_numpy_state_dict,
)
from src.c170_shared_scale_preprocessing_v3 import prepare_model_inputs_shared_scale_v3
from src.c170_ring_representation import RingRepresentationClassifier


def _measure_synthetic_cuda_equivariance_errors():
    rng = np.random.default_rng(20260929)
    raw = rng.normal(size=(8, 500, 4)).astype(np.float32)
    raw *= np.array([0.25, 0.8, 2.0, 6.0], dtype=np.float32)[None, None, :]
    raw += np.array([3.0, -5.0, 0.5, 9.0], dtype=np.float32)[None, None, :]
    splits = {
        "train": np.arange(6, dtype=np.int64),
        "val": np.arange(6, 8, dtype=np.int64),
    }
    reference, reference_scale = prepare_model_inputs_shared_scale_v3(raw, splits)

    torch.manual_seed(2909)
    model = RingRepresentationClassifier(variant="classification_only", n_classes=4)
    model = model.to(torch.device("cuda:0")).eval()
    reference_inputs = torch.as_tensor(reference["val"], device="cuda:0")
    with torch.no_grad():
        expected_logits, expected_projection = model.forward_with_projection(reference_inputs)

    errors = {"preprocessing": 0.0, "logits": 0.0, "projection": 0.0}
    with torch.no_grad():
        for shift in range(4):
            rotated, rotated_scale = prepare_model_inputs_shared_scale_v3(
                np.roll(raw, shift=shift, axis=2), splits
            )
            if reference_scale != rotated_scale:
                raise AssertionError("shared train scale changed under a cyclic channel roll")
            expected_rotated = np.roll(reference["val"], shift=shift, axis=2)
            errors["preprocessing"] = max(
                errors["preprocessing"],
                float(np.max(np.abs(rotated["val"] - expected_rotated))),
            )
            inputs = torch.as_tensor(rotated["val"], device="cuda:0")
            logits, projection = model.forward_with_projection(inputs)
            errors["logits"] = max(
                errors["logits"],
                float(torch.max(torch.abs(logits - expected_logits)).cpu()),
            )
            errors["projection"] = max(
                errors["projection"],
                float(torch.max(torch.abs(projection - expected_projection)).cpu()),
            )
    return errors


class V3RunnerContractTests(unittest.TestCase):
    def test_manifest_contract_is_constructible_without_external_checkpoint_loader(self):
        args = _parser().parse_args(
            [
                "--output-dir",
                "run",
                "--variant",
                "shared_scale_ce",
                "--seed",
                "17",
            ]
        )

        manifest = _manifest_base_v3({}, "config-sha", "2026-09-29T00:00:00Z", args, torch.device("cpu"))

        self.assertEqual(manifest["format"], "c170-ring-cuda-v3-run")
        self.assertEqual(manifest["metadata"]["variant"], "shared_scale_ce")
        self.assertIn("shared_scale_preprocessing", manifest["provenance"]["code_sha256"])

    def test_only_fixed_ce_variants_use_classification_only_without_supcon(self):
        self.assertEqual(
            VARIANT_CONFIG,
            {
                "shared_scale_ce": {
                    "model_variant": "classification_only",
                    "use_supcon": False,
                    "shuffle_labels": False,
                },
                "shared_scale_shuffle": {
                    "model_variant": "classification_only",
                    "use_supcon": False,
                    "shuffle_labels": True,
                },
            },
        )

    def test_cli_keeps_fixed_run_defaults_and_seed_set(self):
        args = _parser().parse_args(
            [
                "--output-dir",
                "run",
                "--variant",
                "shared_scale_ce",
                "--seed",
                "17",
                "--confirm-physical-order",
            ]
        )

        self.assertEqual(args.max_epochs, 160)
        self.assertEqual(args.patience, 25)
        self.assertEqual(args.cpu_threads, 2)
        self.assertEqual(args.device, "cuda")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            _parser().parse_args(["--output-dir", "run", "--variant", "ring_main", "--seed", "17"])

    def test_selected_tensor_weights_round_trip_without_pickle(self):
        state_dict = {
            "linear.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "linear.bias": torch.tensor([0.25, -0.5, 1.0], dtype=torch.float32),
            "counter": torch.tensor(7, dtype=torch.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_weights.npz"
            with mock.patch.object(torch, "load", side_effect=AssertionError("unsafe pickle load")):
                _save_numpy_state_dict(path, state_dict)
                with np.load(path, allow_pickle=False) as archive:
                    self.assertEqual(set(archive.files), set(state_dict))
                    self.assertTrue(all(archive[name].dtype.kind in "biuf" for name in archive.files))
                restored = _load_numpy_state_dict(path, state_dict, torch.device("cpu"))

        self.assertEqual(set(restored), set(state_dict))
        for name in state_dict:
            self.assertTrue(torch.equal(restored[name], state_dict[name]))

    def test_safe_state_loader_rejects_key_shape_dtype_and_object_mismatches(self):
        expected = {
            "weight": torch.zeros((2, 3), dtype=torch.float32),
            "counter": torch.tensor(4, dtype=torch.int64),
        }
        cases = (
            ({"weight": np.zeros((2, 3), dtype=np.float32)}, ValueError),
            ({"weight": np.zeros((3, 2), dtype=np.float32), "counter": np.array(4, dtype=np.int64)}, ValueError),
            ({"weight": np.zeros((2, 3), dtype=np.float64), "counter": np.array(4, dtype=np.int64)}, TypeError),
            ({"weight": np.zeros((2, 3), dtype=np.float32), "counter": np.array({"unsafe": True}, dtype=object)}, ValueError),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (arrays, error_type) in enumerate(cases):
                path = Path(directory) / f"invalid-{index}.npz"
                with path.open("wb") as stream:
                    np.savez(stream, **arrays)
                with self.subTest(case=index), self.assertRaises(error_type):
                    _load_numpy_state_dict(path, expected, torch.device("cpu"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the synthetic ring-roll check")
class SyntheticCudaEquivarianceTests(unittest.TestCase):
    def test_raw_to_ring_logits_and_projection_match_all_four_cyclic_rolls(self):
        errors = _measure_synthetic_cuda_equivariance_errors()
        self.assertLessEqual(errors["preprocessing"], 2e-5)
        self.assertLessEqual(errors["logits"], 1e-5)
        self.assertLessEqual(errors["projection"], 1e-5)


if __name__ == "__main__":
    unittest.main()
