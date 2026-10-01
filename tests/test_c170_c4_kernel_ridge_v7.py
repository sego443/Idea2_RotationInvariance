from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.c170_c4_kernel_ridge_v7 import (
    VIEWS,
    apply_fold_scaler,
    assemble_view,
    build_feature_families,
    fit_fold_scaler,
    kernel_matrix,
    preprocess_windows,
    safe_npz_load,
    safe_npz_save,
    weighted_krr_fit,
)


class C170C4KernelRidgeV7Tests(unittest.TestCase):
    def test_zero_windows_produce_finite_preprocessed_features_for_all_modes(self) -> None:
        zeros = np.zeros((2, 500, 4), dtype=np.float32)
        untouched = zeros.copy()
        for mode in ("bandpass_0p5_40", "demean_only", "lowpass_40"):
            processed = preprocess_windows(zeros, mode)
            families = build_feature_families(processed)
            for view in VIEWS:
                descriptor = assemble_view(families, view)
                self.assertTrue(np.isfinite(processed).all())
                self.assertTrue(np.isfinite(descriptor).all())
        np.testing.assert_array_equal(zeros, untouched)

    def test_view_dimensions_and_shared_fold_statistics(self) -> None:
        rng = np.random.default_rng(4)
        windows = rng.normal(size=(12, 500, 4)).astype(np.float32)
        features = build_feature_families(windows)
        expected_dims = {"ab": 72, "bcd": 14, "abcd": 78}
        for view, dimension in expected_dims.items():
            sensor_features = assemble_view(features, view)
            self.assertEqual(sensor_features.shape, (12, 4, dimension))
            scaler = fit_fold_scaler(sensor_features[:8], view)
            self.assertEqual(scaler["mean"].shape, (dimension,))
            self.assertEqual(scaler["std"].shape, (dimension,))
            normalized = apply_fold_scaler(sensor_features[8:], scaler)
            self.assertEqual(normalized.shape, (4, 4 * dimension))
            self.assertTrue(np.isfinite(normalized).all())
            normalized_by_sensor = normalized.reshape(4, 4, dimension)
            for shift in range(4):
                rolled = apply_fold_scaler(np.roll(sensor_features[8:], shift, axis=1), scaler)
                np.testing.assert_allclose(
                    rolled.reshape(4, 4, dimension),
                    np.roll(normalized_by_sensor, shift, axis=1),
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_group_kernel_is_symmetric_psd_and_c4_invariant(self) -> None:
        torch.manual_seed(5)
        train = torch.randn(9, 4, 7, dtype=torch.float64)
        query = torch.randn(4, 4, 7, dtype=torch.float64)
        gram = kernel_matrix(train, train, sigma=2.3, mode="c4_group")
        self.assertTrue(torch.allclose(gram, gram.T, atol=1e-12, rtol=0.0))
        min_eigenvalue = float(torch.linalg.eigvalsh(gram).min())
        self.assertGreaterEqual(min_eigenvalue, -1e-10)
        base = kernel_matrix(train, query, sigma=2.3, mode="c4_group")
        for shift in range(4):
            rolled_train = torch.roll(train, shifts=shift, dims=1)
            rolled_query = torch.roll(query, shifts=shift, dims=1)
            self.assertTrue(
                torch.allclose(
                    base,
                    kernel_matrix(rolled_train, query, sigma=2.3, mode="c4_group"),
                    atol=1e-11,
                    rtol=0.0,
                )
            )
            self.assertTrue(
                torch.allclose(
                    base,
                    kernel_matrix(train, rolled_query, sigma=2.3, mode="c4_group"),
                    atol=1e-11,
                    rtol=0.0,
                )
            )

    def test_plain_rbf_control_is_not_c4_invariant(self) -> None:
        train = torch.tensor([[[0.0], [1.0], [3.0], [7.0]]], dtype=torch.float64)
        query = torch.tensor([[[0.1], [1.2], [2.8], [7.1]]], dtype=torch.float64)
        base = kernel_matrix(train, query, sigma=1.7, mode="plain_rbf")
        rolled = kernel_matrix(train, torch.roll(query, shifts=1, dims=1), sigma=1.7, mode="plain_rbf")
        self.assertGreater(abs(float(base[0, 0] - rolled[0, 0])), 1e-3)

    def test_weighted_krr_solution_has_small_linear_residual(self) -> None:
        torch.manual_seed(8)
        descriptors = torch.randn(10, 4, 3, dtype=torch.float64)
        gram = kernel_matrix(descriptors, descriptors, sigma=1.4, mode="c4_group")
        targets = torch.nn.functional.one_hot(torch.arange(10) % 4, num_classes=4).to(torch.float64)
        weights = torch.linspace(0.5, 1.5, 10, dtype=torch.float64)
        coefficients, residual = weighted_krr_fit(gram, targets, weights, alpha=0.1)
        self.assertTrue(torch.isfinite(coefficients).all())
        self.assertLess(float(residual), 1e-10)

    def test_fold_scaler_cannot_depend_on_validation_values(self) -> None:
        rng = np.random.default_rng(9)
        train = rng.normal(size=(7, 4, 14))
        scaler_a = fit_fold_scaler(train, "bcd")
        val_a = rng.normal(size=(3, 4, 14))
        val_b = val_a + 1000.0
        out_a = apply_fold_scaler(val_a, scaler_a)
        out_b = apply_fold_scaler(val_b, scaler_a)
        self.assertFalse(np.allclose(out_a, out_b))
        np.testing.assert_array_equal(scaler_a["mean"], fit_fold_scaler(train, "bcd")["mean"])
        np.testing.assert_array_equal(scaler_a["std"], fit_fold_scaler(train, "bcd")["std"])

    def test_numeric_npz_roundtrip_disables_pickle(self) -> None:
        arrays = {
            "descriptor": np.arange(24, dtype=np.float64).reshape(3, 8),
            "coefficients": np.linspace(-1.0, 1.0, 12, dtype=np.float64).reshape(3, 4),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.npz"
            safe_npz_save(path, arrays)
            restored = safe_npz_load(path)
        self.assertEqual(set(restored), set(arrays))
        for name in arrays:
            np.testing.assert_array_equal(restored[name], arrays[name])


if __name__ == "__main__":
    unittest.main()
