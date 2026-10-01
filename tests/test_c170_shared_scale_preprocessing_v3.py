import unittest

import numpy as np

from src.c170_shared_scale_preprocessing_v3 import (
    calibrate_windows_shared_scale_v3,
    prepare_model_inputs_shared_scale_v3,
)


class SharedScaleCalibrationTests(unittest.TestCase):
    def test_one_scalar_is_median_over_all_train_samples_times_and_channels(self):
        filtered = np.array(
            [[[0, 0, 0, 0], [2, 4, 6, 8]]],
            dtype=np.float32,
        )

        normalized, scale = calibrate_windows_shared_scale_v3(filtered)

        self.assertEqual(scale, 2.5)
        np.testing.assert_array_equal(
            normalized,
            np.array([[[-0.4, -0.8, -1.2, -1.6], [0.4, 0.8, 1.2, 1.6]]], dtype=np.float32),
        )

    def test_zero_train_signal_uses_float32_epsilon(self):
        normalized, scale = calibrate_windows_shared_scale_v3(
            np.zeros((2, 8, 4), dtype=np.float32)
        )

        self.assertEqual(scale, float(np.finfo(np.float32).eps))
        self.assertTrue(np.all(normalized == 0.0))

    def test_explicit_validation_scale_must_be_one_finite_positive_scalar(self):
        values = np.arange(2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)

        for bad_scale in (np.ones(4, dtype=np.float32), 0.0, float("inf"), float("nan")):
            with self.subTest(scale=bad_scale):
                with self.assertRaises(ValueError):
                    calibrate_windows_shared_scale_v3(values, scale=bad_scale)

    def test_validation_values_do_not_change_train_scale_or_train_inputs(self):
        rng = np.random.default_rng(29)
        windows = rng.normal(size=(3, 500, 4)).astype(np.float32)
        split = {"train": np.array([0, 1], dtype=np.int64), "val": np.array([2], dtype=np.int64)}
        changed_validation = windows.copy()
        changed_validation[2] *= np.float32(1_000_000.0)

        first, first_scale = prepare_model_inputs_shared_scale_v3(windows, split)
        second, second_scale = prepare_model_inputs_shared_scale_v3(changed_validation, split)

        self.assertEqual(first_scale, second_scale)
        np.testing.assert_array_equal(first["train"], second["train"])

    def test_filter_demean_and_shared_scale_commute_with_all_four_cyclic_rolls(self):
        rng = np.random.default_rng(20260929)
        signal = rng.normal(size=(3, 500, 4)).astype(np.float32)
        signal *= np.array([0.3, 1.0, 2.5, 7.0], dtype=np.float32)[None, None, :]
        signal += np.array([4.0, -2.0, 0.25, 11.0], dtype=np.float32)[None, None, :]
        split = {"train": np.array([0, 1], dtype=np.int64), "val": np.array([2], dtype=np.int64)}
        reference, reference_scale = prepare_model_inputs_shared_scale_v3(signal, split)

        for shift in range(4):
            rotated, rotated_scale = prepare_model_inputs_shared_scale_v3(
                np.roll(signal, shift=shift, axis=2), split
            )
            self.assertEqual(reference_scale, rotated_scale)
            for name in ("train", "val"):
                np.testing.assert_allclose(
                    rotated[name],
                    np.roll(reference[name], shift=shift, axis=2),
                    rtol=2e-5,
                    atol=2e-5,
                )


if __name__ == "__main__":
    unittest.main()
