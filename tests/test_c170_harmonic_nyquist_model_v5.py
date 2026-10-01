"""Focused formula and symmetry tests for the fixed v5 Nyquist features."""

from __future__ import annotations

import math
import unittest

import torch

from src.c170_harmonic_phase_model_v4 import extract_harmonic_features
from src.c170_harmonic_nyquist_model_v5 import (
    ACTIVE_FEATURE_INDICES,
    FEATURE_DIM,
    HarmonicNyquistClassifierV5,
    extract_nyquist_features,
    select_variant_features,
)


def _reconstruct(s: torch.Tensor, a: torch.Tensor, b: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return torch.stack((s + a + n, s + b - n, s - a + n, s - b - n), dim=-1)


class NyquistFeatureContractTests(unittest.TestCase):
    def test_feature_order_formula_and_lag_zero_prefix(self) -> None:
        torch.manual_seed(13)
        samples = torch.randn(2, 500, 4)
        actual = extract_nyquist_features(samples)
        prior = extract_harmonic_features(samples)
        self.assertEqual(tuple(actual.shape), (2, FEATURE_DIM, 500))
        self.assertTrue(torch.equal(actual[:, :10], prior[:, :10]))

        x0, x1, x2, x3 = samples.unbind(dim=-1)
        n = (x0 - x1 + x2 - x3) * 0.25
        a = (x0 - x2) * 0.5
        b = (x1 - x3) * 0.5
        power = a.square() + b.square()
        denominator = torch.sqrt(1.0 + n.square()) * (1.0 + power)
        self.assertTrue(torch.allclose(actual[:, 10], torch.log1p(n.square()), atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(actual[:, 15], n * (a.square() - b.square()) / denominator, atol=1e-7, rtol=0.0))
        self.assertTrue(torch.allclose(actual[:, 16], n * (2.0 * a * b) / denominator, atol=1e-7, rtol=0.0))

        for offset, lag in enumerate((1, 4, 16, 64), start=11):
            self.assertTrue(torch.equal(actual[:, offset, :lag], torch.zeros_like(actual[:, offset, :lag])))
            previous = torch.zeros_like(n)
            previous[:, lag:] = n[:, :-lag]
            expected = n * previous / torch.sqrt((1.0 + n.square()) * (1.0 + previous.square()))
            self.assertTrue(torch.allclose(actual[:, offset], expected, atol=1e-7, rtol=0.0))

    def test_all_feature_channels_are_invariant_to_c4_channel_rolls(self) -> None:
        torch.manual_seed(17)
        samples = torch.randn(2, 500, 4)
        base = extract_nyquist_features(samples)
        for shift in (1, 2, 3):
            rolled = extract_nyquist_features(torch.roll(samples, shifts=shift, dims=2))
            self.assertTrue(torch.allclose(base, rolled, atol=3e-6, rtol=0.0), f"shift={shift}")

    def test_coupling_is_a_negative_control_for_common_z_phase_with_fixed_n(self) -> None:
        torch.manual_seed(23)
        s, a, b, n = (torch.randn(2, 500) for _ in range(4))
        angle = 0.37
        rotated_a = a * math.cos(angle) - b * math.sin(angle)
        rotated_b = a * math.sin(angle) + b * math.cos(angle)
        original = extract_nyquist_features(_reconstruct(s, a, b, n))
        rotated = extract_nyquist_features(_reconstruct(s, rotated_a, rotated_b, n))

        # Existing z-only features and all uncoupled Nyquist features are U(1)-invariant.
        self.assertTrue(torch.allclose(original[:, :15], rotated[:, :15], atol=3e-6, rtol=0.0))
        self.assertTrue(torch.allclose(
            select_variant_features(original, "nyquist_uncoupled"),
            select_variant_features(rotated, "nyquist_uncoupled"),
            atol=3e-6,
            rtol=0.0,
        ))
        # n*z^2 deliberately retains the doubled phase and is not U(1)-invariant.
        self.assertFalse(torch.allclose(original[:, 15:17], rotated[:, 15:17], atol=1e-4, rtol=0.0))
        self.assertFalse(torch.allclose(
            select_variant_features(original, "nyquist_coupled"),
            select_variant_features(rotated, "nyquist_coupled"),
            atol=1e-4,
            rtol=0.0,
        ))

    def test_variant_masks_parameterization_and_output_contract(self) -> None:
        self.assertEqual(ACTIVE_FEATURE_INDICES["nyquist_coupled"], tuple(range(17)))
        self.assertEqual(ACTIVE_FEATURE_INDICES["nyquist_uncoupled"], tuple(range(15)))
        self.assertEqual(ACTIVE_FEATURE_INDICES["gram_control"], tuple(range(10)))
        self.assertEqual(ACTIVE_FEATURE_INDICES["nyquist_shuffle"], tuple(range(17)))

        torch.manual_seed(31)
        reference = HarmonicNyquistClassifierV5("nyquist_coupled").eval()
        reference_parameters = dict(reference.named_parameters())
        self.assertEqual(sum(parameter.numel() for parameter in reference.parameters()), 7012)
        for variant in ("nyquist_uncoupled", "gram_control", "nyquist_shuffle"):
            torch.manual_seed(31)
            model = HarmonicNyquistClassifierV5(variant).eval()
            self.assertEqual(set(dict(model.named_parameters())), set(reference_parameters))
            for name, parameter in model.named_parameters():
                self.assertTrue(torch.equal(parameter, reference_parameters[name]), name)

        samples = torch.randn(2, 500, 4)
        logits = reference(samples)
        logits_with_embedding, embedding = reference(samples, return_embedding=True)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.equal(logits, logits_with_embedding))
        self.assertEqual(tuple(embedding.shape), (2, 192))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(embedding, dim=1), torch.ones(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
