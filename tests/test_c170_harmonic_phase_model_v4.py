"""Focused numerical contract tests for the fixed harmonic-phase v4 model."""

from __future__ import annotations

import math
import unittest

import torch

from src.c170_harmonic_phase_model_v4 import (
    FEATURE_DIM,
    HarmonicPhaseClassifierV4,
    extract_harmonic_features,
    select_variant_features,
)


def _from_sab(s: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack((s + a, s + b, s - a, s - b), dim=-1)


class HarmonicFeatureContractTests(unittest.TestCase):
    def test_selected_lags_match_fixed_complex_product_and_zero_prefix(self) -> None:
        a = torch.tensor([[1.0, 0.0, -1.0, 0.0, 1.0]])
        b = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0]])
        samples = _from_sab(torch.zeros_like(a), a, b)
        features = extract_harmonic_features(samples)

        self.assertEqual(tuple(features.shape), (1, FEATURE_DIM, 5))
        # Feature channels 2/3 are lag-1 Re/Im; 4/5 are lag-4 Re/Im.
        self.assertTrue(torch.equal(features[0, 2, :1], torch.zeros(1)))
        self.assertTrue(torch.equal(features[0, 3, :1], torch.zeros(1)))
        self.assertAlmostEqual(float(features[0, 3, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(features[0, 3, 2]), 0.5, places=6)
        self.assertTrue(torch.equal(features[0, 4, :4], torch.zeros(4)))
        self.assertAlmostEqual(float(features[0, 4, 4]), 0.5, places=6)
        self.assertTrue(torch.allclose(features[0, 1], torch.full((5,), math.log(2.0))))
        self.assertAlmostEqual(float(features[0, 10, 0]), math.asinh(1.0), places=6)

    def test_gram_and_power_features_ignore_common_complex_phase_but_phase_variant_retains_it(self) -> None:
        torch.manual_seed(23)
        s = torch.randn(2, 500)
        a = torch.randn(2, 500)
        b = torch.randn(2, 500)
        angle = 0.73
        rotated_a = a * math.cos(angle) - b * math.sin(angle)
        rotated_b = a * math.sin(angle) + b * math.cos(angle)
        original = extract_harmonic_features(_from_sab(s, a, b))
        rotated = extract_harmonic_features(_from_sab(s, rotated_a, rotated_b))

        for variant in ("harmonic_gram", "harmonic_power"):
            self.assertTrue(torch.allclose(
                select_variant_features(original, variant),
                select_variant_features(rotated, variant),
                atol=2e-6,
                rtol=0.0,
            ))
        self.assertFalse(torch.allclose(
            select_variant_features(original, "harmonic_phase"),
            select_variant_features(rotated, "harmonic_phase"),
            atol=1e-4,
            rtol=0.0,
        ))

    def test_gram_features_are_invariant_to_a_cyclic_channel_roll(self) -> None:
        torch.manual_seed(41)
        samples = torch.randn(2, 500, 4)
        original = select_variant_features(extract_harmonic_features(samples), "harmonic_gram")
        rolled = select_variant_features(
            extract_harmonic_features(torch.roll(samples, shifts=1, dims=2)),
            "harmonic_gram",
        )
        self.assertTrue(torch.allclose(original, rolled, atol=2e-6, rtol=0.0))

    def test_all_variants_share_parameterization_and_return_logits_with_optional_embedding(self) -> None:
        torch.manual_seed(7)
        reference = HarmonicPhaseClassifierV4("harmonic_gram")
        reference_parameters = dict(reference.named_parameters())
        for variant in ("harmonic_power", "harmonic_phase", "harmonic_shuffle"):
            torch.manual_seed(7)
            model = HarmonicPhaseClassifierV4(variant)
            self.assertEqual(set(dict(model.named_parameters())), set(reference_parameters))
            for name, parameter in model.named_parameters():
                self.assertTrue(torch.equal(parameter, reference_parameters[name]), name)

        samples = torch.randn(3, 500, 4)
        reference.eval()
        logits = reference(samples)
        logits_with_embedding, embedding = reference(samples, return_embedding=True)
        self.assertEqual(tuple(logits.shape), (3, 4))
        self.assertTrue(torch.equal(logits, logits_with_embedding))
        self.assertEqual(tuple(embedding.shape), (3, 192))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(embedding, dim=1), torch.ones(3), atol=1e-6))
        self.assertEqual(sum(parameter.numel() for parameter in reference.parameters()), 5932)


if __name__ == "__main__":
    unittest.main()
