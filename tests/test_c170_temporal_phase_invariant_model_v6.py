"""Focused tests for the causal temporal phase-invariant v6 model."""

from __future__ import annotations

import math
import unittest

import torch

from src.c170_temporal_phase_invariant_model_v6 import (
    ACTIVE_FEATURE_INDICES,
    FEATURE_DIM,
    LAGS,
    TemporalPhaseInvariantClassifierV6,
    extract_temporal_phase_features,
)


def _reconstruct(s: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack((s + a, s + b, s - a, s - b), dim=-1)


class TemporalPhaseFeatureTests(unittest.TestCase):
    def test_initial_delta_filters_match_direct_causal_lags(self) -> None:
        torch.manual_seed(11)
        model = TemporalPhaseInvariantClassifierV6("learned_gram").eval()
        a = torch.randn(2, 500)
        b = torch.randn(2, 500)
        filtered_a, filtered_b = model.filter_harmonics(a, b)
        self.assertEqual(tuple(filtered_a.shape), (2, 8, 500))
        self.assertEqual(tuple(filtered_b.shape), (2, 8, 500))
        self.assertTrue(torch.equal(filtered_a[:, 0], a))
        self.assertTrue(torch.equal(filtered_b[:, 0], b))
        for channel, lag in enumerate(LAGS, start=1):
            direct_a = torch.zeros_like(a)
            direct_b = torch.zeros_like(b)
            direct_a[:, lag:] = a[:, :-lag]
            direct_b[:, lag:] = b[:, :-lag]
            self.assertTrue(torch.equal(filtered_a[:, channel], direct_a), f"a lag={lag}")
            self.assertTrue(torch.equal(filtered_b[:, channel], direct_b), f"b lag={lag}")

    def test_fir_frontend_is_causal_at_each_time_index(self) -> None:
        torch.manual_seed(19)
        model = TemporalPhaseInvariantClassifierV6("learned_gram").eval()
        a = torch.randn(2, 500)
        b = torch.randn(2, 500)
        changed_a, changed_b = a.clone(), b.clone()
        cutoff = 173
        changed_a[:, cutoff:] += torch.randn_like(changed_a[:, cutoff:]) * 5.0
        changed_b[:, cutoff:] += torch.randn_like(changed_b[:, cutoff:]) * 5.0
        base_a, base_b = model.filter_harmonics(a, b)
        future_a, future_b = model.filter_harmonics(changed_a, changed_b)
        self.assertTrue(torch.equal(base_a[..., :cutoff], future_a[..., :cutoff]))
        self.assertTrue(torch.equal(base_b[..., :cutoff], future_b[..., :cutoff]))

    def test_feature_tensor_is_invariant_to_c4_and_ideal_global_phase(self) -> None:
        torch.manual_seed(29)
        s, a, b = (torch.randn(2, 500) for _ in range(3))
        raw = _reconstruct(s, a, b)
        angle = 1.13
        rotated = _reconstruct(
            s,
            a * math.cos(angle) - b * math.sin(angle),
            a * math.sin(angle) + b * math.cos(angle),
        )
        model = TemporalPhaseInvariantClassifierV6("learned_gram").eval()
        kernels = model.normalized_fir_kernels()
        base = extract_temporal_phase_features(raw, kernels)
        phase_rotated = extract_temporal_phase_features(rotated, kernels)
        self.assertEqual(tuple(base.shape), (2, FEATURE_DIM, 500))
        self.assertTrue(torch.allclose(base, phase_rotated, atol=3e-6, rtol=0.0))
        for shift in (1, 2, 3):
            rolled = extract_temporal_phase_features(torch.roll(raw, shifts=shift, dims=2), kernels)
            self.assertTrue(torch.allclose(base, rolled, atol=3e-6, rtol=0.0), f"shift={shift}")

    def test_fixed_and_learned_filter_gradients_and_parameter_counts(self) -> None:
        learned = TemporalPhaseInvariantClassifierV6("learned_gram")
        fixed = TemporalPhaseInvariantClassifierV6("fixed_gram")
        self.assertTrue(learned.learned_fir_weights.requires_grad)
        self.assertFalse(fixed.learned_fir_weights.requires_grad)
        self.assertNotIn("identity_kernel", dict(learned.named_parameters()))
        self.assertIn("identity_kernel", dict(learned.named_buffers()))
        self.assertEqual(sum(p.numel() for p in learned.parameters()), 8763)
        self.assertEqual(sum(p.numel() for p in learned.parameters() if p.requires_grad), 8763)
        self.assertEqual(sum(p.numel() for p in fixed.parameters()), 8763)
        self.assertEqual(sum(p.numel() for p in fixed.parameters() if p.requires_grad), 8308)

        torch.manual_seed(37)
        learned = TemporalPhaseInvariantClassifierV6("learned_gram")
        logits = learned(torch.randn(2, 500, 4))
        logits.sum().backward()
        self.assertIsNotNone(learned.learned_fir_weights.grad)
        self.assertGreater(float(learned.learned_fir_weights.grad.abs().sum()), 0.0)

    def test_variant_masks_and_same_backend_initialization(self) -> None:
        self.assertEqual(ACTIVE_FEATURE_INDICES["learned_gram"], tuple(range(23)))
        self.assertEqual(ACTIVE_FEATURE_INDICES["fixed_gram"], tuple(range(23)))
        self.assertEqual(ACTIVE_FEATURE_INDICES["learned_power"], tuple(range(9)))
        self.assertEqual(ACTIVE_FEATURE_INDICES["learned_shuffle"], tuple(range(23)))
        torch.manual_seed(41)
        reference = TemporalPhaseInvariantClassifierV6("learned_gram")
        ref = dict(reference.named_parameters())
        for variant in ("fixed_gram", "learned_power", "learned_shuffle"):
            torch.manual_seed(41)
            model = TemporalPhaseInvariantClassifierV6(variant)
            values = dict(model.named_parameters())
            self.assertEqual(set(values), set(ref))
            for name, tensor in values.items():
                self.assertTrue(torch.equal(tensor, ref[name]), name)

        reference.eval()
        logits, embedding = reference(torch.randn(2, 500, 4), return_embedding=True)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(tuple(embedding.shape), (2, 192))
        self.assertTrue(torch.allclose(torch.linalg.vector_norm(embedding, dim=1), torch.ones(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
