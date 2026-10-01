from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import unittest

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from c170_latent_angle import (  # noqa: E402
    LatentAngleMLPBaseline,
    LatentAngleTemporalClassifier,
)


def _waveforms(batch: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(41)
    return torch.randn(batch, 500, 4, generator=generator)


class LatentAngleModelTests(unittest.TestCase):
    def test_forward_returns_normalized_marginals_and_conditionals(self) -> None:
        model = LatentAngleTemporalClassifier().eval()
        outputs = model(_waveforms())

        self.assertEqual(outputs["scores"].shape, (3, 4, 64))
        self.assertEqual(outputs["joint_log_probs"].shape, (3, 4, 64))
        self.assertEqual(outputs["action_log_probs"].shape, (3, 4))
        self.assertEqual(outputs["angle_log_probs"].shape, (3, 64))
        self.assertEqual(outputs["conditional_angle_log_probs"].shape, (3, 4, 64))
        self.assertEqual(outputs["features"].shape, (3, 516))
        self.assertEqual(outputs["template_features"].shape, (4, 64, 516))
        self.assertEqual(outputs["log_rms"].shape, (3, 4))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))
        torch.testing.assert_close(torch.logsumexp(outputs["joint_log_probs"].flatten(1), dim=1), torch.zeros(3))
        torch.testing.assert_close(torch.logsumexp(outputs["action_log_probs"], dim=1), torch.zeros(3))
        torch.testing.assert_close(torch.logsumexp(outputs["angle_log_probs"], dim=1), torch.zeros(3))
        torch.testing.assert_close(
            torch.logsumexp(outputs["conditional_angle_log_probs"], dim=2),
            torch.zeros(3, 4),
        )


    def test_losses_have_finite_gradients_and_are_per_example(self) -> None:
        model = LatentAngleTemporalClassifier()
        waveforms = _waveforms(batch=4)
        labels = torch.tensor([0, 2, 1, 3])
        nominal_angles = torch.tensor([0.0, 0.7, 2.2, 5.9])

        outputs = model(waveforms)
        losses = model.loss_terms(outputs, labels, nominal_angles)
        total = losses["action_nll"].mean() + 0.2 * losses["angle_nll"].mean() + 0.001 * model.smoothness_loss()
        total.backward()

        self.assertEqual(losses["action_nll"].shape, (4,))
        self.assertEqual(losses["angle_nll"].shape, (4,))
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))


    def test_template_curve_is_periodic_and_smoothness_is_finite(self) -> None:
        model = LatentAngleTemporalClassifier()
        angles = torch.tensor([0.0, 0.4, 1.8, 2.0 * torch.pi])
        curves = model.template_curves(angles)

        self.assertEqual(curves.shape, (4, 4, 516))
        torch.testing.assert_close(curves[:, 0], curves[:, -1], rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(model.smoothness_loss()))


    def test_no_rms_and_mlp_variants_have_expected_dimensions(self) -> None:
        no_rms = LatentAngleTemporalClassifier(include_rms=False).eval()
        mlp = LatentAngleMLPBaseline().eval()

        no_rms_outputs = no_rms(_waveforms(batch=2))
        mlp_outputs = mlp(_waveforms(batch=2))

        self.assertEqual(no_rms_outputs["features"].shape, (2, 512))
        self.assertEqual(no_rms_outputs["template_features"].shape, (4, 64, 512))
        self.assertEqual(mlp_outputs["logits"].shape, (2, 4))
        self.assertEqual(mlp_outputs["action_log_probs"].shape, (2, 4))
        self.assertEqual(sum(parameter.numel() for parameter in mlp.classifier.parameters()), 10_424)


    def test_checkpoint_round_trip_preserves_predictions_and_rms_statistics(self) -> None:
        rms_mean = torch.tensor([-2.0, -2.1, -1.9, -2.2])
        rms_std = torch.tensor([0.5, 0.4, 0.6, 0.7])
        model = LatentAngleTemporalClassifier(rms_mean=rms_mean, rms_std=rms_std).eval()
        samples = _waveforms()
        before = model(samples)
        checkpoint = BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)

        restored = LatentAngleTemporalClassifier().eval()
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        after = restored(samples)

        torch.testing.assert_close(after["action_log_probs"], before["action_log_probs"])
        torch.testing.assert_close(after["angle_log_probs"], before["angle_log_probs"])
        torch.testing.assert_close(restored.rms_mean, rms_mean)
        torch.testing.assert_close(restored.rms_std, rms_std)


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_model_runs_on_cuda_when_available(self) -> None:
        model = LatentAngleTemporalClassifier().cuda().eval()
        outputs = model(_waveforms().cuda())

        self.assertEqual(outputs["action_log_probs"].device.type, "cuda")
        self.assertTrue(torch.isfinite(outputs["joint_log_probs"]).all())
