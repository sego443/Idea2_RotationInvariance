"""Continuous latent-angle action classification for the C170 windows."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


N_ACTIONS = 4
N_ANGLES = 64
FEATURES_PER_SENSOR = 16
TEMPORAL_FEATURES = 512
ANGLE_BANDWIDTH = 8.0
SCORE_SCALE = 16.0


def _four_values(value: Sequence[float] | torch.Tensor | None, default: float) -> torch.Tensor:
    if value is None:
        return torch.full((4,), float(default), dtype=torch.float32)
    values = torch.as_tensor(value, dtype=torch.float32).detach().clone()
    if values.shape != (4,):
        raise ValueError(f"RMS statistics must have shape (4,), got {tuple(values.shape)}")
    if not torch.isfinite(values).all():
        raise ValueError("RMS statistics must be finite")
    return values


class _TemporalResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            FEATURES_PER_SENSOR,
            FEATURES_PER_SENSOR,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=FEATURES_PER_SENSOR,
        )
        self.pointwise = nn.Conv1d(FEATURES_PER_SENSOR, FEATURES_PER_SENSOR, kernel_size=1)
        self.norm = nn.LayerNorm(FEATURES_PER_SENSOR)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values.transpose(1, 2)
        update = self.pointwise(self.depthwise(values)).transpose(1, 2)
        return F.gelu(self.norm(residual + update)).transpose(1, 2)


class _C170TemporalEncoder(nn.Module):
    """Shared per-sensor temporal encoder and optional standardized log-RMS features."""

    def __init__(
        self,
        include_rms: bool,
        rms_mean: Sequence[float] | torch.Tensor | None,
        rms_std: Sequence[float] | torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.include_rms = bool(include_rms)
        self.sensor_stem = nn.Conv1d(1, FEATURES_PER_SENSOR, kernel_size=9, stride=2, padding=4)
        self.temporal_blocks = nn.ModuleList(
            [_TemporalResidualBlock(dilation) for dilation in (1, 2, 4)]
        )
        self.sensor_mixer = nn.Linear(4 * FEATURES_PER_SENSOR, 32)
        self.time_pool = nn.AdaptiveAvgPool1d(16)
        self.register_buffer("rms_mean", _four_values(rms_mean, 0.0))
        std = _four_values(rms_std, 1.0)
        if torch.any(std <= 0):
            raise ValueError("RMS standard deviations must be positive")
        self.register_buffer("rms_std", std)

    @property
    def feature_dim(self) -> int:
        return TEMPORAL_FEATURES + (4 if self.include_rms else 0)

    @staticmethod
    def compute_log_rms(samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError("waveforms must have shape (batch, time, 4)")
        values = samples.float()
        return 0.5 * torch.log(values.square().mean(dim=1).clamp_min(1e-12))

    def set_rms_stats(
        self, rms_mean: Sequence[float] | torch.Tensor, rms_std: Sequence[float] | torch.Tensor
    ) -> None:
        mean = _four_values(rms_mean, 0.0).to(device=self.rms_mean.device)
        std = _four_values(rms_std, 1.0).to(device=self.rms_std.device)
        if torch.any(std <= 0):
            raise ValueError("RMS standard deviations must be positive")
        with torch.no_grad():
            self.rms_mean.copy_(mean)
            self.rms_std.copy_(std)

    def forward(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if samples.ndim != 3 or tuple(samples.shape[1:]) != (500, 4):
            raise ValueError(f"waveforms must have shape (batch, 500, 4), got {tuple(samples.shape)}")

        batch_size = samples.shape[0]
        per_sensor = samples.float().permute(0, 2, 1).contiguous().reshape(batch_size * 4, 1, 500)
        values = self.sensor_stem(per_sensor)
        for block in self.temporal_blocks:
            values = block(values)

        time_steps = values.shape[-1]
        values = (
            values.reshape(batch_size, 4, FEATURES_PER_SENSOR, time_steps)
            .permute(0, 3, 1, 2)
            .contiguous()
            .reshape(batch_size, time_steps, 4 * FEATURES_PER_SENSOR)
        )
        values = self.sensor_mixer(values).transpose(1, 2)
        temporal_features = self.time_pool(values).flatten(1)
        log_rms = self.compute_log_rms(samples)

        with torch.autocast(device_type=samples.device.type, enabled=False):
            temporal_features = temporal_features.float()
            log_rms = log_rms.float()
            if self.include_rms:
                standardized_rms = (log_rms - self.rms_mean.float()) / self.rms_std.float()
                features = torch.cat((temporal_features, standardized_rms), dim=1)
            else:
                features = temporal_features
        return features, log_rms


class LatentAngleTemporalClassifier(nn.Module):
    """Match a temporal feature to action-conditioned periodic angle templates."""

    def __init__(
        self,
        num_angles: int = N_ANGLES,
        include_rms: bool = True,
        rms_mean: Sequence[float] | torch.Tensor | None = None,
        rms_std: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_angles != N_ANGLES:
            raise ValueError(f"the design fixes the angle grid at {N_ANGLES} nodes")
        self.num_angles = num_angles
        self.encoder = _C170TemporalEncoder(include_rms, rms_mean, rms_std)
        feature_dim = self.encoder.feature_dim
        self.template_coefficients = nn.Parameter(torch.empty(N_ACTIONS, 5, feature_dim))
        nn.init.normal_(self.template_coefficients, mean=0.0, std=0.02)
        theta_grid = torch.arange(N_ANGLES, dtype=torch.float32) * (2.0 * math.pi / N_ANGLES)
        self.register_buffer("theta_grid", theta_grid)

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

    @property
    def rms_mean(self) -> torch.Tensor:
        return self.encoder.rms_mean

    @property
    def rms_std(self) -> torch.Tensor:
        return self.encoder.rms_std

    def set_rms_stats(
        self, rms_mean: Sequence[float] | torch.Tensor, rms_std: Sequence[float] | torch.Tensor
    ) -> None:
        self.encoder.set_rms_stats(rms_mean, rms_std)

    def template_curves(self, angles_rad: torch.Tensor) -> torch.Tensor:
        """Return the unnormalized 4 x angle x feature Fourier template curves."""
        angles = torch.as_tensor(angles_rad, device=self.template_coefficients.device, dtype=torch.float32)
        basis = torch.stack(
            (
                torch.ones_like(angles),
                torch.cos(angles),
                torch.sin(angles),
                torch.cos(2.0 * angles),
                torch.sin(2.0 * angles),
            ),
            dim=-1,
        )
        flat_basis = basis.reshape(-1, 5)
        curves = torch.einsum("cfd,jf->cjd", self.template_coefficients.float(), flat_basis)
        return curves.reshape(N_ACTIONS, *angles.shape, self.feature_dim)

    def _normalized_template_grid(self) -> torch.Tensor:
        with torch.autocast(device_type=self.template_coefficients.device.type, enabled=False):
            curves = self.template_curves(self.theta_grid)
            return F.normalize(curves.float(), p=2.0, dim=-1, eps=1e-8)

    def forward(self, samples: torch.Tensor) -> dict[str, torch.Tensor]:
        features, log_rms = self.encoder(samples)
        with torch.autocast(device_type=samples.device.type, enabled=False):
            features = features.float()
            normalized_features = F.normalize(features, p=2.0, dim=-1, eps=1e-8)
            template_features = self._normalized_template_grid().float()
            scores = SCORE_SCALE * torch.einsum("bd,cjd->bcj", normalized_features, template_features)
            joint_log_probs = F.log_softmax(scores.flatten(1), dim=1).reshape(
                samples.shape[0], N_ACTIONS, self.num_angles
            )
            action_log_probs = torch.logsumexp(joint_log_probs, dim=2)
            angle_log_probs = torch.logsumexp(joint_log_probs, dim=1)
            conditional_angle_log_probs = joint_log_probs - action_log_probs.unsqueeze(-1)
        return {
            "scores": scores,
            "joint_log_probs": joint_log_probs,
            "action_log_probs": action_log_probs,
            "angle_log_probs": angle_log_probs,
            "conditional_angle_log_probs": conditional_angle_log_probs,
            "features": features,
            "normalized_features": normalized_features,
            "template_features": template_features,
            "log_rms": log_rms,
        }

    def loss_terms(
        self,
        outputs: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        nominal_angle_rad: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return un-reduced action and angle NLLs; the angle target bandwidth is fixed at 8."""
        action_log_probs = outputs["action_log_probs"]
        conditional = outputs["conditional_angle_log_probs"]
        batch_size = action_log_probs.shape[0]
        if actions.ndim != 1 or actions.shape[0] != batch_size:
            raise ValueError("actions must have shape (batch,)")
        if nominal_angle_rad.ndim != 1 or nominal_angle_rad.shape[0] != batch_size:
            raise ValueError("nominal angles must have shape (batch,)")
        actions = actions.to(device=action_log_probs.device, dtype=torch.long)
        nominal_angle_rad = nominal_angle_rad.to(device=conditional.device, dtype=torch.float32)

        with torch.autocast(device_type=action_log_probs.device.type, enabled=False):
            action_nll = -action_log_probs.float().gather(1, actions.unsqueeze(1)).squeeze(1)
            target_log_density = ANGLE_BANDWIDTH * torch.cos(
                self.theta_grid.float().unsqueeze(0) - nominal_angle_rad.unsqueeze(1)
            )
            target_distribution = F.softmax(target_log_density, dim=1)
            conditional_for_action = conditional.float().gather(
                1,
                actions.view(-1, 1, 1).expand(-1, 1, self.num_angles),
            ).squeeze(1)
            angle_nll = -(target_distribution * conditional_for_action).sum(dim=1)
        return {"action_nll": action_nll, "angle_nll": angle_nll}

    def smoothness_loss(self) -> torch.Tensor:
        """Periodic second-difference penalty on unit-normalized action templates."""
        with torch.autocast(device_type=self.template_coefficients.device.type, enabled=False):
            templates = self._normalized_template_grid().float()
            delta = 2.0 * math.pi / self.num_angles
            second_difference = (
                torch.roll(templates, shifts=-1, dims=1)
                - 2.0 * templates
                + torch.roll(templates, shifts=1, dims=1)
            ) / (delta * delta)
            return second_difference.square().sum(dim=-1).mean()


class LatentAngleMLPBaseline(nn.Module):
    """Same temporal encoder with the matched-capacity 516-to-20-to-4 control head."""

    def __init__(
        self,
        include_rms: bool = True,
        rms_mean: Sequence[float] | torch.Tensor | None = None,
        rms_std: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = _C170TemporalEncoder(include_rms, rms_mean, rms_std)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, 20),
            nn.GELU(),
            nn.Linear(20, N_ACTIONS),
        )

    def set_rms_stats(
        self, rms_mean: Sequence[float] | torch.Tensor, rms_std: Sequence[float] | torch.Tensor
    ) -> None:
        self.encoder.set_rms_stats(rms_mean, rms_std)

    def forward(self, samples: torch.Tensor) -> dict[str, torch.Tensor]:
        features, log_rms = self.encoder(samples)
        logits = self.classifier(features)
        with torch.autocast(device_type=samples.device.type, enabled=False):
            action_log_probs = F.log_softmax(logits.float(), dim=1)
        return {
            "logits": logits,
            "action_log_probs": action_log_probs,
            "features": features,
            "log_rms": log_rms,
        }
