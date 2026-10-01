"""Low-capacity first-harmonic phase feature model for the C170 v4 study."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_DIM: Final[int] = 12
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "asinh_s",
    "log1p_p",
    "lag1_re",
    "lag1_im",
    "lag4_re",
    "lag4_im",
    "lag16_re",
    "lag16_im",
    "lag64_re",
    "lag64_im",
    "asinh_a",
    "asinh_b",
)
LAGS: Final[tuple[int, ...]] = (1, 4, 16, 64)
VARIANTS: Final[tuple[str, ...]] = (
    "harmonic_gram",
    "harmonic_power",
    "harmonic_phase",
    "harmonic_shuffle",
)
ACTIVE_FEATURE_INDICES: Final[dict[str, tuple[int, ...]]] = {
    "harmonic_gram": tuple(range(10)),
    "harmonic_power": (0, 1),
    "harmonic_phase": tuple(range(FEATURE_DIM)),
    "harmonic_shuffle": tuple(range(10)),
}


def extract_harmonic_features(samples: torch.Tensor) -> torch.Tensor:
    """Return the fixed 12 real features as (batch, feature, time).

    Input must be filtered, per-window demeaned, globally scaled and shaped
    (B,T,4). Lagged products use zero padding at the start, never wrap.
    """
    if not isinstance(samples, torch.Tensor):
        raise TypeError("samples must be a torch.Tensor")
    if samples.ndim != 3 or samples.shape[-1] != 4:
        raise ValueError(f"expected samples with shape (B,T,4), got {tuple(samples.shape)}")
    if not samples.is_floating_point():
        raise TypeError("samples must use a floating-point dtype")
    if samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError("samples require non-empty batch and time dimensions")

    x0, x1, x2, x3 = samples.unbind(dim=-1)
    s = (x0 + x1 + x2 + x3) * 0.25
    a = (x0 - x2) * 0.5
    b = (x1 - x3) * 0.5
    power = a.square() + b.square()

    sequence: list[torch.Tensor] = [torch.asinh(s), torch.log1p(power)]
    time_count = samples.shape[1]
    for lag in LAGS:
        previous_a = torch.zeros_like(a)
        previous_b = torch.zeros_like(b)
        previous_power = torch.zeros_like(power)
        if lag < time_count:
            previous_a[:, lag:] = a[:, :-lag]
            previous_b[:, lag:] = b[:, :-lag]
            previous_power[:, lag:] = power[:, :-lag]
        denominator = torch.sqrt((1.0 + power) * (1.0 + previous_power))
        correlation_re = (a * previous_a + b * previous_b) / denominator
        correlation_im = (b * previous_a - a * previous_b) / denominator
        if lag:
            correlation_re = torch.cat(
                (torch.zeros_like(correlation_re[:, :lag]), correlation_re[:, lag:]), dim=1
            )
            correlation_im = torch.cat(
                (torch.zeros_like(correlation_im[:, :lag]), correlation_im[:, lag:]), dim=1
            )
        sequence.extend((correlation_re, correlation_im))
    sequence.extend((torch.asinh(a), torch.asinh(b)))
    return torch.stack(sequence, dim=1)


def select_variant_features(features: torch.Tensor, variant: str) -> torch.Tensor:
    """Apply one fixed, parameter-free feature mask for a registered arm."""
    if variant not in ACTIVE_FEATURE_INDICES:
        raise ValueError(f"unknown harmonic feature variant: {variant}")
    if features.ndim != 3 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected features with shape (B,{FEATURE_DIM},T), got {tuple(features.shape)}")
    mask = features.new_zeros((1, FEATURE_DIM, 1))
    mask[:, ACTIVE_FEATURE_INDICES[variant], :] = 1.0
    return features * mask


class _DilatedResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        padding = 3 * dilation
        self.depthwise = nn.Conv1d(
            24,
            24,
            kernel_size=7,
            padding=padding,
            dilation=dilation,
            groups=24,
            bias=False,
        )
        self.depthwise_norm = nn.GroupNorm(4, 24)
        self.activation = nn.SiLU()
        self.pointwise = nn.Conv1d(24, 24, kernel_size=1, bias=False)
        self.pointwise_norm = nn.GroupNorm(4, 24)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.depthwise(values)
        update = self.activation(self.depthwise_norm(update))
        update = self.pointwise_norm(self.pointwise(update))
        return self.activation(values + update)


class HarmonicPhaseClassifierV4(nn.Module):
    """Shared low-capacity temporal classifier with fixed feature-mask variants."""

    def __init__(self, variant: str, n_classes: int = 4) -> None:
        super().__init__()
        if variant not in ACTIVE_FEATURE_INDICES:
            raise ValueError(f"unknown harmonic feature variant: {variant}")
        if n_classes != 4:
            raise ValueError("the C170 experiment has exactly four action classes")
        self.variant = variant
        active = ACTIVE_FEATURE_INDICES[variant]
        mask = torch.zeros((1, FEATURE_DIM, 1), dtype=torch.float32)
        mask[:, active, :] = 1.0
        self.register_buffer("feature_mask", mask, persistent=False)

        self.stem = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, 24, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
        )
        self.residual_blocks = nn.Sequential(
            _DilatedResidualBlock(1),
            _DilatedResidualBlock(2),
            _DilatedResidualBlock(4),
        )
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(192, n_classes)

    def encode(self, samples: torch.Tensor) -> torch.Tensor:
        features = extract_harmonic_features(samples)
        features = features * self.feature_mask
        temporal = self.residual_blocks(self.stem(features))
        pooled = torch.cat(
            (
                F.adaptive_avg_pool1d(temporal, 1).flatten(start_dim=1),
                F.adaptive_avg_pool1d(temporal, 2).flatten(start_dim=1),
                F.adaptive_avg_pool1d(temporal, 4).flatten(start_dim=1),
                torch.std(temporal, dim=-1, unbiased=False),
            ),
            dim=1,
        )
        if pooled.shape[1] != 192:
            raise RuntimeError(f"expected 192 pooled dimensions, got {pooled.shape[1]}")
        return pooled

    def forward(
        self,
        samples: torch.Tensor,
        *,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(samples)
        logits = self.classifier(self.dropout(representation))
        if not return_embedding:
            return logits
        diagnostic_embedding = F.normalize(representation, p=2.0, dim=1, eps=1e-12)
        return logits, diagnostic_embedding
