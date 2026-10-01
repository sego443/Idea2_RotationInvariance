"""Low-capacity C4 model with first-harmonic and Nyquist-coupling features."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from src.c170_harmonic_phase_model_v4 import (
    FEATURE_NAMES as V4_FEATURE_NAMES,
    LAGS,
    _DilatedResidualBlock,
    extract_harmonic_features,
)


FEATURE_DIM: Final[int] = 17
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(V4_FEATURE_NAMES[:10]) + (
    "log1p_n_squared",
    "lag1_n_correlation",
    "lag4_n_correlation",
    "lag16_n_correlation",
    "lag64_n_correlation",
    "nyquist_z2_coupling_re",
    "nyquist_z2_coupling_im",
)
VARIANTS: Final[tuple[str, ...]] = (
    "nyquist_coupled",
    "nyquist_uncoupled",
    "gram_control",
    "nyquist_shuffle",
)
ACTIVE_FEATURE_INDICES: Final[dict[str, tuple[int, ...]]] = {
    "nyquist_coupled": tuple(range(17)),
    "nyquist_uncoupled": tuple(range(15)),
    "gram_control": tuple(range(10)),
    "nyquist_shuffle": tuple(range(17)),
}


def extract_nyquist_features(samples: torch.Tensor) -> torch.Tensor:
    """Return v4 gram's first ten channels plus seven C4-invariant features.

    Input has already passed the frozen v3 filter, per-window channel demeaning,
    and shared train-derived scalar normalization. Output shape is (B,17,T).
    Lag channels use a zero prefix and never wrap around time.
    """
    if not isinstance(samples, torch.Tensor):
        raise TypeError("samples must be a torch.Tensor")
    if samples.ndim != 3 or samples.shape[-1] != 4:
        raise ValueError(f"expected samples with shape (B,T,4), got {tuple(samples.shape)}")
    if not samples.is_floating_point():
        raise TypeError("samples must use a floating-point dtype")
    if samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError("samples require non-empty batch and time dimensions")

    v4_gram = extract_harmonic_features(samples)[:, :10]
    x0, x1, x2, x3 = samples.unbind(dim=-1)
    a = (x0 - x2) * 0.5
    b = (x1 - x3) * 0.5
    n = (x0 - x1 + x2 - x3) * 0.25
    power = a.square() + b.square()
    n_squared = n.square()

    features: list[torch.Tensor] = [v4_gram, torch.log1p(n_squared).unsqueeze(1)]
    time_count = samples.shape[1]
    for lag in LAGS:
        previous_n = torch.zeros_like(n)
        if lag < time_count:
            previous_n[:, lag:] = n[:, :-lag]
        correlation = (n * previous_n) / torch.sqrt(
            (1.0 + n_squared) * (1.0 + previous_n.square())
        )
        if lag:
            correlation = torch.cat(
                (torch.zeros_like(correlation[:, :lag]), correlation[:, lag:]), dim=1
            )
        features.append(correlation.unsqueeze(1))

    coupling_denominator = torch.sqrt(1.0 + n_squared) * (1.0 + power)
    coupling_re = n * (a.square() - b.square()) / coupling_denominator
    coupling_im = n * (2.0 * a * b) / coupling_denominator
    features.extend((coupling_re.unsqueeze(1), coupling_im.unsqueeze(1)))
    result = torch.cat(features, dim=1)
    if result.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} features, got {result.shape[1]}")
    return result


def select_variant_features(features: torch.Tensor, variant: str) -> torch.Tensor:
    """Apply a fixed feature mask for one registered v5 arm."""
    if variant not in ACTIVE_FEATURE_INDICES:
        raise ValueError(f"unknown v5 Nyquist variant: {variant}")
    if features.ndim != 3 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected features with shape (B,{FEATURE_DIM},T), got {tuple(features.shape)}")
    mask = features.new_zeros((1, FEATURE_DIM, 1))
    mask[:, ACTIVE_FEATURE_INDICES[variant], :] = 1.0
    return features * mask


class HarmonicNyquistClassifierV5(nn.Module):
    """Fixed 17-channel feature model with the v4 low-capacity backend."""

    def __init__(self, variant: str, n_classes: int = 4) -> None:
        super().__init__()
        if variant not in ACTIVE_FEATURE_INDICES:
            raise ValueError(f"unknown v5 Nyquist variant: {variant}")
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
        features = extract_nyquist_features(samples) * self.feature_mask
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
