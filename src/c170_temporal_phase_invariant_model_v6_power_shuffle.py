"""Causal temporal filtering followed by ideal global-phase-invariant features."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from src.c170_harmonic_phase_model_v4 import _DilatedResidualBlock


FEATURE_DIM: Final[int] = 23
FIR_COUNT: Final[int] = 8
FIR_KERNEL_SIZE: Final[int] = 65
FIR_LEFT_PADDING: Final[int] = 64
LAGS: Final[tuple[int, ...]] = (1, 2, 4, 8, 16, 32, 64)
VARIANTS: Final[tuple[str, ...]] = (
    "learned_gram",
    "fixed_gram",
    "learned_power",
    "learned_power_shuffle",
    "learned_shuffle",
)
FEATURE_NAMES: Final[tuple[str, ...]] = (
    "asinh_s",
    *(f"log1p_p{index}" for index in range(FIR_COUNT)),
    *(name for lag in LAGS for name in (f"q_lag{lag}_re", f"q_lag{lag}_im")),
)
ACTIVE_FEATURE_INDICES: Final[dict[str, tuple[int, ...]]] = {
    "learned_gram": tuple(range(FEATURE_DIM)),
    "fixed_gram": tuple(range(FEATURE_DIM)),
    "learned_power": tuple(range(9)),
    "learned_power_shuffle": tuple(range(9)),
    "learned_shuffle": tuple(range(FEATURE_DIM)),
}


def _validate_samples(samples: torch.Tensor) -> None:
    if not isinstance(samples, torch.Tensor):
        raise TypeError("samples must be a torch.Tensor")
    if samples.ndim != 3 or samples.shape[-1] != 4:
        raise ValueError(f"expected samples with shape (B,T,4), got {tuple(samples.shape)}")
    if not samples.is_floating_point():
        raise TypeError("samples must use a floating-point dtype")
    if samples.shape[0] < 1 or samples.shape[1] < 1:
        raise ValueError("samples require non-empty batch and time dimensions")


def extract_temporal_phase_features(
    samples: torch.Tensor,
    normalized_fir_kernels: torch.Tensor,
) -> torch.Tensor:
    """Build the ordered 23 channels from v3-preprocessed `[B,T,4]` windows.

    The FIR kernels have shape `(8,1,65)` and already have per-filter L2
    normalization. Causal padding is on the left only; output time `t` never
    depends on input samples after `t`.
    """
    _validate_samples(samples)
    if normalized_fir_kernels.shape != (FIR_COUNT, 1, FIR_KERNEL_SIZE):
        raise ValueError(
            f"expected FIR kernels with shape {(FIR_COUNT, 1, FIR_KERNEL_SIZE)}, "
            f"got {tuple(normalized_fir_kernels.shape)}"
        )
    if normalized_fir_kernels.device != samples.device or normalized_fir_kernels.dtype != samples.dtype:
        raise ValueError("FIR kernels must match the sample device and dtype")

    x0, x1, x2, x3 = samples.unbind(dim=-1)
    s = (x0 + x1 + x2 + x3) * 0.25
    a = (x0 - x2) * 0.5
    b = (x1 - x3) * 0.5

    padded_a = F.pad(a.unsqueeze(1), (FIR_LEFT_PADDING, 0))
    padded_b = F.pad(b.unsqueeze(1), (FIR_LEFT_PADDING, 0))
    filtered_a = F.conv1d(padded_a, normalized_fir_kernels, bias=None, stride=1, padding=0)
    filtered_b = F.conv1d(padded_b, normalized_fir_kernels, bias=None, stride=1, padding=0)
    power = filtered_a.square() + filtered_b.square()

    features: list[torch.Tensor] = [torch.asinh(s).unsqueeze(1), torch.log1p(power)]
    reference_a = filtered_a[:, :1]
    reference_b = filtered_b[:, :1]
    reference_power = power[:, :1]
    for filter_index in range(1, FIR_COUNT):
        response_a = filtered_a[:, filter_index : filter_index + 1]
        response_b = filtered_b[:, filter_index : filter_index + 1]
        response_power = power[:, filter_index : filter_index + 1]
        denominator = torch.sqrt((1.0 + reference_power) * (1.0 + response_power))
        correlation_re = (reference_a * response_a + reference_b * response_b) / denominator
        correlation_im = (reference_b * response_a - reference_a * response_b) / denominator
        features.extend((correlation_re, correlation_im))

    result = torch.cat(features, dim=1)
    if result.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"expected {FEATURE_DIM} features, got {result.shape[1]}")
    return result


def select_variant_features(features: torch.Tensor, variant: str) -> torch.Tensor:
    if variant not in ACTIVE_FEATURE_INDICES:
        raise ValueError(f"unknown v6 temporal phase-invariant variant: {variant}")
    if features.ndim != 3 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected features with shape (B,{FEATURE_DIM},T), got {tuple(features.shape)}")
    mask = features.new_zeros((1, FEATURE_DIM, 1))
    mask[:, ACTIVE_FEATURE_INDICES[variant], :] = 1.0
    return features * mask


class TemporalPhaseInvariantClassifierV6(nn.Module):
    """Causal eight-filter complex bank and the unchanged low-capacity backend."""

    def __init__(self, variant: str, n_classes: int = 4) -> None:
        super().__init__()
        if variant not in ACTIVE_FEATURE_INDICES:
            raise ValueError(f"unknown v6 temporal phase-invariant variant: {variant}")
        if n_classes != 4:
            raise ValueError("the C170 experiment has exactly four action classes")
        self.variant = variant

        identity = torch.zeros((1, 1, FIR_KERNEL_SIZE), dtype=torch.float32)
        identity[0, 0, FIR_LEFT_PADDING] = 1.0
        self.register_buffer("identity_kernel", identity)
        learned_kernels = torch.zeros((len(LAGS), 1, FIR_KERNEL_SIZE), dtype=torch.float32)
        for index, lag in enumerate(LAGS):
            learned_kernels[index, 0, FIR_LEFT_PADDING - lag] = 1.0
        self.learned_fir_weights = nn.Parameter(
            learned_kernels,
            requires_grad=(variant != "fixed_gram"),
        )

        active = ACTIVE_FEATURE_INDICES[variant]
        feature_mask = torch.zeros((1, FEATURE_DIM, 1), dtype=torch.float32)
        feature_mask[:, active, :] = 1.0
        self.register_buffer("feature_mask", feature_mask, persistent=False)

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

    def normalized_fir_kernels(self) -> torch.Tensor:
        raw = torch.cat((self.identity_kernel, self.learned_fir_weights), dim=0)
        norms = torch.sqrt(raw.square().sum(dim=(1, 2), keepdim=True) + 1e-8)
        return raw / norms

    def filter_harmonics(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if a.ndim != 2 or b.shape != a.shape:
            raise ValueError("a and b must have the same shape (B,T)")
        kernels = self.normalized_fir_kernels()
        if kernels.dtype != a.dtype or kernels.device != a.device:
            raise ValueError("a and b must match the model FIR kernel dtype and device")
        padded_a = F.pad(a.unsqueeze(1), (FIR_LEFT_PADDING, 0))
        padded_b = F.pad(b.unsqueeze(1), (FIR_LEFT_PADDING, 0))
        return (
            F.conv1d(padded_a, kernels, bias=None, stride=1, padding=0),
            F.conv1d(padded_b, kernels, bias=None, stride=1, padding=0),
        )

    def encode(self, samples: torch.Tensor) -> torch.Tensor:
        features = extract_temporal_phase_features(samples, self.normalized_fir_kernels())
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
        return logits, F.normalize(representation, p=2.0, dim=1, eps=1e-12)
