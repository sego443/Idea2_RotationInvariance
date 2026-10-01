from __future__ import annotations

from typing import Final

import math
import torch
from torch import nn
from torch.nn import functional as F

from src.c170_harmonic_phase_model_v4 import _DilatedResidualBlock

ARCHITECTURES: Final[tuple[str, ...]] = ('complex_power', 'complex_gram', 'orbit_mean', 'orbit_lse')
VARIANTS: Final[tuple[str, ...]] = ARCHITECTURES + tuple(f'{name}_shuffle' for name in ARCHITECTURES)
FEATURE_NAMES: Final[tuple[str, ...]] = ('S0', 'S1', 'S2', 'S3')
LOG_RMS_EPSILON: Final[float] = 1e-8
COMPLEX_GATE_EPSILON: Final[float] = 1.0


def architecture_for_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f'unknown v11 variant: {variant}')
    return variant[:-8] if variant.endswith('_shuffle') else variant


def is_shuffle_variant(variant: str) -> bool:
    architecture_for_variant(variant)
    return variant.endswith('_shuffle')


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.real_weight = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.imag_weight = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.real_weight(real) - self.imag_weight(imag),
                self.real_weight(imag) + self.imag_weight(real))


class PhaseEquivariantGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((1, channels, 1), 0.1))
        self.beta = nn.Parameter(torch.zeros((1, channels, 1)))

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        power = real.square() + imag.square()
        gate = torch.sigmoid(self.gamma * torch.log1p(power / COMPLEX_GATE_EPSILON) + self.beta)
        return real * gate, imag * gate


def _temporal_descriptor(values: torch.Tensor) -> torch.Tensor:
    parts = [F.adaptive_avg_pool1d(values, bins).flatten(1) for bins in (1, 2, 4)]
    parts.append(torch.std(values, dim=-1, unbiased=False))
    return torch.cat(parts, dim=1)


class ComplexPowerGramClassifier(nn.Module):
    def __init__(self, architecture: str, n_classes: int = 4) -> None:
        super().__init__()
        if architecture not in ('complex_power', 'complex_gram') or n_classes != 4:
            raise ValueError('invalid complex v11 architecture or class count')
        self.architecture = architecture
        self.complex_conv1 = ComplexConv1d(1, 16, kernel_size=33, stride=2, padding=16)
        self.gate1 = PhaseEquivariantGate(16)
        self.complex_conv2 = ComplexConv1d(16, 24, kernel_size=9, stride=2, padding=4)
        self.gate2 = PhaseEquivariantGate(24)
        self.s_stem = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
        )
        head_dim = 384 if architecture == 'complex_power' else 672
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(head_dim, n_classes)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f'expected [B,T,4], got {tuple(samples.shape)}')
        x0, x1, x2, x3 = samples.unbind(dim=-1)
        s = (x0 + x1 + x2 + x3) * 0.25
        a = ((x0 - x2) * 0.5).unsqueeze(1)
        b = ((x1 - x3) * 0.5).unsqueeze(1)
        r1, i1 = self.complex_conv1(a, b)
        r1, i1 = self.gate1(r1, i1)
        r2, i2 = self.complex_conv2(r1, i1)
        r2, i2 = self.gate2(r2, i2)
        p1 = torch.log1p(r1.square() + i1.square())
        p2 = torch.log1p(r2.square() + i2.square())
        descriptors = [_temporal_descriptor(p1), _temporal_descriptor(p2)]
        if self.architecture == 'complex_gram':
            batch, channels, time = r2.shape
            rr = r2.reshape(batch, 6, 4, time)
            ii = i2.reshape(batch, 6, 4, time)
            ref_r, ref_i = rr[:, :, :1, :], ii[:, :, :1, :]
            other_r, other_i = rr[:, :, 1:, :], ii[:, :, 1:, :]
            ref_p = ref_r.square() + ref_i.square()
            other_p = other_r.square() + other_i.square()
            denom = torch.sqrt((1.0 + ref_p) * (1.0 + other_p))
            corr_r = (ref_r * other_r + ref_i * other_i) / denom
            corr_i = (ref_i * other_r - ref_r * other_i) / denom
            correlations = torch.cat((corr_r, corr_i), dim=2).reshape(batch, 36, time)
            descriptors.append(_temporal_descriptor(correlations))
        s_features = self.s_stem(s.unsqueeze(1))
        descriptors.append(_temporal_descriptor(s_features))
        representation = torch.cat(descriptors, dim=1)
        expected = 384 if self.architecture == 'complex_power' else 672
        if representation.shape[1] != expected:
            raise RuntimeError(f'expected descriptor size {expected}, got {representation.shape[1]}')
        return self.classifier(self.dropout(representation))


class OrbitClassifier(nn.Module):
    def __init__(self, architecture: str, n_classes: int = 4) -> None:
        super().__init__()
        if architecture not in ('orbit_mean', 'orbit_lse') or n_classes != 4:
            raise ValueError('invalid orbit v11 architecture or class count')
        self.architecture = architecture
        self.stem = nn.Sequential(
            nn.Conv1d(4, 24, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
        )
        self.residual_blocks = nn.Sequential(
            _DilatedResidualBlock(1),
            _DilatedResidualBlock(2),
            _DilatedResidualBlock(4),
        )
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(196, n_classes)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f'expected [B,T,4], got {tuple(samples.shape)}')
        batch = samples.shape[0]
        rolls = torch.stack([torch.roll(samples, shifts=shift, dims=-1) for shift in range(4)], dim=1)
        rolled = torch.asinh(rolls).reshape(batch * 4, samples.shape[1], 4).transpose(1, 2)
        temporal = self.residual_blocks(self.stem(rolled))
        pooled = _temporal_descriptor(temporal)
        rolled_rms = torch.log(torch.sqrt(rolls.square().mean(dim=2) + LOG_RMS_EPSILON))
        representation = torch.cat((pooled, rolled_rms.reshape(batch * 4, 4)), dim=1)
        if representation.shape[1] != 196:
            raise RuntimeError(f'expected orbit descriptor size 196, got {representation.shape[1]}')
        orbit_logits = self.classifier(self.dropout(representation)).reshape(batch, 4, 4)
        if self.architecture == 'orbit_mean':
            return orbit_logits.mean(dim=1)
        return torch.logsumexp(orbit_logits, dim=1) - math.log(4.0)


class C170ComplexOrbitClassifierV11(nn.Module):
    def __init__(self, variant: str, n_classes: int = 4) -> None:
        super().__init__()
        self.variant = variant
        self.architecture = architecture_for_variant(variant)
        if self.architecture.startswith('complex_'):
            self.core = ComplexPowerGramClassifier(self.architecture, n_classes)
        else:
            self.core = OrbitClassifier(self.architecture, n_classes)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.core(samples)


MODEL_PARAMETER_COUNTS: Final[dict[str, int]] = {
    'complex_power': 9676,
    'complex_gram': 10828,
    'orbit_mean': 4796,
    'orbit_lse': 4796,
}
VARIANT_ARCHITECTURE: Final[dict[str, str]] = {name: architecture_for_variant(name) for name in VARIANTS}
ACTIVE_FEATURE_INDICES: Final[dict[str, tuple[int, ...]]] = {name: (0, 1, 2, 3) for name in VARIANTS}
