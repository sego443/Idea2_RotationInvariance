"""Multi-scale complex stem and sign-invariant Nyquist branch for C170 v16."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from src.c170_complex_orbit_model_v11 import (
    C170ComplexOrbitClassifierV11,
    ComplexConv1d,
    ComplexPowerGramClassifier,
    _temporal_descriptor,
)

ARCHITECTURES: Final[tuple[str, ...]] = ("multiscale", "nyquist", "multiscale_nyquist")
DESCRIPTOR_DIM: Final[dict[str, int]] = {"multiscale": 672, "nyquist": 736, "multiscale_nyquist": 736}
PARAMETER_COUNT: Final[dict[str, int]] = {"multiscale": 10956, "nyquist": 11172, "multiscale_nyquist": 11300}


class MultiScaleComplexConv1d(nn.Module):
    """Parallel no-bias complex convolutions with matched stride-two lengths."""

    def __init__(self) -> None:
        super().__init__()
        self.short = ComplexConv1d(1, 8, kernel_size=9, stride=2, padding=4)
        self.long = ComplexConv1d(1, 8, kernel_size=65, stride=2, padding=32)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sr, si = self.short(real, imag)
        lr, li = self.long(real, imag)
        if sr.shape[-1] != lr.shape[-1]:
            raise RuntimeError(f"multi-scale convolution lengths differ: {sr.shape[-1]} vs {lr.shape[-1]}")
        return torch.cat((sr, lr), dim=1), torch.cat((si, li), dim=1)


class C170MultiscaleNyquistClassifierV16(C170ComplexOrbitClassifierV11):
    def __init__(self, variant: str, architecture: str, n_classes: int = 4) -> None:
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown v16 architecture: {architecture}")
        if n_classes != 4:
            raise ValueError("v16 is fixed to four action classes")
        super().__init__(variant, n_classes)
        if variant.removesuffix("_shuffle") != "complex_gram":
            raise ValueError("v16 wraps only the complex_gram baseline")
        self.architecture_v16 = architecture
        self.use_multiscale = architecture in {"multiscale", "multiscale_nyquist"}
        self.use_nyquist = architecture in {"nyquist", "multiscale_nyquist"}
        if self.use_multiscale:
            self.core.complex_conv1 = MultiScaleComplexConv1d()
        if self.use_nyquist:
            self.nyquist_conv = nn.Conv1d(1, 8, kernel_size=9, stride=2, padding=4, bias=False)
            self.nyquist_norm = nn.GroupNorm(4, 8)
            old = self.core.classifier
            expanded = nn.Linear(736, 4)
            with torch.no_grad():
                expanded.weight[:, :672].copy_(old.weight)
                expanded.bias.copy_(old.bias)
            self.core.classifier = expanded

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f"expected [B,T,4], got {tuple(samples.shape)}")
        core = self.core
        x0, x1, x2, x3 = samples.unbind(dim=-1)
        s = (x0 + x1 + x2 + x3) * 0.25
        a = ((x0 - x2) * 0.5).unsqueeze(1)
        b = ((x1 - x3) * 0.5).unsqueeze(1)
        r1, i1 = core.complex_conv1(a, b)
        r1, i1 = core.gate1(r1, i1)
        r2, i2 = core.complex_conv2(r1, i1)
        r2, i2 = core.gate2(r2, i2)
        p1 = torch.log1p(r1.square() + i1.square())
        p2 = torch.log1p(r2.square() + i2.square())
        descriptors = [_temporal_descriptor(p1), _temporal_descriptor(p2)]
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
        descriptors.append(_temporal_descriptor(torch.cat((corr_r, corr_i), dim=2).reshape(batch, 36, time)))
        descriptors.append(_temporal_descriptor(core.s_stem(s.unsqueeze(1))))
        if self.use_nyquist:
            # This alternating-sign sensor mode changes sign under a one-slot C4 shift.
            n = ((x0 - x1 + x2 - x3) * 0.5).unsqueeze(1)
            n_pos = F.silu(self.nyquist_norm(self.nyquist_conv(n)))
            n_neg = F.silu(self.nyquist_norm(self.nyquist_conv(-n)))
            descriptors.append(0.5 * (_temporal_descriptor(n_pos) + _temporal_descriptor(n_neg)))
        representation = torch.cat(descriptors, dim=1)
        expected = DESCRIPTOR_DIM[self.architecture_v16]
        if representation.shape[1] != expected:
            raise RuntimeError(f"expected v16 descriptor size {expected}, got {representation.shape[1]}")
        return core.classifier(core.dropout(representation))


def parameter_count(architecture: str) -> int:
    return PARAMETER_COUNT[architecture]
