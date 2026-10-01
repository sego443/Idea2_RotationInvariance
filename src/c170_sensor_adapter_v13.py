"""Identity-initialized, label-independent C170 sensor calibration adapters."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


ADAPTERS = ("per_sensor_gain", "per_sensor_fir_gain", "shared_fir")
FIR_TAPS = 9
REGULARIZATION_WEIGHT = 1e-3


class SensorAdapterV13(nn.Module):
    """Apply a fixed-ID sensor gain and/or zero-sum residual FIR before a model."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in ADAPTERS:
            raise ValueError(f"unknown sensor adapter: {kind}")
        self.kind = kind
        self.log_gain = (
            nn.Parameter(torch.zeros(4))
            if kind in {"per_sensor_gain", "per_sensor_fir_gain"}
            else None
        )
        if kind == "per_sensor_fir_gain":
            self.fir_residual = nn.Parameter(torch.zeros(4, 1, FIR_TAPS))
        elif kind == "shared_fir":
            self.fir_residual = nn.Parameter(torch.zeros(1, 1, FIR_TAPS))
        else:
            self.register_parameter("fir_residual", None)

    def centered_log_gain(self) -> torch.Tensor | None:
        if self.log_gain is None:
            return None
        return self.log_gain - self.log_gain.mean()

    def centered_fir_residual(self) -> torch.Tensor | None:
        if self.fir_residual is None:
            return None
        return self.fir_residual - self.fir_residual.mean(dim=-1, keepdim=True)

    def regularization_loss(self) -> torch.Tensor:
        terms = []
        gains = self.centered_log_gain()
        residual = self.centered_fir_residual()
        if gains is not None:
            terms.append(gains.square().mean())
        if residual is not None:
            terms.append(residual.square().mean())
        if not terms:
            raise RuntimeError("sensor adapter has no regularized parameters")
        return REGULARIZATION_WEIGHT * torch.stack(terms).sum()

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f"expected [B,T,4] sensor windows, got {tuple(samples.shape)}")
        result = samples
        residual = self.centered_fir_residual()
        if residual is not None:
            if self.kind == "shared_fir":
                weight = residual.expand(4, -1, -1).contiguous()
            else:
                weight = residual
            filtered_residual = F.conv1d(
                result.transpose(1, 2), weight, bias=None, stride=1, padding=FIR_TAPS // 2, groups=4
            ).transpose(1, 2)
            result = result + filtered_residual
        gains = self.centered_log_gain()
        if gains is not None:
            result = result * torch.exp(gains).view(1, 1, 4)
        return result
