"""Domain-adversarial auxiliary heads for the frozen v11 complex_gram trunk."""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from src.c170_complex_orbit_model_v11 import C170ComplexOrbitClassifierV11, _temporal_descriptor


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


class ComplexGramDomainV12(nn.Module):
    def __init__(self, domain_mode: str, n_positions: int = 11, hidden_dim: int = 32) -> None:
        super().__init__()
        if domain_mode not in {"unconditional", "classconditional"}:
            raise ValueError(f"unknown domain mode: {domain_mode}")
        if n_positions != 11:
            raise ValueError("the frozen training split has 11 position domains")
        self.domain_mode = domain_mode
        self.n_positions = n_positions
        self.backbone = C170ComplexOrbitClassifierV11("complex_gram", n_classes=4)
        out_dim = n_positions if domain_mode == "unconditional" else 4 * n_positions
        self.domain_head = nn.Sequential(
            nn.Linear(672, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def extract_embedding(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f"expected [B,T,4], got {tuple(samples.shape)}")
        core = self.backbone.core
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
        if channels != 24:
            raise RuntimeError(f"expected 24 complex stage-2 filters, got {channels}")
        rr = r2.reshape(batch, 6, 4, time)
        ii = i2.reshape(batch, 6, 4, time)
        ref_r, ref_i = rr[:, :, :1, :], ii[:, :, :1, :]
        other_r, other_i = rr[:, :, 1:, :], ii[:, :, 1:, :]
        ref_p = ref_r.square() + ref_i.square()
        other_p = other_r.square() + other_i.square()
        denominator = torch.sqrt((1.0 + ref_p) * (1.0 + other_p))
        corr_r = (ref_r * other_r + ref_i * other_i) / denominator
        corr_i = (ref_i * other_r - ref_r * other_i) / denominator
        descriptors.append(_temporal_descriptor(torch.cat((corr_r, corr_i), dim=2).reshape(batch, 36, time)))
        descriptors.append(_temporal_descriptor(core.s_stem(s.unsqueeze(1))))
        embedding = torch.cat(descriptors, dim=1)
        if embedding.shape[1] != 672:
            raise RuntimeError(f"expected 672-dimensional complex_gram embedding, got {embedding.shape[1]}")
        return embedding

    def forward(
        self,
        samples: torch.Tensor,
        *,
        domain_targets: torch.Tensor | None = None,
        conditioning_targets: torch.Tensor | None = None,
        reversal_scale: float = 0.0,
    ):
        embedding = self.extract_embedding(samples)
        logits = self.backbone.core.classifier(self.backbone.core.dropout(embedding))
        if domain_targets is None:
            return logits
        reversed_embedding = _GradientReverse.apply(embedding, reversal_scale)
        domain_logits = self.domain_head(reversed_embedding)
        if self.domain_mode == "classconditional":
            if conditioning_targets is None:
                raise ValueError("classconditional domain head requires the training target")
            domain_logits = domain_logits.reshape(-1, 4, self.n_positions)
            row = torch.arange(domain_logits.shape[0], device=domain_logits.device)
            domain_logits = domain_logits[row, conditioning_targets]
        return logits, domain_logits
