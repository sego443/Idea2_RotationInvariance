"""Temporal complex-Gram image classifiers for C170 v19."""

from __future__ import annotations

from typing import Final

import torch
from torch import nn
from torch.nn import functional as F

from src.c170_complex_orbit_model_v11 import ComplexPowerGramClassifier, _temporal_descriptor


ARCHITECTURES: Final[tuple[str, ...]] = ("gz", "gzn", "hybrid")
RESOLUTIONS: Final[tuple[int, ...]] = (64, 128)
IMAGE_FEATURE_DIM: Final[int] = 528
S_BRANCH_DIM: Final[int] = 64
POWER_FEATURE_DIM: Final[int] = 8
DESCRIPTOR_DIMS: Final[dict[str, int]] = {
    "gz": IMAGE_FEATURE_DIM + S_BRANCH_DIM + POWER_FEATURE_DIM,
    "gzn": IMAGE_FEATURE_DIM + S_BRANCH_DIM + POWER_FEATURE_DIM * 2,
    "hybrid": 672 + IMAGE_FEATURE_DIM + POWER_FEATURE_DIM,
}


class TemporalGramCNN(nn.Module):
    def __init__(self, image_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.layers(image)
        pooled = [F.adaptive_avg_pool2d(features, (size, size)).flatten(1) for size in (1, 2, 4)]
        pooled.append(torch.std(features, dim=(2, 3), unbiased=False))
        result = torch.cat(pooled, dim=1)
        if result.shape[1] != IMAGE_FEATURE_DIM:
            raise RuntimeError(f"expected {IMAGE_FEATURE_DIM} image features, got {result.shape[1]}")
        return result


class C170TemporalGramClassifierV19(nn.Module):
    """C4-invariant v19 model; inputs use the frozen v11 preprocessor."""

    def __init__(self, variant: str, architecture: str, resolution: int, n_classes: int = 4) -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown v19 architecture: {architecture}")
        if resolution not in RESOLUTIONS:
            raise ValueError(f"unknown v19 temporal resolution: {resolution}")
        if n_classes != 4 or variant.removesuffix("_shuffle") != "complex_gram":
            raise ValueError("v19 wraps the frozen four-class complex_gram training protocol")
        self.variant = variant
        self.architecture_v19 = architecture
        self.resolution = int(resolution)
        self.use_nyquist = architecture in {"gzn", "hybrid"}
        self.image_encoder = TemporalGramCNN(4 if self.use_nyquist else 3)

        if architecture == "hybrid":
            # Keep the frozen v11 convolutional descriptor through its 672D
            # pre-dropout representation; the v19 image adds 528D + Nyquist 8D.
            self.complex_gram = ComplexPowerGramClassifier("complex_gram", n_classes)
            del self.complex_gram.dropout
            del self.complex_gram.classifier
        else:
            self.s_stem = nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=9, stride=2, padding=4, bias=False),
                nn.GroupNorm(4, 8),
                nn.SiLU(),
            )

        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(DESCRIPTOR_DIMS[architecture], n_classes)

    def _gram_image_and_powers(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f"expected [B,T,4], got {tuple(samples.shape)}")
        x0, x1, x2, x3 = samples.unbind(dim=-1)
        zr = F.adaptive_avg_pool1d(((x0 - x2) * 0.5).unsqueeze(1), self.resolution).squeeze(1)
        zi = F.adaptive_avg_pool1d(((x1 - x3) * 0.5).unsqueeze(1), self.resolution).squeeze(1)
        n = F.adaptive_avg_pool1d(((x0 - x1 + x2 - x3) * 0.5).unsqueeze(1), self.resolution).squeeze(1)

        p = zr.square() + zi.square()
        denom = torch.sqrt(1.0 + p)
        ur, ui = zr / denom, zi / denom
        real = ur.unsqueeze(2) * ur.unsqueeze(1) + ui.unsqueeze(2) * ui.unsqueeze(1)
        imag = ui.unsqueeze(2) * ur.unsqueeze(1) - ur.unsqueeze(2) * ui.unsqueeze(1)
        log_power = torch.log1p(p)
        amplitude = 0.5 * (log_power.unsqueeze(2) + log_power.unsqueeze(1))
        channels = [real, imag, amplitude]
        if self.use_nyquist:
            g = n / torch.sqrt(1.0 + n.square())
            channels.append(g.unsqueeze(2) * g.unsqueeze(1))
        image = torch.stack(channels, dim=1)

        z_power = _temporal_descriptor(log_power.unsqueeze(1))
        n_power = _temporal_descriptor(torch.log1p(n.square()).unsqueeze(1))
        return image, z_power, n_power

    @staticmethod
    def _complex_gram_descriptor(core: ComplexPowerGramClassifier, samples: torch.Tensor) -> torch.Tensor:
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
        batch, _, time = r2.shape
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
        descriptors.extend((_temporal_descriptor(correlations), _temporal_descriptor(core.s_stem(s.unsqueeze(1)))))
        result = torch.cat(descriptors, dim=1)
        if result.shape[1] != 672:
            raise RuntimeError(f"expected original 672D complex_gram descriptor, got {result.shape[1]}")
        return result

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        image, z_power, n_power = self._gram_image_and_powers(samples)
        descriptors = [self.image_encoder(image)]
        if self.architecture_v19 == "hybrid":
            descriptors.extend((self._complex_gram_descriptor(self.complex_gram, samples), n_power))
        else:
            s = samples.mean(dim=-1).unsqueeze(1)
            descriptors.extend((_temporal_descriptor(self.s_stem(s)), z_power))
            if self.architecture_v19 == "gzn":
                descriptors.append(n_power)
        representation = torch.cat(descriptors, dim=1)
        expected = DESCRIPTOR_DIMS[self.architecture_v19]
        if representation.shape[1] != expected:
            raise RuntimeError(f"expected v19 descriptor size {expected}, got {representation.shape[1]}")
        return self.classifier(self.dropout(representation))


def parameter_count(architecture: str, resolution: int) -> int:
    with torch.random.fork_rng(devices=[]):
        model = C170TemporalGramClassifierV19("complex_gram", architecture, resolution)
        return sum(parameter.numel() for parameter in model.parameters())
