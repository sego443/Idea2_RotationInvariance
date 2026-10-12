"""v21 wear-calibrated direction classifier: encoders, prototype heads, calibration and augmentation.

See docs/designs/2026-10-12-v21-wear-calibrated-classifier.md. The encoder maps a preprocessed window [B,T,4] to an
L2-normalized embedding. Learned class prototypes give ZERO-mode logits; CAL mode shrinks per-wear calibration means
toward them; UNSUP mode refines them transductively with unlabeled strokes from the same wear.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.c170_complex_orbit_model_v11 import ComplexPowerGramClassifier, LOG_RMS_EPSILON, _temporal_descriptor
from src.c170_harmonic_phase_model_v4 import _DilatedResidualBlock

BACKBONES: Final[tuple[str, ...]] = ("b1_complex_gram", "b2_signed_cnn")
N_CLASSES: Final[int] = 4
EMBEDDING_DIM: Final[int] = 64
COSINE_SCALE: Final[float] = 16.0


class ComplexGramEncoder(nn.Module):
    """B1: the v11 complex-Gram descriptor (C4-invariant by construction) with an embedding projection."""

    def __init__(self) -> None:
        super().__init__()
        self.descriptor = ComplexPowerGramClassifier("complex_gram")
        self.descriptor.classifier = nn.Identity()
        self.projection = nn.Linear(672, EMBEDDING_DIM)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.projection(self.descriptor(samples))


class SignedCnnEncoder(nn.Module):
    """B2: signed four-channel temporal CNN without rotation pooling."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, 24, kernel_size=15, stride=2, padding=7, bias=False),
                                  nn.GroupNorm(4, 24), nn.SiLU())
        self.blocks = nn.Sequential(_DilatedResidualBlock(1), _DilatedResidualBlock(2), _DilatedResidualBlock(4))
        self.dropout = nn.Dropout(p=0.2)
        self.projection = nn.Linear(196, EMBEDDING_DIM)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3 or samples.shape[-1] != 4:
            raise ValueError(f"expected [B,T,4], got {tuple(samples.shape)}")
        temporal = self.blocks(self.stem(torch.asinh(samples).transpose(1, 2)))
        rms = torch.log(torch.sqrt(samples.square().mean(dim=1) + LOG_RMS_EPSILON))
        return self.projection(self.dropout(torch.cat((_temporal_descriptor(temporal), rms), dim=1)))


class WearCalibratedClassifier(nn.Module):
    def __init__(self, backbone: str) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown v21 backbone: {backbone}")
        self.backbone = backbone
        self.encoder = ComplexGramEncoder() if backbone == "b1_complex_gram" else SignedCnnEncoder()
        self.prototypes = nn.Parameter(torch.randn(N_CLASSES, EMBEDDING_DIM) * 0.1)

    def embed(self, samples: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(samples), dim=1)

    def global_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes, dim=1)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return cosine_logits(self.embed(samples), self.global_prototypes())


def cosine_logits(embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    return COSINE_SCALE * embeddings @ prototypes.T


def calibrated_prototypes(global_prototypes: torch.Tensor, support: torch.Tensor, support_labels: torch.Tensor,
                          tau: float) -> torch.Tensor:
    """p_c = normalize(sum of support embeddings of class c + tau * g_c); classes without support keep g_c."""
    if tau < 0:
        raise ValueError("tau must be non-negative")
    sums = torch.zeros_like(global_prototypes).index_add_(0, support_labels, support)
    counts = torch.zeros(len(global_prototypes), device=support.device).index_add_(
        0, support_labels, torch.ones(len(support_labels), device=support.device))
    calibrated = F.normalize(sums + tau * global_prototypes, dim=1)
    return torch.where((counts > 0)[:, None], calibrated, global_prototypes)


def transductive_prototypes(global_prototypes: torch.Tensor, unlabeled: torch.Tensor, tau: float,
                            iterations: int = 2) -> torch.Tensor:
    """Soft k-means refinement of prototypes with unlabeled embeddings from one wear."""
    prototypes = global_prototypes
    for _ in range(iterations):
        weights = torch.softmax(cosine_logits(unlabeled, prototypes), dim=1)
        prototypes = F.normalize(weights.T @ unlabeled + tau * global_prototypes, dim=1)
    return prototypes


def episode_loss(model: WearCalibratedClassifier, samples: torch.Tensor, labels: torch.Tensor, groups: np.ndarray,
                 rng: np.random.Generator, shots: int, tau: float, max_query: int) -> torch.Tensor | None:
    """Prototypical wear-adaptation loss: per group, support = `shots` strokes per class, query = others."""
    embeddings = model.embed(samples)
    prototypes = model.global_prototypes()
    losses = []
    labels_np = labels.cpu().numpy()
    for group in np.unique(groups):
        support, query = [], []
        for label in range(N_CLASSES):
            members = np.flatnonzero((groups == group) & (labels_np == label))
            if len(members) <= shots:
                break
            members = rng.permutation(members)
            support.extend(members[:shots])
            query.extend(members[shots:])
        else:
            query = rng.permutation(np.asarray(query))[:max_query]
            support_index = torch.as_tensor(np.asarray(support), device=samples.device)
            query_index = torch.as_tensor(query, device=samples.device)
            calibrated = calibrated_prototypes(prototypes, embeddings[support_index], labels[support_index], tau)
            losses.append(F.cross_entropy(cosine_logits(embeddings[query_index], calibrated), labels[query_index]))
    return torch.stack(losses).mean() if losses else None


def support_query_loss(model: WearCalibratedClassifier, samples: torch.Tensor, labels: torch.Tensor,
                       episodes: list[tuple[np.ndarray, np.ndarray]], tau: float) -> torch.Tensor:
    """Mean query cross-entropy over sub-episodes whose support and query positions index into `samples`."""
    embeddings = model.embed(samples)
    prototypes = model.global_prototypes()
    losses = []
    for support, query in episodes:
        support_t = torch.as_tensor(support, device=samples.device)
        query_t = torch.as_tensor(query, device=samples.device)
        calibrated = calibrated_prototypes(prototypes, embeddings[support_t], labels[support_t], tau)
        losses.append(F.cross_entropy(cosine_logits(embeddings[query_t], calibrated), labels[query_t]))
    return torch.stack(losses).mean()


def augment(samples: torch.Tensor, generator: torch.Generator, max_shift: int = 50, warp: float = 0.10,
            gain_db: float = 6.0, noise: float = 0.01) -> torch.Tensor:
    """Per-channel/global gain, zero-padded time shift, time warp and Gaussian noise on [B,T,4]."""
    batch, time, channels = samples.shape
    device = samples.device

    def uniform(*shape):
        return torch.rand(*shape, generator=generator).to(device) * 2 - 1

    log_gain = (uniform(batch, 1, channels) + uniform(batch, 1, 1)) * (gain_db / 20 * np.log(10))
    values = samples * torch.exp(log_gain)
    factors = 1 + warp * uniform(batch)
    shifts = torch.round(uniform(batch) * max_shift)
    centre = (time - 1) / 2
    grid_time = (torch.arange(time, device=device)[None, :] - centre) * factors[:, None] + centre - shifts[:, None]
    grid_x = grid_time / (time - 1) * 2 - 1
    grid = torch.stack((grid_x, torch.zeros_like(grid_x)), dim=-1)[:, None, :, :]
    warped = F.grid_sample(values.transpose(1, 2)[:, :, None, :], grid, align_corners=True, padding_mode="zeros")
    values = warped[:, :, 0, :].transpose(1, 2)
    rms = torch.sqrt(values.square().mean(dim=(1, 2), keepdim=True) + 1e-12)
    return values + noise * rms * torch.randn(values.shape, generator=generator).to(device)
