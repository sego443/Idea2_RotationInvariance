from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


N_SENSORS = 4
N_CLASSES = 4
INPUT_LENGTH = 500
ENCODED_FEATURES = 32
ENCODED_LENGTH = 125
CONTRASTIVE_TEMPERATURE = 0.1


class CrossPositionMasks(NamedTuple):
    positive: torch.Tensor
    negative: torch.Tensor
    valid: torch.Tensor


class TemporalRingBlock(nn.Module):
    def __init__(self, features: int, dilation: int) -> None:
        super().__init__()
        self.self_conv = nn.Conv1d(features, features, kernel_size=1)
        self.plus_conv = nn.Conv1d(
            features,
            features,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.minus_conv = nn.Conv1d(
            features,
            features,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=features)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 4 or nodes.shape[1] != N_SENSORS:
            raise ValueError("ring nodes must have shape (batch, 4, features, time)")
        batch_size, node_count, features, time_steps = nodes.shape
        self_nodes = nodes.reshape(batch_size * node_count, features, time_steps)
        plus_nodes = torch.roll(nodes, shifts=-1, dims=1).reshape_as(self_nodes)
        minus_nodes = torch.roll(nodes, shifts=1, dims=1).reshape_as(self_nodes)
        updates = (
            self.self_conv(self_nodes)
            + self.plus_conv(plus_nodes)
            + self.minus_conv(minus_nodes)
        )
        updated = self.norm(self_nodes + F.gelu(updates))
        return updated.reshape(batch_size, node_count, features, time_steps)


class RingRepresentationClassifier(nn.Module):
    def __init__(self, variant: str, n_classes: int = N_CLASSES) -> None:
        super().__init__()
        if variant not in {"main", "classification_only", "slot_readout"}:
            raise ValueError(f"unsupported C170 ring variant: {variant}")
        if n_classes != N_CLASSES:
            raise ValueError("C170 ring classifier requires four direction classes")
        self.variant = variant
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.GELU(),
            nn.Conv1d(16, ENCODED_FEATURES, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(num_groups=8, num_channels=ENCODED_FEATURES),
            nn.GELU(),
            nn.Conv1d(
                ENCODED_FEATURES,
                ENCODED_FEATURES,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.GroupNorm(num_groups=8, num_channels=ENCODED_FEATURES),
            nn.GELU(),
        )
        self.ring_blocks = nn.ModuleList(
            [TemporalRingBlock(ENCODED_FEATURES, dilation) for dilation in (1, 2)]
        )
        if variant == "slot_readout":
            readout_features = N_SENSORS * ENCODED_FEATURES
            self.ordered_edge_encoder = None
        else:
            readout_features = ENCODED_FEATURES * 3
            self.ordered_edge_encoder = nn.Conv1d(
                ENCODED_FEATURES * 2,
                ENCODED_FEATURES,
                kernel_size=3,
                padding=1,
            )
        self.temporal_head = nn.Sequential(
            nn.Conv1d(readout_features, 64, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(64 * 8, 64),
            nn.GELU(),
        )
        self.classifier = nn.Linear(64, n_classes)
        self.projection_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 32),
        )

    def encode_nodes(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 3 or waveforms.shape[1:] != (INPUT_LENGTH, N_SENSORS):
            raise ValueError("waveforms must have shape (batch, 500, 4)")
        batch_size = waveforms.shape[0]
        sensor_inputs = waveforms.permute(0, 2, 1).reshape(
            batch_size * N_SENSORS,
            1,
            INPUT_LENGTH,
        )
        encoded = self.encoder(sensor_inputs)
        if encoded.shape[1:] != (ENCODED_FEATURES, ENCODED_LENGTH):
            raise RuntimeError(f"unexpected shared encoder output shape: {tuple(encoded.shape)}")
        nodes = encoded.reshape(
            batch_size,
            N_SENSORS,
            ENCODED_FEATURES,
            ENCODED_LENGTH,
        )
        for block in self.ring_blocks:
            nodes = block(nodes)
        return nodes

    def readout_sequence(self, nodes: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 4 or nodes.shape[1:] != (
            N_SENSORS,
            ENCODED_FEATURES,
            ENCODED_LENGTH,
        ):
            raise ValueError("nodes must have shape (batch, 4, 32, 125)")
        if self.variant == "slot_readout":
            return nodes.reshape(nodes.shape[0], N_SENSORS * ENCODED_FEATURES, ENCODED_LENGTH)

        ordered_nodes = torch.cat(
            (nodes, torch.roll(nodes, shifts=-1, dims=1)),
            dim=2,
        )
        ordered_edges = ordered_nodes.reshape(
            nodes.shape[0] * N_SENSORS,
            ENCODED_FEATURES * 2,
            ENCODED_LENGTH,
        )
        edge_features = F.gelu(self.ordered_edge_encoder(ordered_edges)).reshape(
            nodes.shape[0],
            N_SENSORS,
            ENCODED_FEATURES,
            ENCODED_LENGTH,
        )
        pooled_mean = nodes.mean(dim=1)
        pooled_max = nodes.max(dim=1).values
        pooled_ordered_edges = edge_features.mean(dim=1)
        return torch.cat((pooled_mean, pooled_max, pooled_ordered_edges), dim=1)

    def forward_with_projection(
        self,
        waveforms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.encode_nodes(waveforms)
        temporal_representation = self.temporal_head(self.readout_sequence(nodes))
        logits = self.classifier(temporal_representation)
        projection = F.normalize(self.projection_head(temporal_representation), dim=1)
        return logits, projection

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_projection(waveforms)
        return logits


def build_cross_position_masks(
    labels: torch.Tensor | np.ndarray,
    distance_mm: torch.Tensor | np.ndarray,
    blocks: torch.Tensor | np.ndarray,
) -> CrossPositionMasks:
    label_values = torch.as_tensor(labels, dtype=torch.long)
    distance_values = torch.as_tensor(
        distance_mm,
        dtype=torch.long,
        device=label_values.device,
    )
    block_values = torch.as_tensor(
        blocks,
        dtype=torch.long,
        device=label_values.device,
    )
    if label_values.ndim != 1 or not len(label_values):
        raise ValueError("contrastive labels must be a non-empty vector")
    if distance_values.shape != label_values.shape or block_values.shape != label_values.shape:
        raise ValueError("contrastive labels, distances, and blocks must have the same shape")
    if torch.any((label_values < 0) | (label_values >= N_CLASSES)):
        raise ValueError("contrastive labels must be four-class direction IDs")

    not_self = ~torch.eye(
        len(label_values),
        dtype=torch.bool,
        device=label_values.device,
    )
    same_label = label_values[:, None] == label_values[None, :]
    different_distance = distance_values[:, None] != distance_values[None, :]
    different_block = block_values[:, None] != block_values[None, :]
    positive = same_label & different_distance & different_block & not_self
    negative = (~same_label) & not_self
    valid = positive | negative
    positive_counts = positive.sum(dim=1)
    if torch.any(positive_counts == 0):
        raise ValueError("each contrastive anchor requires a cross-position positive")
    if not torch.all(different_distance[positive]) or not torch.all(different_block[positive]):
        raise RuntimeError("cross-position positives must use different distances and blocks")
    if not torch.equal(valid, positive | negative):
        raise RuntimeError("contrastive denominator contains an unclassified pair")
    return CrossPositionMasks(positive=positive, negative=negative, valid=valid)


def cross_position_supervised_contrastive_loss(
    projection: torch.Tensor,
    labels: torch.Tensor | np.ndarray,
    distance_mm: torch.Tensor | np.ndarray,
    blocks: torch.Tensor | np.ndarray,
    temperature: float = CONTRASTIVE_TEMPERATURE,
) -> torch.Tensor:
    if projection.ndim != 2 or projection.shape[0] == 0:
        raise ValueError("projection must have shape (batch, features)")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    masks = build_cross_position_masks(labels, distance_mm, blocks)
    if masks.valid.shape[0] != projection.shape[0]:
        raise ValueError("projection batch size does not match contrastive metadata")
    positive = masks.positive.to(device=projection.device)
    valid = masks.valid.to(device=projection.device)
    normalized = F.normalize(projection, dim=1)
    similarities = normalized @ normalized.T / temperature
    log_denominator = torch.logsumexp(
        similarities.masked_fill(~valid, float("-inf")),
        dim=1,
    )
    log_probabilities = similarities - log_denominator[:, None]
    positive_counts = positive.sum(dim=1)
    mean_positive_log_probability = (
        log_probabilities.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_counts.to(dtype=projection.dtype)
    )
    return -mean_positive_log_probability.mean()


def build_balanced_batches(
    indices: np.ndarray,
    labels: np.ndarray,
    distance_mm: np.ndarray,
    blocks: np.ndarray,
    seed: int,
) -> list[np.ndarray]:
    sample_indices = np.asarray(indices, dtype=np.int64)
    label_values = np.asarray(labels, dtype=np.int64)
    distances = np.asarray(distance_mm, dtype=np.int64)
    block_values = np.asarray(blocks, dtype=np.int64)
    if sample_indices.ndim != 1 or not len(sample_indices):
        raise ValueError("balanced sampler indices must be a non-empty vector")
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError("balanced sampler indices must be unique")
    if label_values.ndim != 1 or distances.shape != label_values.shape or block_values.shape != label_values.shape:
        raise ValueError("balanced sampler metadata must have matching one-dimensional shapes")
    if np.any(sample_indices < 0) or np.any(sample_indices >= len(label_values)):
        raise ValueError("balanced sampler indices are out of range")

    positions = np.sort(np.unique(distances[sample_indices]))
    classes = np.sort(np.unique(label_values[sample_indices]))
    if len(positions) != 11 or not np.array_equal(classes, np.arange(N_CLASSES)):
        raise ValueError("balanced C170 batches require 11 train positions and all four classes")

    groups: list[np.ndarray] = []
    for position in positions:
        position_indices = sample_indices[distances[sample_indices] == position]
        position_blocks = np.unique(block_values[position_indices])
        if len(position_blocks) != N_CLASSES:
            raise ValueError(f"position {int(position)} mm must contain four recording blocks")
        seen_labels: set[int] = set()
        for class_id in classes:
            group = position_indices[label_values[position_indices] == class_id]
            group_blocks = np.unique(block_values[group])
            if not len(group) or len(group_blocks) != 1:
                raise ValueError(
                    f"position {int(position)} mm and class {int(class_id)} must map to one recording block"
                )
            block = int(group_blocks[0])
            complete_block = np.flatnonzero(block_values == block)
            if not np.array_equal(np.sort(group), complete_block):
                raise ValueError(f"balanced sampler requires complete recording block {block}")
            if int(class_id) in seen_labels:
                raise ValueError(f"position {int(position)} mm repeats a training label")
            seen_labels.add(int(class_id))
            groups.append(group)

    rng = np.random.default_rng(seed)
    shuffled_groups = [rng.permutation(group) for group in groups]
    steps = max(len(group) for group in shuffled_groups)
    batches: list[np.ndarray] = []
    for step in range(steps):
        batch = np.asarray(
            [group[step % len(group)] for group in shuffled_groups],
            dtype=np.int64,
        )
        batches.append(rng.permutation(batch))
    return batches


def projection_diagnostics(
    projection: torch.Tensor | np.ndarray,
    labels: torch.Tensor | np.ndarray,
) -> dict[str, float | int]:
    vectors = np.asarray(torch.as_tensor(projection, dtype=torch.float64).cpu())
    label_values = np.asarray(torch.as_tensor(labels, dtype=torch.long).cpu())
    if vectors.ndim != 2 or not len(vectors) or label_values.shape != (len(vectors),):
        raise ValueError("projection diagnostics require aligned (samples, features) and label arrays")
    if not np.isfinite(vectors).all():
        raise ValueError("projection diagnostics received non-finite vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, np.finfo(np.float64).eps)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / len(centered)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total_variance = float(eigenvalues.sum())
    if total_variance <= np.finfo(np.float64).eps:
        effective_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0] / total_variance
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))

    class_centers = np.asarray(
        [vectors[label_values == class_id].mean(axis=0) for class_id in np.unique(label_values)]
    )
    if len(class_centers) < 2:
        raise ValueError("projection diagnostics require at least two training classes")
    pairwise = np.linalg.norm(class_centers[:, None, :] - class_centers[None, :, :], axis=2)
    pairwise_distances = pairwise[np.triu_indices(len(class_centers), k=1)]
    return {
        "projection_effective_rank": effective_rank,
        "projection_dimension": int(vectors.shape[1]),
        "class_center_distance_mean": float(pairwise_distances.mean()),
        "class_center_distance_min": float(pairwise_distances.min()),
        "class_center_distance_max": float(pairwise_distances.max()),
    }
