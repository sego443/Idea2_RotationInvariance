"""Independent sensor LSTMs with graph attention over four wrist channels."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourChannelLSTMGraphFusion(nn.Module):
    """Produce one fused time series from four independently encoded sensors."""

    def __init__(
        self,
        head: str = "cnn",
        fusion: str = "graph",
        hidden_size: int = 12,
        num_layers: int = 1,
        graph_heads: int = 1,
        graph_topology: str = "full",
    ) -> None:
        super().__init__()
        if head not in {"cnn", "mlp"}:
            raise ValueError(f"unknown classifier head: {head}")
        if fusion not in {"graph", "mean"}:
            raise ValueError(f"unknown fusion: {fusion}")
        if graph_topology not in {"full", "ring"}:
            raise ValueError(f"unknown graph topology: {graph_topology}")
        if (hidden_size * 2) % graph_heads:
            raise ValueError("LSTM feature dimension must be divisible by graph heads")
        self.fusion = fusion
        self.feature_dim = hidden_size * 2
        self.graph_heads = graph_heads
        self.graph_topology = graph_topology
        self.encoders = nn.ModuleList(
            [nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True) for _ in range(4)]
        )
        if fusion == "graph":
            self.query = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
            self.key = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
            self.value = nn.Linear(self.feature_dim, self.feature_dim, bias=False)
            self.node_norm = nn.LayerNorm(self.feature_dim)
            self.node_score = nn.Linear(self.feature_dim, 1)
        if head == "cnn":
            self.classifier = nn.Sequential(
                nn.Conv1d(self.feature_dim, 24, kernel_size=9, padding=4),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Conv1d(24, 24, kernel_size=5, padding=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
                nn.Linear(24 * 8, 32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 4),
            )
        else:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(8),
                nn.Flatten(),
                nn.Linear(self.feature_dim * 8, 32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 4),
            )

    def fuse(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if samples.ndim != 3 or samples.shape[1] != 4:
            raise ValueError(f"expected (batch, 4, time), got {tuple(samples.shape)}")
        encoded = torch.stack(
            [encoder(samples[:, channel, :, None])[0] for channel, encoder in enumerate(self.encoders)],
            dim=2,
        )  # batch, time, channel, feature
        if self.fusion == "mean":
            attention = encoded.new_full((*encoded.shape[:3], 4), 0.25)
            return encoded.mean(dim=2), attention

        batch, time, channels, _ = encoded.shape
        per_head = self.feature_dim // self.graph_heads
        query = self.query(encoded).reshape(batch, time, channels, self.graph_heads, per_head).transpose(2, 3)
        key = self.key(encoded).reshape(batch, time, channels, self.graph_heads, per_head).transpose(2, 3)
        value = self.value(encoded).reshape(batch, time, channels, self.graph_heads, per_head).transpose(2, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(per_head)
        if self.graph_topology == "ring":
            opposite = torch.eye(4, device=scores.device, dtype=torch.bool).roll(2, dims=1)
            scores = scores.masked_fill(opposite, torch.finfo(scores.dtype).min)
        head_attention = torch.softmax(scores, dim=-1)
        messages = torch.matmul(head_attention, value).transpose(2, 3).reshape(batch, time, channels, self.feature_dim)
        attention = head_attention.mean(dim=2)
        nodes = self.node_norm(encoded + messages)
        weights = torch.softmax(self.node_score(nodes), dim=2)
        fused = (weights * nodes).sum(dim=2)
        return fused, attention

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        fused, _ = self.fuse(samples)
        return self.classifier(fused.transpose(1, 2))
