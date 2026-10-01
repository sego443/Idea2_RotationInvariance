"""Conditional gesture/angle heads on the existing four-channel backbones."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lstm_graph_fusion import FourChannelLSTMGraphFusion
from rotation_poc import TinyTemporalCNN


def angle_features(angles_deg: torch.Tensor) -> torch.Tensor:
    radians = torch.deg2rad(torch.remainder(angles_deg.float(), 360.0))
    return torch.stack((torch.cos(radians), torch.sin(radians)), dim=-1)


def decode_angle_deg(vectors: torch.Tensor) -> torch.Tensor:
    return torch.remainder(torch.rad2deg(torch.atan2(vectors[:, 1], vectors[:, 0])), 360.0)


def circular_error_deg(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.abs((np.asarray(predicted) - np.asarray(target) + 180.0) % 360.0 - 180.0)


def angle_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    errors = circular_error_deg(predicted, target)
    return {
        "mae_deg": float(np.mean(errors)),
        "median_ae_deg": float(np.median(errors)),
        "within_15_deg": float(np.mean(errors <= 15.0)),
        "within_30_deg": float(np.mean(errors <= 30.0)),
    }


def angle_vector_loss(predicted: torch.Tensor, target_deg: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(predicted, angle_features(target_deg))


class ConditionalRotationModel(nn.Module):
    """Waveform encoder with two separate heads, each seeing only its known condition."""

    def __init__(self, backbone: str, graph_config: dict[str, object] | None = None) -> None:
        super().__init__()
        self.backbone = backbone
        if backbone == "cnn":
            base = TinyTemporalCNN(in_channels=4, n_classes=4)
            self.features = base.features
            self.feature_head = nn.Sequential(*list(base.classifier.children())[:-1])
        elif backbone == "graph":
            self.graph = FourChannelLSTMGraphFusion(head="cnn", fusion="graph", **(graph_config or {}))
        else:
            raise ValueError(f"unknown backbone: {backbone}")
        self.angle_head = nn.Sequential(nn.Linear(32 + 4, 32), nn.GELU(), nn.Linear(32, 2))
        self.gesture_head = nn.Sequential(nn.Linear(32 + 2, 32), nn.GELU(), nn.Linear(32, 4))

    def encode(self, samples: torch.Tensor) -> torch.Tensor:
        if self.backbone == "cnn":
            return self.feature_head(self.features(samples))
        fused, _ = self.graph.fuse(samples)
        return self.graph.classifier[:-1](fused.transpose(1, 2))

    def predict_angle(self, samples: torch.Tensor, known_gesture: torch.Tensor) -> torch.Tensor:
        features = self.encode(samples)
        condition = F.one_hot(known_gesture.long(), num_classes=4).to(features.dtype)
        return self.angle_head(torch.cat((features, condition), dim=1))

    def predict_gesture(self, samples: torch.Tensor, known_angle_deg: torch.Tensor) -> torch.Tensor:
        features = self.encode(samples)
        condition = angle_features(known_angle_deg).to(features.dtype)
        return self.gesture_head(torch.cat((features, condition), dim=1))

    def predict_both(
        self, samples: torch.Tensor, known_gesture: torch.Tensor, known_angle_deg: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(samples)
        gesture_condition = F.one_hot(known_gesture.long(), num_classes=4).to(features.dtype)
        angle_condition = angle_features(known_angle_deg).to(features.dtype)
        return (
            self.angle_head(torch.cat((features, gesture_condition), dim=1)),
            self.gesture_head(torch.cat((features, angle_condition), dim=1)),
        )
