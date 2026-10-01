"""Core utilities for the round-1 rotation-invariance feasibility probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import copy
import re

import numpy as np
from scipy import signal as scipy_signal
import torch
from torch import nn


MOVEMENT_RE = re.compile(
    r"_B(?P<block>\d{3})_A(?P<angle>\d{3})_"
    r"(?P<gesture>UP|DOWN|LEFT|RIGHT)_R5\.txt$"
)
TRAIN_ANGLES = {0, 60, 120, 180, 240, 300}
VAL_ANGLES = {30, 150, 270}
TEST_ANGLES = {90, 210, 330}


@dataclass(frozen=True)
class SegmentationResult:
    strokes: np.ndarray
    rest_windows: np.ndarray
    centers: np.ndarray
    period_samples: float
    score: float


@dataclass
class TrainingResult:
    model: nn.Module
    best_epoch: int
    best_val_loss: float
    best_val_macro_f1: float
    train_predictions: np.ndarray
    val_predictions: np.ndarray


def read_openbci(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Read EXG 0-3 and the acquisition metadata from an OpenBCI text file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    sample_rate = None
    declared_channels = None
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("%Sample Rate ="):
            sample_rate = int(re.search(r"(\d+)\s*Hz", line).group(1))
        elif line.startswith("%Number of channels ="):
            declared_channels = int(line.rsplit("=", 1)[1].strip())
        elif line.startswith("Sample Index"):
            header_index = index
            break
    if sample_rate is None or declared_channels is None or header_index is None:
        raise ValueError(f"missing OpenBCI metadata or header in {path}")

    reader = csv.reader(lines[header_index:], skipinitialspace=True)
    header = [item.strip() for item in next(reader)]
    required = [f"EXG Channel {channel}" for channel in range(4)]
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"missing required columns in {path}: {missing}")
    indices = [header.index(name) for name in required]
    rows: list[list[float]] = []
    for row in reader:
        try:
            rows.append([float(row[index].strip()) for index in indices])
        except (IndexError, ValueError):
            continue
    if not rows:
        raise ValueError(f"no valid EXG rows in {path}")
    samples = np.asarray(rows, dtype=np.float64)
    return samples, {
        "sample_rate_hz": sample_rate,
        "declared_channels": declared_channels,
        "valid_rows": len(samples),
    }


def parse_movement_filename(path: Path) -> tuple[int, int, str] | None:
    """Parse an accepted round-1 movement filename."""
    match = MOVEMENT_RE.search(path.name)
    if match is None:
        return None
    return (
        int(match.group("block")),
        int(match.group("angle")),
        match.group("gesture").lower(),
    )


def select_recordings(session_dir: Path, expected_blocks: set[int]) -> list[Path]:
    """Select exactly one accepted recording for every requested movement block."""
    by_block: dict[int, Path] = {}
    for path in sorted(session_dir.glob("*.txt")):
        parsed = parse_movement_filename(path)
        if parsed is None:
            continue
        block, _, _ = parsed
        if block not in expected_blocks:
            continue
        if block in by_block:
            raise ValueError(f"duplicate accepted block B{block:03d}")
        by_block[block] = path

    missing = sorted(expected_blocks - set(by_block))
    if missing:
        formatted = ", ".join(f"B{block:03d}" for block in missing)
        raise ValueError(f"missing accepted blocks: {formatted}")
    return [by_block[block] for block in sorted(expected_blocks)]


def angle_split(angle: int) -> str:
    if angle in TRAIN_ANGLES:
        return "train"
    if angle in VAL_ANGLES:
        return "val"
    if angle in TEST_ANGLES:
        return "test"
    raise ValueError(f"angle {angle} is outside the frozen round-1 split")


def _fixed_window(samples: np.ndarray, center: int, length: int) -> np.ndarray:
    start = center - length // 2
    end = start + length
    if start < 0 or end > len(samples):
        raise ValueError("requested window extends beyond recording")
    return samples[start:end]


def _movement_energy(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    sos = scipy_signal.butter(
        3, [0.5, 40.0], btype="bandpass", fs=sample_rate, output="sos"
    )
    filtered = scipy_signal.sosfiltfilt(sos, samples, axis=0)
    center = np.median(filtered, axis=0)
    mad = np.median(np.abs(filtered - center), axis=0) * 1.4826
    floor = np.percentile(np.abs(filtered - center), 35, axis=0)
    scale = np.maximum(mad, floor + 1e-6)
    standardized = (filtered - center) / scale
    energy = np.sqrt(np.mean(standardized**2, axis=1))
    width = max(11, int(round(sample_rate * 0.24)) | 1)
    return scipy_signal.savgol_filter(energy, width, 2)


def segment_five_strokes(
    samples: np.ndarray,
    sample_rate: int = 250,
    window_samples: int = 500,
) -> SegmentationResult:
    """Find five approximately periodic action bursts and matching pre-action rest."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError(f"expected samples shaped (time, 4), got {samples.shape}")
    minimum = 5 * window_samples + 2 * sample_rate
    if len(samples) < minimum:
        raise ValueError(f"recording is too short for five strokes: {len(samples)} samples")

    energy = _movement_energy(samples, sample_rate)
    half_search = int(round(0.55 * sample_rate))
    first_min = max(window_samples // 2, int(round(1.8 * sample_rate)))
    first_max = min(int(round(5.0 * sample_rate)), len(samples) - window_samples // 2)
    period_min = int(round(2.2 * sample_rate))
    period_max = int(round(4.8 * sample_rate))
    best: tuple[float, np.ndarray, float] | None = None

    for first in range(first_min, first_max + 1, 10):
        max_period_from_length = (len(samples) - window_samples // 2 - first) // 4
        for period in range(period_min, min(period_max, max_period_from_length) + 1, 10):
            centers = []
            peak_values = []
            timing_offsets = []
            valid = True
            for idx in range(5):
                target = first + idx * period
                lo = max(window_samples // 2, target - half_search)
                hi = min(len(samples) - window_samples // 2, target + half_search + 1)
                if hi <= lo:
                    valid = False
                    break
                center = lo + int(np.argmax(energy[lo:hi]))
                centers.append(center)
                peak_values.append(float(energy[center]))
                timing_offsets.append(abs(center - target) / half_search)
            if not valid or np.min(np.diff(centers)) < window_samples:
                continue
            local_floor = float(np.median(energy[first_min : len(samples) - window_samples // 2]))
            strength = float(np.mean(np.log1p(np.asarray(peak_values) / max(local_floor, 1e-6))))
            regularity = float(np.std(np.diff(centers)) / max(np.mean(np.diff(centers)), 1.0))
            score = strength - 0.55 * regularity - 0.08 * float(np.mean(timing_offsets))
            if best is None or score > best[0]:
                best = (score, np.asarray(centers, dtype=int), float(period))

    if best is None:
        raise ValueError("could not locate five regular action bursts")

    score, centers, period = best
    strokes = np.stack([_fixed_window(samples, int(c), window_samples) for c in centers])
    rest_length = sample_rate
    rest_centers = centers - np.minimum(
        int(round(1.35 * sample_rate)), np.rint(np.r_[period, np.diff(centers)] / 2).astype(int)
    )
    if np.any(rest_centers - rest_length // 2 < 0):
        raise ValueError("insufficient quiet baseline before first detected stroke")
    rest = np.stack([_fixed_window(samples, int(c), rest_length) for c in rest_centers])
    return SegmentationResult(
        strokes=strokes.astype(np.float32),
        rest_windows=rest.astype(np.float32),
        centers=centers,
        period_samples=period,
        score=float(score),
    )


def harmonic_invariants(samples: np.ndarray) -> np.ndarray:
    """Return three C4-invariant ring features per time sample."""
    samples = np.asarray(samples)
    if samples.ndim != 3 or samples.shape[2] != 4:
        raise ValueError(f"expected (batch, time, 4), got {samples.shape}")
    spectrum = np.fft.rfft(samples, axis=2)
    result = np.stack(
        [spectrum[:, :, 0].real / 4.0, np.abs(spectrum[:, :, 1]) / 2.0, np.abs(spectrum[:, :, 2]) / 4.0],
        axis=2,
    )
    return result.astype(np.float32)


def filter_windows(samples: np.ndarray, sample_rate: int = 250) -> np.ndarray:
    """Band-limit windows while retaining their original time/channel layout."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[2] != 4:
        raise ValueError(f"expected (batch, time, 4), got {samples.shape}")
    sos = scipy_signal.butter(
        3, [0.5, 40.0], btype="bandpass", fs=sample_rate, output="sos"
    )
    return scipy_signal.sosfiltfilt(sos, samples, axis=1).astype(np.float32)


def calibrate_windows(
    samples: np.ndarray, scale: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each trial's DC level and apply train-derived physical-sensor scales."""
    samples = np.asarray(samples, dtype=np.float32)
    centered = samples - np.mean(samples, axis=1, keepdims=True)
    if scale is None:
        scale = np.median(np.abs(centered), axis=(0, 1))
        scale = np.maximum(scale, np.finfo(np.float32).eps).astype(np.float32)
    else:
        scale = np.asarray(scale, dtype=np.float32)
        if scale.shape != (4,):
            raise ValueError(f"expected four channel scales, got {scale.shape}")
    return (centered / scale).astype(np.float32), scale


class TinyTemporalCNN(nn.Module):
    """Small temporal classifier sized for the 120-sample training split."""

    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 12, kernel_size=15, padding=7),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(12, 20, kernel_size=9, padding=4),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(20, 20, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 8, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(samples))


class OrbitCNN(nn.Module):
    """Make a classifier exactly invariant to the four integer ring shifts."""

    def __init__(self, classifier: nn.Module, aggregation: str = "mean_logits") -> None:
        super().__init__()
        if aggregation not in {"mean_logits", "mean_probabilities"}:
            raise ValueError(f"unsupported orbit aggregation: {aggregation}")
        self.classifier = classifier
        self.aggregation = aggregation

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        logits = [self.classifier(torch.roll(samples, shifts=shift, dims=1)) for shift in range(4)]
        if self.aggregation == "mean_logits":
            return torch.stack(logits, dim=0).mean(dim=0)
        probabilities = torch.stack(
            [torch.softmax(values, dim=1) for values in logits], dim=0
        ).mean(dim=0)
        return torch.log(probabilities)


def _make_model(
    model_kind: str, in_channels: int, n_classes: int, model_config: dict[str, object] | None = None
) -> nn.Module:
    if model_kind in {"lstm_graph_mlp", "lstm_graph_cnn", "lstm_mean_mlp", "lstm_mean_cnn"}:
        if in_channels != 4 or n_classes != 4:
            raise ValueError("four-channel LSTM fusion requires four sensors and four classes")
        from lstm_graph_fusion import FourChannelLSTMGraphFusion

        head = "mlp" if model_kind.endswith("mlp") else "cnn"
        fusion = "mean" if "_mean_" in model_kind else "graph"
        return FourChannelLSTMGraphFusion(head=head, fusion=fusion, **(model_config or {}))
    base = TinyTemporalCNN(in_channels=in_channels, n_classes=n_classes)
    if model_kind in {"raw", "harmonic"}:
        return base
    if model_kind in {"orbit", "orbit_prob"}:
        if in_channels != 4:
            raise ValueError("orbit model requires four ring channels")
        aggregation = "mean_logits" if model_kind == "orbit" else "mean_probabilities"
        return OrbitCNN(base, aggregation=aggregation)
    raise ValueError(f"unknown model kind: {model_kind}")


def _predict(model: nn.Module, samples: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(samples, dtype=np.float32)).permute(0, 2, 1)
    model.eval()
    with torch.no_grad():
        return model(tensor).argmax(dim=1).cpu().numpy()


def train_temporal_model(
    train_samples: np.ndarray,
    train_labels: np.ndarray,
    val_samples: np.ndarray,
    val_labels: np.ndarray,
    model_kind: str,
    seed: int,
    max_epochs: int = 220,
    patience: int = 35,
    model_config: dict[str, object] | None = None,
) -> TrainingResult:
    """Train one small classifier with validation-only early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_samples = np.asarray(train_samples, dtype=np.float32)
    val_samples = np.asarray(val_samples, dtype=np.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    if train_samples.ndim != 3 or val_samples.ndim != 3:
        raise ValueError("training arrays must be shaped (sample, time, channel)")
    if train_samples.shape[2] != val_samples.shape[2]:
        raise ValueError("train and validation channel counts differ")
    n_classes = int(max(train_labels.max(), val_labels.max())) + 1
    model = _make_model(model_kind, train_samples.shape[2], n_classes, model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    train_x = torch.from_numpy(train_samples).permute(0, 2, 1)
    train_y = torch.from_numpy(train_labels)
    val_x = torch.from_numpy(val_samples).permute(0, 2, 1)
    val_y = torch.from_numpy(val_labels)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    stale_epochs = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())
            val_predictions = val_logits.argmax(dim=1).cpu().numpy()
        val_f1 = float(macro_metrics(val_labels, val_predictions, n_classes)["macro_f1"])
        improved = val_f1 > best_val_f1 + 1e-12 or (
            abs(val_f1 - best_val_f1) <= 1e-12 and val_loss < best_val_loss - 1e-6
        )
        if improved:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_f1 = val_f1
            best_val_loss = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_macro_f1=best_val_f1,
        train_predictions=_predict(model, train_samples),
        val_predictions=_predict(model, val_samples),
    )


def predict_temporal_model(model: nn.Module, samples: np.ndarray) -> np.ndarray:
    return _predict(model, samples)


def macro_metrics(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    recalls: list[float] = []
    f1s: list[float] = []
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for truth, pred in zip(labels, predictions, strict=True):
        confusion[truth, pred] += 1
    for class_id in range(n_classes):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_recall": recalls,
        "prediction_counts": np.bincount(predictions, minlength=n_classes).tolist(),
        "confusion_matrix": confusion.tolist(),
    }
