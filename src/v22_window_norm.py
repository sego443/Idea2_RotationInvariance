"""v22 window and amplitude-normalization variants for the v21 wear-calibrated classifier (exploratory ablation).

Each 500-sample raw window (2 s at 250 Hz, centred on the localization envelope peak) usually holds two bursts:
the outbound stroke and the return. Window variants keep a visible time span, band-pass only that span (so no
sample outside it reaches the model through the zero-phase filter), taper its cut edges and zero the rest. The
input shape stays [N,500,4], so the v21 encoders are unchanged.

  full             all 500 samples (v21 baseline)
  prefix_200/250/300  samples [0, P): a fixed-latency cut; the model has seen P/250 s of the window
  outbound_valley  samples [0, cut), cut = envelope minimum between the two strongest bursts (fallback: 250);
                   the cut is located with the whole window, so it is an oracle segmentation, not a causal one
  return_only      samples [250, 500): diagnostic for how much class information the return carries

Normalizations act on visible samples only and are applied after per-window, per-channel demeaning:

  shared        one train-fitted scalar (median |x| over all train samples/channels); v21 baseline
  window_global one scalar per window (RMS over time and channels); keeps the ratios between channels
  window_sensor one scalar per window and channel (RMS); keeps only each channel's waveform shape
  group_sensor  one scalar per wear group and channel (median |x| over all of that group's strokes); label-free and
                transductive (held-out groups use their own unlabeled strokes, including scored ones)
"""
from __future__ import annotations

from typing import Final

import numpy as np
from scipy import signal

SAMPLE_RATE: Final[int] = 250
WINDOW_SAMPLES: Final[int] = 500
CENTRE: Final[int] = WINDOW_SAMPLES // 2
TAPER_SAMPLES: Final[int] = 20
WINDOWS: Final[tuple[str, ...]] = ("full", "prefix_200", "prefix_250", "prefix_300", "outbound_valley", "return_only")
NORMS: Final[tuple[str, ...]] = ("shared", "window_global", "window_sensor", "group_sensor")
SCALE_FLOOR: Final[float] = float(np.finfo(np.float32).eps)


def _bandpass(values: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [0.5, 40], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return signal.sosfiltfilt(sos, values, axis=0)


def _envelope(window: np.ndarray) -> np.ndarray:
    filtered = _bandpass(window)
    filtered = filtered / np.maximum(np.median(np.abs(filtered), axis=0, keepdims=True), 1e-9)
    return signal.savgol_filter(np.sqrt(np.mean(filtered**2, axis=1)), 31, 2)


def valley_cut(window: np.ndarray) -> tuple[int, bool]:
    """Envelope minimum between the two strongest bursts; (CENTRE, False) when fewer than two bursts are found."""
    envelope = _envelope(window)
    peaks, _ = signal.find_peaks(envelope, prominence=0.25 * (envelope.max() - np.median(envelope)), distance=25)
    if len(peaks) < 2:
        return CENTRE, False
    first, second = np.sort(peaks[np.argsort(envelope[peaks])[::-1][:2]])
    return int(first + np.argmin(envelope[first:second + 1])), True


def visible_span(window: np.ndarray, variant: str) -> tuple[int, int, bool]:
    """(start, stop, two_bursts_found) of the samples a variant shows the model."""
    if variant == "full":
        return 0, WINDOW_SAMPLES, True
    if variant.startswith("prefix_"):
        return 0, int(variant.split("_")[1]), True
    if variant == "outbound_valley":
        cut, found = valley_cut(window)
        return 0, cut, found
    if variant == "return_only":
        return CENTRE, WINDOW_SAMPLES, True
    raise ValueError(f"unknown window variant: {variant}")


def apply_window(raw: np.ndarray, variant: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Band-pass each visible span on its own, demean it, taper cut edges and zero-pad to 500 samples.

    Returns (values [N,500,4] float32, mask [N,500] bool, statistics of the visible spans).
    """
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (WINDOW_SAMPLES, 4):
        raise ValueError(f"expected [N,{WINDOW_SAMPLES},4], got {raw.shape}")
    if variant not in WINDOWS:
        raise ValueError(f"unknown window variant: {variant}")
    values = np.zeros(raw.shape, dtype=np.float32)
    mask = np.zeros(raw.shape[:2], dtype=bool)
    spans, found = [], []
    for index, window in enumerate(raw):
        start, stop, two = visible_span(window, variant)
        segment = _bandpass(window[start:stop])
        segment = segment - segment.mean(axis=0, keepdims=True)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, TAPER_SAMPLES))
        if start > 0:
            segment[:TAPER_SAMPLES] *= ramp[:, None]
        if stop < WINDOW_SAMPLES:
            segment[-TAPER_SAMPLES:] *= ramp[::-1, None]
        values[index, start:stop] = segment
        mask[index, start:stop] = True
        spans.append((start, stop))
        found.append(two)
    lengths = np.array([stop - start for start, stop in spans])
    stats = {"variant": variant, "n": len(raw), "visible_samples_median": float(np.median(lengths)),
             "visible_samples_iqr": [float(v) for v in np.percentile(lengths, [25, 75])],
             "two_bursts_found_fraction": float(np.mean(found))}
    return values, mask, stats


def fit_shared_scale(values: np.ndarray, mask: np.ndarray) -> float:
    """Median |x| over every visible train sample and channel (v21's shared scale when nothing is masked)."""
    return max(float(np.median(np.abs(values[mask]))), SCALE_FLOOR)


def normalize(values: np.ndarray, mask: np.ndarray, kind: str, groups: np.ndarray,
              shared_scale: float | None = None) -> np.ndarray:
    """Apply a normalization to windowed values; `shared` requires the train-fitted scale."""
    values = np.asarray(values, dtype=np.float32)
    weights = mask[:, :, None].astype(np.float32)
    counts = np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
    if kind == "shared":
        if shared_scale is None or not np.isfinite(shared_scale) or shared_scale <= 0:
            raise ValueError("shared normalization requires a positive train-fitted scale")
        out = values / np.float32(shared_scale)
    elif kind == "window_global":
        rms = np.sqrt((values**2).sum(axis=(1, 2), keepdims=True) / (4 * counts))
        out = values / np.maximum(rms, SCALE_FLOOR)
    elif kind == "window_sensor":
        rms = np.sqrt((values**2).sum(axis=1, keepdims=True) / counts)
        out = values / np.maximum(rms, SCALE_FLOOR)
    elif kind == "group_sensor":
        out = np.empty_like(values)
        for group in np.unique(groups):
            members = groups == group
            visible = values[members][mask[members]]
            scale = np.maximum(np.median(np.abs(visible), axis=0), SCALE_FLOOR)
            out[members] = values[members] / scale.astype(np.float32)
    else:
        raise ValueError(f"unknown normalization: {kind}")
    out = out * weights
    if not np.isfinite(out).all():
        raise ValueError("normalized values contain non-finite entries")
    return out.astype(np.float32)
