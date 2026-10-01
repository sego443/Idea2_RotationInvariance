from pathlib import Path

import numpy as np
import pytest
import torch

from rotation_poc import (
    OrbitCNN,
    TinyTemporalCNN,
    angle_split,
    calibrate_windows,
    filter_windows,
    harmonic_invariants,
    macro_metrics,
    parse_movement_filename,
    read_openbci,
    segment_five_strokes,
    select_recordings,
    train_temporal_model,
)


def test_parse_movement_filename_extracts_block_angle_and_gesture() -> None:
    parsed = parse_movement_filename(
        Path("OpenBCI-RAW-2026-08-27_14-39-34_B001_A000_UP_R5.txt")
    )

    assert parsed == (1, 0, "up")


def test_read_openbci_keeps_only_first_four_exg_channels(tmp_path: Path) -> None:
    source = tmp_path / "recording.txt"
    source.write_text(
        "%OpenBCI Raw EXG Data\n"
        "%Number of channels = 8\n"
        "%Sample Rate = 250 Hz\n"
        "Sample Index, EXG Channel 0, EXG Channel 1, EXG Channel 2, EXG Channel 3, "
        "EXG Channel 4, Timestamp\n"
        "0, 10, 11, 12, 13, 99, 1.0\n"
        "1, 20, 21, 22, 23, 98, 1.004\n",
        encoding="utf-8",
    )

    samples, metadata = read_openbci(source)

    np.testing.assert_array_equal(samples, [[10, 11, 12, 13], [20, 21, 22, 23]])
    assert metadata == {"sample_rate_hz": 250, "declared_channels": 8, "valid_rows": 2}


def test_select_recordings_ignores_rejected_and_requires_unique_blocks(tmp_path: Path) -> None:
    accepted = tmp_path / "OpenBCI-RAW-x_B001_A000_UP_R5.txt"
    accepted.touch()
    (tmp_path / "OpenBCI-RAW-y_B001_A000_UP_R5.txt.rejected").touch()

    selected = select_recordings(tmp_path, expected_blocks={1})

    assert selected == [accepted]


def test_select_recordings_rejects_duplicate_accepted_block(tmp_path: Path) -> None:
    (tmp_path / "OpenBCI-RAW-x_B001_A000_UP_R5.txt").touch()
    (tmp_path / "OpenBCI-RAW-y_B001_A000_UP_R5.txt").touch()

    with pytest.raises(ValueError, match="duplicate accepted block B001"):
        select_recordings(tmp_path, expected_blocks={1})


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0, "train"), (60, "train"), (30, "val"), (150, "val"), (90, "test"), (330, "test")],
)
def test_angle_split_matches_frozen_collection_plan(angle: int, expected: str) -> None:
    assert angle_split(angle) == expected


def test_segment_five_strokes_recovers_regular_bursts_without_edge_artifact() -> None:
    rng = np.random.default_rng(7)
    n_samples = 5_500
    signal = rng.normal(0.0, 0.02, size=(n_samples, 4))
    true_centers = np.array([800, 1_650, 2_500, 3_350, 4_200])
    shape = np.sin(np.linspace(0, 3 * np.pi, 180)) * np.hanning(180)
    for center in true_centers:
        signal[center - 90 : center + 90, 0] += 8.0 * shape
        signal[center - 90 : center + 90, 1] -= 5.0 * shape
    signal[:80] += 20.0  # startup artifact must not become a stroke

    result = segment_five_strokes(signal, sample_rate=250, window_samples=500)

    assert result.strokes.shape == (5, 500, 4)
    assert result.rest_windows.shape == (5, 250, 4)
    assert np.max(np.abs(result.centers - true_centers)) <= 35
    assert np.all(result.rest_windows.std(axis=(1, 2)) < 0.08)


def test_harmonic_invariants_are_unchanged_by_integer_channel_roll() -> None:
    rng = np.random.default_rng(11)
    samples = rng.normal(size=(3, 120, 4)).astype(np.float32)

    original = harmonic_invariants(samples)
    shifted = harmonic_invariants(np.roll(samples, shift=1, axis=2))

    np.testing.assert_allclose(original, shifted, rtol=1e-5, atol=1e-5)


def test_filter_and_calibrate_uses_training_scale_for_all_splits() -> None:
    time = np.arange(500) / 250.0
    wave = np.sin(2 * np.pi * 6 * time)
    train = np.stack([np.column_stack([wave, 2 * wave, 3 * wave, 4 * wave])] * 3)
    held_out = train[:1] * 2

    filtered_train = filter_windows(train, sample_rate=250)
    normalized_train, scale = calibrate_windows(filtered_train)
    normalized_held_out, reused = calibrate_windows(filter_windows(held_out, 250), scale=scale)

    np.testing.assert_array_equal(scale, reused)
    assert np.allclose(np.median(np.abs(normalized_train), axis=(0, 1)), 1.0, atol=0.08)
    assert np.allclose(np.median(np.abs(normalized_held_out), axis=(0, 1)), 2.0, atol=0.16)


def test_orbit_cnn_logits_are_invariant_to_channel_roll() -> None:
    torch.manual_seed(3)
    model = OrbitCNN(TinyTemporalCNN(in_channels=4, n_classes=4)).eval()
    samples = torch.randn(5, 4, 250)

    original = model(samples)
    shifted = model(torch.roll(samples, shifts=1, dims=1))

    torch.testing.assert_close(original, shifted, rtol=1e-5, atol=1e-6)


def test_train_temporal_model_learns_trivial_four_class_signal() -> None:
    labels = np.tile(np.arange(4), 8)
    samples = np.zeros((32, 80, 4), dtype=np.float32)
    for row, label in enumerate(labels):
        samples[row, :, label] = 1.0

    result = train_temporal_model(
        samples[:24],
        labels[:24],
        samples[24:],
        labels[24:],
        model_kind="raw",
        seed=5,
        max_epochs=80,
        patience=25,
    )

    assert result.best_epoch >= 0
    assert np.mean(result.train_predictions == labels[:24]) >= 0.95
    assert result.val_predictions.shape == (8,)


def test_macro_metrics_reports_perfect_balanced_predictions() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])

    metrics = macro_metrics(labels, labels, n_classes=4)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class_recall"] == [1.0, 1.0, 1.0, 1.0]
    assert metrics["prediction_counts"] == [2, 2, 2, 2]
