import importlib

import numpy as np
import pytest


def experiment():
    try:
        return importlib.import_module("scripts.runners.run_conditional_experiment")
    except ModuleNotFoundError as error:
        pytest.fail(f"conditional experiment runner is missing: {error}")


def test_selection_uses_validation_score_not_test_score() -> None:
    runner = experiment()
    trials = [
        {"config": {"hidden_size": 8}, "mean_val_score": 0.6, "mean_val_loss": 0.8, "test_score": 0.1},
        {"config": {"hidden_size": 16}, "mean_val_score": 0.5, "mean_val_loss": 0.1, "test_score": 1.0},
    ]

    assert runner.select_trial(trials)["config"]["hidden_size"] == 8


def test_shuffled_control_breaks_waveform_label_pairing_only_on_training_split() -> None:
    runner = experiment()
    gestures = np.tile(np.arange(4), 8)
    angles = np.repeat(np.arange(8) * 30, 4)
    train_mask = np.arange(32) < 24

    shuffled_gestures, shuffled_angles = runner.permute_train_label_pairs(
        gestures, angles, train_mask, seed=17
    )

    assert np.array_equal(shuffled_gestures[~train_mask], gestures[~train_mask])
    assert np.array_equal(shuffled_angles[~train_mask], angles[~train_mask])
    assert sorted(zip(shuffled_gestures[train_mask], shuffled_angles[train_mask])) == sorted(
        zip(gestures[train_mask], angles[train_mask])
    )
    assert np.any(shuffled_angles[train_mask] != angles[train_mask])


@pytest.mark.parametrize("mode", ["angle", "gesture"])
def test_condition_shift_diagnostic_reports_changed_condition(mode: str) -> None:
    runner = experiment()
    model = runner.ConditionalRotationModel("cnn")
    samples = np.random.default_rng(2).normal(size=(8, 32, 4)).astype(np.float32)
    gestures = np.tile(np.arange(4), 2)
    angles = np.tile(np.array([0.0, 90.0, 180.0, 270.0]), 2)

    result = runner.evaluate_condition_shift(model, samples, gestures, angles, mode)

    if mode == "angle":
        assert 0 <= result["correct_mae_deg"] <= 180
        assert 0 <= result["wrong_gesture_mae_deg"] <= 180
    else:
        assert 0 <= result["prediction_agreement"] <= 1
        assert 0 <= result["correct_macro_f1"] <= 1
        assert 0 <= result["shifted_angle_macro_f1"] <= 1


@pytest.mark.parametrize("mode", ["angle", "gesture", "joint"])
def test_each_training_mode_produces_its_own_evaluation_metrics(mode: str) -> None:
    runner = experiment()
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(16, 32, 4)).astype(np.float32)
    gestures = np.tile(np.arange(4), 4)
    angles = np.tile(np.array([0.0, 60.0, 120.0, 180.0]), 4)
    masks = {"train": np.arange(16) < 12, "val": np.arange(16) >= 12}

    result = runner.fit_model(
        samples, gestures, angles, masks, backbone="cnn", mode=mode,
        graph_config=None, angle_weight=1.0, seed=4, max_epochs=2, patience=2,
    )
    metrics, predictions = runner.evaluate_model(
        result.model, samples[masks["val"]], gestures[masks["val"]], angles[masks["val"]], mode
    )

    assert result.best_epoch in (0, 1)
    assert "angle_mae_deg" in metrics if mode != "gesture" else "angle_mae_deg" not in metrics
    assert "gesture_macro_f1" in metrics if mode != "angle" else "gesture_macro_f1" not in metrics
    if mode != "gesture":
        assert 1 <= metrics["angle_prediction_30deg_bins"] <= 12
    if mode != "angle":
        assert 1 <= metrics["gesture_unique_predictions"] <= 4
    assert len(predictions) == 4
