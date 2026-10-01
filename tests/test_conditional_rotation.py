import importlib

import numpy as np
import pytest
import torch


def conditional_rotation():
    try:
        return importlib.import_module("conditional_rotation")
    except ModuleNotFoundError as error:
        pytest.fail(f"conditional rotation experiment module is missing: {error}")


def test_circular_angle_encoding_and_error_wrap_at_zero() -> None:
    experiment = conditional_rotation()
    encoded = experiment.angle_features(torch.tensor([0.0, 360.0, 90.0]))

    torch.testing.assert_close(encoded[0], encoded[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(encoded[2], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0)
    assert np.allclose(
        experiment.circular_error_deg(np.array([359.0, 1.0]), np.array([1.0, 359.0])),
        [2.0, 2.0],
    )


@pytest.mark.parametrize("backbone", ["cnn", "graph"])
def test_each_direction_uses_only_its_supplied_condition(backbone: str) -> None:
    experiment = conditional_rotation()
    torch.manual_seed(4)
    model = experiment.ConditionalRotationModel(
        backbone,
        graph_config={"hidden_size": 8, "num_layers": 1, "graph_heads": 2, "graph_topology": "ring"},
    ).eval()
    samples = torch.randn(3, 4, 32)
    gestures = torch.tensor([0, 1, 2])
    angles = torch.tensor([0.0, 30.0, 330.0])

    with torch.no_grad():
        angle_vectors = model.predict_angle(samples, gestures)
        gesture_logits = model.predict_gesture(samples, angles)
    assert angle_vectors.shape == (3, 2)
    assert gesture_logits.shape == (3, 4)
    assert torch.isfinite(angle_vectors).all()
    assert torch.isfinite(gesture_logits).all()
    if backbone == "graph":
        assert len(model.graph.encoders) == 4
        assert len({id(encoder.weight_ih_l0) for encoder in model.graph.encoders}) == 4


def test_angle_only_training_cannot_use_angle_as_condition() -> None:
    experiment = conditional_rotation()
    model = experiment.ConditionalRotationModel("cnn")
    samples = torch.randn(2, 4, 32)
    gestures = torch.tensor([0, 1])

    predicted = model.predict_angle(samples, gestures)
    loss = experiment.angle_vector_loss(predicted, torch.tensor([30.0, 330.0]))
    loss.backward()

    assert loss.item() > 0
    assert any(parameter.grad is not None for parameter in model.angle_head.parameters())
    assert all(parameter.grad is None for parameter in model.gesture_head.parameters())


def test_gesture_only_training_cannot_use_gesture_as_condition() -> None:
    experiment = conditional_rotation()
    model = experiment.ConditionalRotationModel("cnn")
    samples = torch.randn(2, 4, 32)
    angles = torch.tensor([30.0, 330.0])

    logits = model.predict_gesture(samples, angles)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.gesture_head.parameters())
    assert all(parameter.grad is None for parameter in model.angle_head.parameters())


def test_angle_metrics_use_circular_distance() -> None:
    experiment = conditional_rotation()
    metrics = experiment.angle_metrics(
        np.array([359.0, 10.0, 180.0]), np.array([1.0, 350.0, 210.0])
    )
    assert metrics["mae_deg"] == pytest.approx((2 + 20 + 30) / 3)
    assert metrics["within_30_deg"] == pytest.approx(1.0)
