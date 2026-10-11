import numpy as np
import torch

from lstm_graph_fusion import FourChannelLSTMGraphFusion
from rotation_poc import predict_temporal_model, train_temporal_model
from scripts.runners.run_lstm_graph_experiment import prepare_model_inputs


def test_four_channel_lstm_parameters_are_independent() -> None:
    model = FourChannelLSTMGraphFusion(head="mlp")

    assert len(model.encoders) == 4
    assert len({id(encoder.weight_ih_l0) for encoder in model.encoders}) == 4


def test_attention_fuses_each_channel_into_one_temporal_feature() -> None:
    torch.manual_seed(4)
    model = FourChannelLSTMGraphFusion(head="cnn").eval()
    samples = torch.randn(2, 4, 32)

    fused, attention = model.fuse(samples)
    logits = model(samples)

    assert fused.shape == (2, 32, model.feature_dim)
    assert attention.shape == (2, 32, 4, 4)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 32, 4))
    assert logits.shape == (2, 4)


def test_each_sensor_can_affect_fused_feature() -> None:
    torch.manual_seed(5)
    model = FourChannelLSTMGraphFusion(head="mlp")
    samples = torch.randn(2, 4, 28, requires_grad=True)

    fused, _ = model.fuse(samples)
    fused.square().sum().backward()

    assert samples.grad is not None
    assert torch.all(samples.grad.abs().sum(dim=(0, 2)) > 0)


def test_ring_attention_masks_opposite_sensor_with_two_heads() -> None:
    torch.manual_seed(6)
    model = FourChannelLSTMGraphFusion(
        head="cnn", hidden_size=8, num_layers=2, graph_heads=2, graph_topology="ring"
    ).eval()
    samples = torch.randn(2, 4, 24)

    _, attention = model.fuse(samples)

    assert all(encoder.num_layers == 2 for encoder in model.encoders)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 24, 4))
    for sensor in range(4):
        assert torch.count_nonzero(attention[:, :, sensor, (sensor + 2) % 4]) == 0


def test_graph_attention_rejects_incompatible_head_dimension() -> None:
    import pytest

    with pytest.raises(ValueError, match="divisible"):
        FourChannelLSTMGraphFusion(hidden_size=7, graph_heads=3)


def test_existing_training_pipeline_accepts_graph_fusion_model() -> None:
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(16, 24, 4)).astype(np.float32)
    labels = np.tile(np.arange(4), 4)

    result = train_temporal_model(
        samples[:12], labels[:12], samples[12:], labels[12:],
        model_kind="lstm_graph_mlp", seed=7, max_epochs=2, patience=2,
        model_config={"hidden_size": 8, "num_layers": 2, "graph_heads": 2, "graph_topology": "ring"},
    )

    assert result.train_predictions.shape == (12,)
    assert result.model.encoders[0].num_layers == 2
    assert result.model.graph_heads == 2
    assert predict_temporal_model(result.model, samples[12:]).shape == (4,)


def test_preparation_reuses_training_channel_scale_at_unseen_angles() -> None:
    time = np.arange(500) / 250.0
    wave = np.sin(2 * np.pi * 4 * time)
    one = np.column_stack([wave, 2 * wave, 3 * wave, 4 * wave])
    windows = np.stack([one, 2 * one]).astype(np.float32)
    angles = np.array([0, 30])

    prepared, scale = prepare_model_inputs(windows, angles)

    assert prepared.shape == (2, 125, 4)
    assert scale.shape == (4,)
    np.testing.assert_allclose(
        np.median(np.abs(prepared[1]), axis=0),
        2 * np.median(np.abs(prepared[0]), axis=0),
        atol=0.03,
    )
