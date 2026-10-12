import numpy as np
import pytest
import torch

from src import v21_data
from src import v21_wear_calibrated as v21


@pytest.mark.parametrize("source", ["c170", "w01"])
def test_loaded_labels_support_cross_entropy_on_windows(tmp_path, source):
    path = tmp_path / "dataset.npz"
    if source == "c170":
        np.savez(path, X_raw=np.zeros((4, 500, 4)), is_duplicate=np.zeros(4, dtype=bool),
                 distance_mm=np.zeros(4), direction=v21_data.CLASSES,
                 block=np.arange(4), repetition=np.ones(4))
        labels = v21_data.load_c170(path)["y"]
    else:
        np.savez(path, strokes=np.zeros((4, 500, 4)), angles=np.zeros(4),
                 labels=np.arange(4, dtype=np.int32), blocks=np.arange(4), repetitions=np.ones(4))
        labels = v21_data.load_w01(path)["y"]
    loss = torch.nn.functional.cross_entropy(torch.randn(4, 4), torch.as_tensor(labels))
    assert torch.isfinite(loss)


@pytest.mark.parametrize("backbone", v21.BACKBONES)
def test_embeddings_are_unit_norm_and_logits_have_four_classes(backbone: str) -> None:
    model = v21.WearCalibratedClassifier(backbone).eval()
    samples = torch.randn(5, 500, 4)
    embeddings = model.embed(samples)
    assert embeddings.shape == (5, v21.EMBEDDING_DIM)
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(5))
    assert model(samples).shape == (5, 4)


def test_complex_gram_backbone_is_invariant_to_cyclic_channel_shift() -> None:
    model = v21.WearCalibratedClassifier("b1_complex_gram").eval()
    samples = torch.randn(3, 500, 4)
    torch.testing.assert_close(model.embed(samples), model.embed(torch.roll(samples, 1, dims=-1)), atol=1e-4, rtol=1e-4)


def test_calibrated_prototypes_follow_support_and_keep_unsupported_classes() -> None:
    global_prototypes = torch.eye(4, 8)
    support = torch.nn.functional.normalize(torch.tensor([[0.0, 0, 0, 0, 1, 0, 0, 0]]), dim=1)
    labels = torch.tensor([0])
    pulled = v21.calibrated_prototypes(global_prototypes, support, labels, tau=0.0)
    torch.testing.assert_close(pulled[0], support[0])
    torch.testing.assert_close(pulled[1:], global_prototypes[1:])
    kept = v21.calibrated_prototypes(global_prototypes, support, labels, tau=1e6)
    torch.testing.assert_close(kept, global_prototypes, atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        v21.calibrated_prototypes(global_prototypes, support, labels, tau=-1.0)


def test_augment_preserves_shape_and_identity_settings_return_input() -> None:
    samples = torch.randn(4, 500, 4)
    generator = torch.Generator().manual_seed(0)
    assert v21.augment(samples, generator).shape == samples.shape
    unchanged = v21.augment(samples, generator, max_shift=0, warp=0.0, gain_db=0.0, noise=0.0)
    torch.testing.assert_close(unchanged, samples, atol=1e-3, rtol=0)


def test_episode_loss_is_finite_and_differentiable() -> None:
    model = v21.WearCalibratedClassifier("b2_signed_cnn")
    samples = torch.randn(40, 500, 4)
    labels = torch.tensor(np.tile(np.repeat(np.arange(4, dtype=np.int64), 5), 2))
    groups = np.repeat(["A", "B"], 20)
    loss = v21.episode_loss(model, samples, labels, groups, np.random.default_rng(0), shots=2, tau=4.0, max_query=12)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.prototypes.grad is not None


def test_block_null_permutes_whole_blocks_within_groups() -> None:
    data = {"y": np.tile(np.repeat(np.arange(4), 3), 2), "group": np.repeat(["A", "B"], 12)}
    permuted = v21_data.block_permuted_labels(data, seed=5)
    for group in ("A", "B"):
        mask = data["group"] == group
        assert sorted(set(permuted[mask])) == [0, 1, 2, 3]
        for label in range(4):
            assert len(set(permuted[mask & (data["y"] == label)])) == 1
