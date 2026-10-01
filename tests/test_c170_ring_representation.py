from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_experiment as c170
import run_c170_ring_experiment as ring_experiment
from c170_ring_representation import (
    RingRepresentationClassifier,
    build_balanced_batches,
    build_cross_position_masks,
    cross_position_supervised_contrastive_loss,
)


TRAIN_DISTANCE_MM = (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)
VAL_DISTANCE_MM = (5, 40, 90, 130)
TEST_DISTANCE_MM = (15, 25, 110, 150)


@pytest.fixture(scope="module")
def real_c170() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    torch.set_num_threads(1)
    dataset = c170.load_dataset(c170.DEFAULT_DATASET_PATH)
    splits = c170.split_by_distance(
        dataset["distance_mm"],
        {
            "train": TRAIN_DISTANCE_MM,
            "val": VAL_DISTANCE_MM,
            "test": TEST_DISTANCE_MM,
        },
    )
    normalized, _ = c170.prepare_model_inputs(dataset["X_raw"], splits)
    return dataset, splits, normalized


def _sample_index(
    dataset: dict[str, np.ndarray],
    indices: np.ndarray,
    distance_mm: int,
    label: int,
    repetition: int | None = None,
) -> int:
    candidates = indices[
        (dataset["distance_mm"][indices] == distance_mm)
        & (dataset["labels"][indices] == label)
    ]
    if repetition is not None:
        candidates = candidates[dataset["repetition"][candidates] == repetition]
    assert len(candidates) > 0
    return int(candidates[0])


def test_real_waveforms_have_c4_equivariant_nodes_and_invariant_logits(real_c170) -> None:
    dataset, splits, normalized = real_c170
    selection = np.asarray(
        [
            _sample_index(dataset, splits["train"], 0, 0),
            _sample_index(dataset, splits["train"], 10, 1),
            _sample_index(dataset, splits["train"], 80, 2),
        ],
        dtype=np.int64,
    )
    waveforms = torch.from_numpy(normalized["train"][
        np.searchsorted(splits["train"], selection)
    ])
    model = RingRepresentationClassifier("main").eval()

    nodes = model.encode_nodes(waveforms)
    logits = model(waveforms)

    assert nodes.shape == (3, 4, 32, 125)
    assert logits.shape == (3, 4)
    for shift in range(1, 4):
        shifted_nodes = model.encode_nodes(torch.roll(waveforms, shifts=shift, dims=2))
        shifted_logits = model(torch.roll(waveforms, shifts=shift, dims=2))
        torch.testing.assert_close(
            shifted_nodes,
            torch.roll(nodes, shifts=shift, dims=1),
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(shifted_logits, logits, rtol=1e-5, atol=1e-6)


def test_ring_messages_keep_distinct_oriented_temporal_weights() -> None:
    model = RingRepresentationClassifier("main")

    assert len(model.ring_blocks) == 2
    for block, dilation in zip(model.ring_blocks, (1, 2), strict=True):
        assert block.plus_conv.weight.data_ptr() != block.minus_conv.weight.data_ptr()
        assert block.plus_conv.dilation == (dilation,)
        assert block.minus_conv.dilation == (dilation,)


def test_real_metadata_pair_masks_enforce_cross_position_and_cross_block(real_c170) -> None:
    dataset, splits, _ = real_c170
    train_indices = splits["train"]
    anchor = _sample_index(dataset, train_indices, 0, 0, repetition=1)
    same_position = _sample_index(dataset, train_indices, 0, 0, repetition=2)
    positive = _sample_index(dataset, train_indices, 10, 0, repetition=1)
    negative = _sample_index(dataset, train_indices, 10, 1, repetition=1)
    negative_positive = _sample_index(dataset, train_indices, 0, 1, repetition=1)
    selected = np.asarray(
        [anchor, same_position, positive, negative, negative_positive],
        dtype=np.int64,
    )

    masks = build_cross_position_masks(
        dataset["labels"][selected],
        dataset["distance_mm"][selected],
        dataset["block"][selected],
    )

    assert masks.positive[0, 2]
    assert masks.negative[0, 3]
    assert not masks.valid[0, 1]
    positive_pairs = np.argwhere(masks.positive.cpu().numpy())
    assert np.all(
        dataset["distance_mm"][selected[positive_pairs[:, 0]]]
        != dataset["distance_mm"][selected[positive_pairs[:, 1]]]
    )
    assert np.all(
        dataset["block"][selected[positive_pairs[:, 0]]]
        != dataset["block"][selected[positive_pairs[:, 1]]]
    )


def test_real_train_batches_cover_every_position_and_direction(real_c170) -> None:
    dataset, splits, _ = real_c170
    batches = build_balanced_batches(
        splits["train"],
        dataset["labels"],
        dataset["distance_mm"],
        dataset["block"],
        seed=17,
    )

    assert len(batches) == 20
    for batch in batches:
        assert len(batch) == 44
        assert len(np.unique(dataset["distance_mm"][batch])) == 11
        assert len(np.unique(dataset["labels"][batch])) == 4
        assert len(set(zip(dataset["distance_mm"][batch].tolist(), dataset["labels"][batch].tolist()))) == 44
        assert len(np.unique(dataset["block"][batch])) == 44


def test_contrastive_loss_has_finite_gradient_on_real_train_batch(real_c170) -> None:
    dataset, splits, normalized = real_c170
    batch = build_balanced_batches(
        splits["train"],
        dataset["labels"],
        dataset["distance_mm"],
        dataset["block"],
        seed=29,
    )[0]
    positions = {int(index): offset for offset, index in enumerate(splits["train"])}
    train_offsets = np.asarray([positions[int(index)] for index in batch], dtype=np.int64)
    waveforms = torch.from_numpy(normalized["train"][train_offsets])
    labels = torch.from_numpy(dataset["labels"][batch].astype(np.int64))
    distances = torch.from_numpy(dataset["distance_mm"][batch].astype(np.int64))
    blocks = torch.from_numpy(dataset["block"][batch].astype(np.int64))
    model = RingRepresentationClassifier("main")
    logits, projection = model.forward_with_projection(waveforms)

    loss = cross_position_supervised_contrastive_loss(
        projection,
        labels,
        distances,
        blocks,
        temperature=0.1,
    )
    total = torch.nn.functional.cross_entropy(logits, labels) + 0.1 * loss
    total.backward()

    assert torch.isfinite(loss)
    assert model.encoder[0].weight.grad is not None
    assert torch.isfinite(model.encoder[0].weight.grad).all()


def test_slot_readout_accepts_real_waveforms(real_c170) -> None:
    _, _, normalized = real_c170
    waveforms = torch.from_numpy(normalized["train"][:2])
    model = RingRepresentationClassifier("slot_readout").eval()

    assert model(waveforms).shape == (2, 4)


def test_block_permutation_labels_reach_real_sampler_and_training_losses(real_c170) -> None:
    dataset, splits, normalized = real_c170
    train_targets_all, batches_by_epoch, shuffle_seed = ring_experiment._build_training_schedule(
        dataset,
        splits["train"],
        variant="main_block_permutation",
        seed=17,
        epoch_count=1,
    )
    batch = batches_by_epoch[0][0]
    assert shuffle_seed == 10_017
    assert not np.array_equal(train_targets_all[batch], dataset["labels"][batch])
    for distance_mm in np.unique(dataset["distance_mm"][batch]):
        at_position = batch[dataset["distance_mm"][batch] == distance_mm]
        assert np.array_equal(np.sort(train_targets_all[at_position]), np.arange(4))

    train_offsets = np.searchsorted(splits["train"], batch)
    waveforms = torch.from_numpy(normalized["train"][train_offsets])
    batch_targets = torch.from_numpy(train_targets_all[batch].astype(np.int64))
    model = RingRepresentationClassifier("main")
    logits, projection = model.forward_with_projection(waveforms)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.02)
    ce_loss, total_loss, contrastive_loss = ring_experiment._training_losses(
        logits,
        projection,
        batch_targets,
        dataset["distance_mm"][batch],
        dataset["block"][batch],
        criterion,
        contrastive_enabled=True,
    )

    expected_ce = criterion(logits, batch_targets)
    expected_contrastive = cross_position_supervised_contrastive_loss(
        projection,
        batch_targets,
        dataset["distance_mm"][batch],
        dataset["block"][batch],
        temperature=ring_experiment.CONTRASTIVE_TEMPERATURE,
    )
    true_label_ce = criterion(
        logits,
        torch.from_numpy(dataset["labels"][batch].astype(np.int64)),
    )
    true_label_contrastive = cross_position_supervised_contrastive_loss(
        projection,
        dataset["labels"][batch],
        dataset["distance_mm"][batch],
        dataset["block"][batch],
        temperature=ring_experiment.CONTRASTIVE_TEMPERATURE,
    )

    torch.testing.assert_close(ce_loss, expected_ce)
    torch.testing.assert_close(contrastive_loss, expected_contrastive)
    torch.testing.assert_close(
        total_loss,
        expected_ce + ring_experiment.CONTRASTIVE_WEIGHT * expected_contrastive,
    )
    assert not torch.isclose(ce_loss, true_label_ce)
    assert not torch.isclose(contrastive_loss, true_label_contrastive)
