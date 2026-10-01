"""Contract tests for the isolated second-round C170 ring runner."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from run_c170_ring_cuda_experiment_v2 import (
    VARIANT_CONFIG,
    _batches_for_epoch,
    _classification_metrics,
    _make_training_targets,
    _training_groups,
    batch_objective,
)


def synthetic_training_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances: list[int] = []
    labels: list[int] = []
    blocks: list[int] = []
    for position_index, distance in enumerate((0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160)):
        for label in range(4):
            count = 19 if (distance, label) == (140, 3) else 20
            distances.extend([distance] * count)
            labels.extend([label] * count)
            blocks.extend([position_index * 4 + label] * count)
    return (
        np.asarray(distances, dtype=np.int64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(blocks, dtype=np.int64),
    )


class RingRunnerContractTests(unittest.TestCase):
    def test_variant_map_matches_the_frozen_four_arm_design(self) -> None:
        self.assertEqual(
            set(VARIANT_CONFIG),
            {"ring_main", "ring_ce", "slot_supcon", "ring_shuffle"},
        )
        self.assertEqual(VARIANT_CONFIG["ring_main"], {"model_variant": "main", "use_supcon": True, "shuffle_labels": False})
        self.assertEqual(VARIANT_CONFIG["ring_ce"], {"model_variant": "classification_only", "use_supcon": False, "shuffle_labels": False})
        self.assertEqual(VARIANT_CONFIG["slot_supcon"], {"model_variant": "slot_readout", "use_supcon": True, "shuffle_labels": False})
        self.assertEqual(VARIANT_CONFIG["ring_shuffle"], {"model_variant": "main", "use_supcon": True, "shuffle_labels": True})

    def test_epoch_sampler_uses_each_sample_once_in_identical_batches(self) -> None:
        distances, labels, _blocks = synthetic_training_metadata()
        indices = np.arange(len(labels), dtype=np.int64)
        groups = _training_groups(indices, labels, distances)
        first = _batches_for_epoch(groups, seed=29, epoch=6)
        second = _batches_for_epoch(groups, seed=29, epoch=6)
        self.assertEqual([len(batch) for batch in first], [44] * 19 + [43])
        self.assertEqual(len(np.concatenate(first)), 879)
        self.assertEqual(len(np.unique(np.concatenate(first))), 879)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

    def test_weighted_ce_and_batch_mean_supcon_scaling(self) -> None:
        ce = torch.tensor([1.0, 3.0])
        weights = torch.tensor([0.25, 0.75])
        supcon = torch.tensor(2.0)
        self.assertAlmostEqual(float(batch_objective(ce, weights, supcon, use_supcon=True)), 50.2, places=5)
        self.assertAlmostEqual(float(batch_objective(ce, weights, None, use_supcon=False)), 50.0, places=5)

    def test_shuffle_targets_remain_constant_per_block_and_balanced_per_position(self) -> None:
        distances, labels, blocks = synthetic_training_metadata()
        indices = np.arange(len(labels), dtype=np.int64)
        first = _make_training_targets(labels, distances, blocks, indices, "ring_shuffle", seed=17)
        second = _make_training_targets(labels, distances, blocks, indices, "ring_shuffle", seed=17)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(any(first[blocks == block][0] != labels[blocks == block][0] for block in np.unique(blocks)))
        for block in np.unique(blocks):
            self.assertEqual(len(np.unique(first[blocks == block])), 1)
        for distance in np.unique(distances):
            self.assertEqual(sorted(np.unique(first[distances == distance]).tolist()), [0, 1, 2, 3])
        unchanged = _make_training_targets(labels, distances, blocks, indices, "ring_main", seed=17)
        np.testing.assert_array_equal(unchanged, labels)

    def test_action_metrics_report_confusion_and_macro_f1(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
        predictions = np.asarray([0, 1, 1, 1, 2, 3, 3, 3])
        metrics = _classification_metrics(labels, predictions)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 1, 1], [0, 0, 0, 2]])
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.8 + 2 / 3 + 0.8) / 4)
        self.assertEqual(metrics["prediction_unique_class_count"], 4)


if __name__ == "__main__":
    unittest.main()
