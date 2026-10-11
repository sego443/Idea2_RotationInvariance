"""Standard-library tests for fail-closed v2 validation aggregation."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.analysis.analyze_c170_ring_cuda_v2_results import (
    AuditError,
    FORMAL_SEEDS,
    FORMAL_VARIANTS,
    _classification_metrics,
    _aggregate,
    _stats,
    _validate_exact_matrix,
    _write_report,
)


class RingV2AggregationTests(unittest.TestCase):
    def test_metrics_recompute_confusion_and_macro_f1(self) -> None:
        result = _classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.8) / 4)

    def test_statistics_use_population_standard_deviation(self) -> None:
        self.assertEqual(_stats([1.0, 3.0]), {"n": 2, "mean": 2.0, "std_population": 1.0})

    def test_matrix_rejects_missing_duplicate_and_unexpected_pairs(self) -> None:
        rows = [{"variant": variant, "seed": seed} for variant in FORMAL_VARIANTS for seed in FORMAL_SEEDS]
        self.assertEqual(len(_validate_exact_matrix(rows)), 20)
        with self.assertRaises(AuditError):
            _validate_exact_matrix(rows[:-1])
        with self.assertRaises(AuditError):
            _validate_exact_matrix(rows + [rows[0]])
        with self.assertRaises(AuditError):
            _validate_exact_matrix(rows + [{"variant": "other", "seed": 17}])

    def test_aggregate_and_report_include_gate_pairs_positions_and_shuffle_counts(self) -> None:
        rows = []
        confusion = [[20, 0, 0, 0], [0, 20, 0, 0], [0, 0, 20, 0], [0, 0, 0, 20]]
        for variant in FORMAL_VARIANTS:
            for seed in FORMAL_SEEDS:
                metric = {
                    "macro_f1": 0.9,
                    "per_class_recall": [0.9, 0.8, 0.85, 0.95],
                    "confusion_matrix": confusion,
                }
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "validation_macro_f1": 0.9 if variant == "ring_main" else 0.7,
                        "validation_cross_entropy": 0.4,
                        "validation_per_class_recall": dict(zip(("UP", "DOWN", "LEFT", "RIGHT"), metric["per_class_recall"])),
                        "validation_confusion_matrix": confusion,
                        "validation_prediction_counts_by_class": dict(zip(("UP", "DOWN", "LEFT", "RIGHT"), (80, 80, 80, 80))),
                        "validation_by_position": {str(position): metric for position in (5, 40, 90, 130)},
                        "validation_by_position_and_channel_map": {
                            str(position): {"0,2,4,6" if position in (5, 40) else "0,1,2,3": metric}
                            for position in (5, 40, 90, 130)
                        },
                        "prediction_unique_class_count": 4,
                        "selected_checkpoint_train_target_macro_f1": 0.95,
                        "selected_checkpoint_train_target_weighted_ce": 0.2,
                        "train_target_minus_validation_macro_f1": 0.05,
                        "train_target_minus_validation_weighted_ce": -0.2,
                        "selected_epoch_one_based": 10,
                        "actual_epochs": 20,
                        "optimizer_updates": 400,
                        "epoch_curve": [
                            {
                                "epoch": 1,
                                "val_macro_f1": 0.5,
                                "val_cross_entropy": 1.0,
                                "train_weighted_ce_online": 1.2,
                                "train_supcon_batch_mean_online": 2.0 if variant != "ring_ce" else None,
                                "epoch_seconds": 1.0,
                            }
                        ],
                    }
                )
        summary = _aggregate(rows, {"runner": "abc"}, [Path("root")])
        self.assertTrue(summary["gate"]["passed"])
        self.assertIn("ring_main_minus_ring_ce", summary["paired_seed_macro_f1_differences"])
        self.assertIn("130", summary["variants"]["ring_main"]["validation_by_position"])
        self.assertEqual(summary["variants"]["ring_shuffle"]["validation_prediction_counts_by_class"]["UP"]["mean"], 80.0)
        with TemporaryDirectory() as temp:
            report_dir = Path(temp) / "report"
            _write_report(report_dir, summary)
            self.assertTrue((report_dir / "validation_summary.md").is_file())
            self.assertTrue((report_dir / "learning_curve_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
