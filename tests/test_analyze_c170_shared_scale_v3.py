"""Standard-library tests for the v3 validation-only aggregation."""

from __future__ import annotations

import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_c170_shared_scale_v3_results import (
    AuditError,
    FORMAL_SEEDS,
    FORMAL_VARIANTS,
    _aggregate_v3,
    _read_v3_snapshot,
    _validate_exact_matrix_v3,
    _write_report,
)


def _row(variant: str, seed: int, macro_f1: float) -> dict[str, object]:
    per_class = {"UP": 0.70, "DOWN": 0.75, "LEFT": 0.80, "RIGHT": 0.85}
    confusion = [[20, 0, 0, 0], [0, 20, 0, 0], [0, 0, 20, 0], [0, 0, 0, 20]]
    position_metric = {
        "macro_f1": macro_f1 - 0.02,
        "per_class_recall": [0.70, 0.75, 0.80, 0.85],
        "confusion_matrix": confusion,
    }
    return {
        "variant": variant,
        "seed": seed,
        "run_dir": f"{variant}-{seed}",
        "config_sha256": "test-config-hash",
        "artifact_sha256": {},
        "validation_macro_f1": macro_f1,
        "validation_cross_entropy": 0.5,
        "validation_per_class_recall": per_class,
        "validation_confusion_matrix": confusion,
        "validation_prediction_counts_by_class": {"UP": 80, "DOWN": 80, "LEFT": 80, "RIGHT": 80},
        "validation_by_position": {str(pos): position_metric for pos in (5, 40, 90, 130)},
        "prediction_unique_class_count": 4,
        "selected_checkpoint_train_target_macro_f1": 0.95,
        "selected_checkpoint_train_target_weighted_ce": 0.2,
        "train_target_minus_validation_macro_f1": 0.95 - macro_f1,
        "train_target_minus_validation_weighted_ce": -0.3,
        "selected_epoch_one_based": 10,
        "actual_epochs": 20,
        "optimizer_updates": 400,
        "shared_sensor_scale": 2.5,
        "epoch_curve": [],
    }


class SharedScaleV3AggregationTests(unittest.TestCase):
    def test_snapshot_verification_hashes_copied_source_files(self) -> None:
        source_files = {
            "runner": "run_c170_ring_cuda_experiment_v3.py",
            "shared_scale_preprocessing": "src/c170_shared_scale_preprocessing_v3.py",
            "ring_representation": "src/c170_ring_representation.py",
            "frozen_v2_training_helpers": "run_c170_ring_cuda_experiment_v2.py",
            "dataset_preprocessing_helpers": "run_c170_experiment.py",
            "preprocessing_transitive_signal_processing": "src/rotation_poc.py",
        }
        with TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot_dir = root / "source_snapshot"
            code_dir = snapshot_dir / "code"
            code_dir.mkdir(parents=True)
            code_hashes = {}
            file_hashes = {}
            for key, relative in source_files.items():
                saved = code_dir / relative
                saved.parent.mkdir(parents=True, exist_ok=True)
                payload = ("frozen source: " + key).encode("utf-8")
                saved.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                code_hashes[key] = digest
                file_hashes[relative] = digest
            (snapshot_dir / "snapshot_manifest.json").write_text(
                json.dumps({
                    "format": "c170-ring-v3-source-snapshot",
                    "provenance_code_sha256": code_hashes,
                    "snapshot_file_sha256": file_hashes,
                }),
                encoding="utf-8",
            )
            _path, verified, _snapshot = _read_v3_snapshot(root)
            self.assertEqual(verified, code_hashes)

    def test_matrix_requires_exact_two_by_five_variant_seed_pairs(self) -> None:
        rows = [
            {"variant": variant, "seed": seed}
            for variant in FORMAL_VARIANTS
            for seed in FORMAL_SEEDS
        ]
        self.assertEqual(len(_validate_exact_matrix_v3(rows)), 10)
        with self.assertRaises(AuditError):
            _validate_exact_matrix_v3(rows[:-1])
        with self.assertRaises(AuditError):
            _validate_exact_matrix_v3(rows + [rows[0]])
        with self.assertRaises(AuditError):
            _validate_exact_matrix_v3(rows + [{"variant": "ring_main", "seed": 17}])

    def test_summary_has_gate_positions_shuffle_diversity_and_v2_paired_differences(self) -> None:
        score_by_seed = dict(zip(FORMAL_SEEDS, (0.80, 0.82, 0.84, 0.86, 0.88)))
        v3_rows = []
        v2_rows = []
        for seed in FORMAL_SEEDS:
            ce_score = score_by_seed[seed]
            v3_rows.append(_row("shared_scale_ce", seed, ce_score))
            v3_rows.append(_row("shared_scale_shuffle", seed, 0.55))
            v2_rows.append(_row("ring_ce", seed, 0.60))

        summary = _aggregate_v3(
            v3_rows,
            v2_rows,
            {"runner": "fixed-hash"},
            [Path("v3-root")],
            Path("v2-root"),
        )

        ce_summary = summary["variants"]["shared_scale_ce"]
        self.assertAlmostEqual(ce_summary["validation_macro_f1"]["mean"], 0.84)
        self.assertAlmostEqual(ce_summary["validation_macro_f1"]["std_population"], 0.0282842712474619)
        self.assertTrue(summary["gate"]["passed"])
        self.assertEqual(set(ce_summary["validation_by_position"]), {"5", "40", "90", "130"})
        self.assertEqual(summary["variants"]["shared_scale_shuffle"]["prediction_unique_class_count"]["mean"], 4.0)
        self.assertAlmostEqual(
            summary["paired_seed_macro_f1_differences"]["shared_scale_ce_minus_v2_ring_ce"]["mean"],
            0.24,
        )
        self.assertFalse(summary["test_metrics_computed"])
        with TemporaryDirectory() as temp:
            report_dir = Path(temp) / "report"
            _write_report(report_dir, summary)
            with (report_dir / "per_seed_validation.csv").open(encoding="utf-8-sig", newline="") as stream:
                header = stream.readline().strip()
            self.assertIn("validation_recall_up", header)
            self.assertIn("validation_f1_at_130mm", header)


if __name__ == "__main__":
    unittest.main()
