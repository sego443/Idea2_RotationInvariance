from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from launch_c170_latent_angle_cuda import (  # noqa: E402
    LauncherError,
    RunSpec,
    _parse_args,
    build_run_specs,
    config_sha256,
    preserve_completed_aggregate,
    validate_completed_run,
)


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


class LauncherContractTests(unittest.TestCase):
    def test_resume_verification_does_not_replace_original_wall_time(self) -> None:
        manifest = {
            "aggregate": {
                "launcher_wall_seconds": 18.25,
                "total_optimizer_updates": 800,
                "updates_per_second_overall": 800 / 18.25,
            }
        }

        preserve_completed_aggregate(manifest, verification_seconds=0.08)

        self.assertEqual(manifest["aggregate"]["launcher_wall_seconds"], 18.25)
        self.assertEqual(manifest["aggregate"]["total_optimizer_updates"], 800)
        self.assertEqual(manifest["aggregate"]["resume_verification_seconds"], 0.08)

    def test_resume_keeps_recovered_unknown_wall_time_unknown(self) -> None:
        manifest = {
            "aggregate": {
                "launcher_wall_seconds": None,
                "updates_per_second_overall": None,
                "wall_time_unavailable_on_recovery": True,
            }
        }

        preserve_completed_aggregate(manifest, verification_seconds=0.04)

        self.assertIsNone(manifest["aggregate"]["launcher_wall_seconds"])
        self.assertIsNone(manifest["aggregate"]["updates_per_second_overall"])
        self.assertTrue(manifest["aggregate"]["wall_time_unavailable_on_recovery"])
        self.assertEqual(manifest["aggregate"]["resume_verification_seconds"], 0.04)

    def test_formal_suite_has_the_frozen_25_runs(self) -> None:
        specs = build_run_specs("formal")

        self.assertEqual(len(specs), 25)
        self.assertEqual(
            [(spec.variant, spec.seed) for spec in specs[:5]],
            [(variant, seed) for variant in ("mlp", "no_angle", "full", "no_rms", "shuffle") for seed in (17, 29, 43, 61, 89)][:5],
        )
        self.assertTrue(all(spec.max_epochs == 160 and spec.patience == 25 for spec in specs))

    def test_probe_suite_has_the_same_eight_full_model_tasks(self) -> None:
        specs = build_run_specs("probe")

        self.assertEqual(
            [(spec.variant, spec.seed) for spec in specs],
            [("full", seed) for seed in (17, 29, 43, 61, 89, 101, 113, 127)],
        )
        self.assertTrue(all(spec.max_epochs == 5 and spec.patience == 5 for spec in specs))

    def test_formal_suite_accepts_eight_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            args = _parse_args(
                ["--suite", "formal", "--concurrency", "8", "--output-dir", temp_dir]
            )

        self.assertEqual(args.suite, "formal")
        self.assertEqual(args.concurrency, 8)

    def test_runner_completion_is_reusable_only_when_hashes_and_cli_config_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoint = run_dir / "best_checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint fixture")
            predictions = run_dir / "validation_predictions.csv"
            predictions.write_text("sample,prediction\n", encoding="utf-8")
            with (run_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("epoch", "optimizer_updates", "epoch_seconds"))
                writer.writeheader()
                writer.writerow({"epoch": 1, "optimizer_updates": 20, "epoch_seconds": 2.0})
            learning_curve = run_dir / "learning_curve.csv"

            config = {
                "variant": "full",
                "seed": 17,
                "max_epochs": 5,
                "patience": 5,
                "cpu_threads": 2,
                "device": "cuda",
                "dataset_path": "E:/data/dataset.npz",
                "dataset_sha256": "ab" * 32,
                "train_distance_mm": [0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160],
                "val_distance_mm": [5, 40, 90, 130],
                "registered_test_distance_mm": [15, 25, 110, 150],
                "excluded_distance_mm": [70],
                "input_shape": [500, 4],
                "num_actions": 4,
                "num_angles": 64,
                "updates_per_epoch": 20,
                "batch_sizes_per_epoch": [44] * 19 + [43],
                "batch_loss_weight_multiplier": 20,
                "sampler_group_key": "distance_mm_and_original_direction_label",
                "sampler_rng": "numpy_seedsequence_seed_epoch",
                "sample_weight_rule": "position_voronoi_weight/(4*original_group_size)",
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.001,
                "angle_weight": 0.2,
                "smoothness_weight": 0.001,
                "angle_target_bandwidth": 8.0,
                "include_rms": True,
                "sampler": "original_distance_label_groups_no_replacement_19x44_plus_43",
                "label_shuffle_seed": None,
            }
            digest = config_sha256(config)
            checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            _write_json(
                run_dir / "run_manifest.json",
                {
                    "format": "c170-latent-angle-run",
                    "format_version": 1,
                    "status": "completed",
                    "config": config,
                    "config_sha256": digest,
                    "timing": {
                        "preprocessing_seconds": 1.0,
                        "training_seconds": 2.0,
                        "total_elapsed_seconds": 3.5,
                    },
                    "selection": {"best_epoch_zero_based": 1},
                    "artifacts": {
                        "checkpoint": {
                            "path": checkpoint.name,
                            "sha256": checkpoint_digest,
                        },
                        "learning_curve": {
                            "path": learning_curve.name,
                            "sha256": hashlib.sha256(learning_curve.read_bytes()).hexdigest(),
                        },
                        "validation_predictions": {
                            "path": predictions.name,
                            "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                        },
                    },
                    "test_metrics_computed": False,
                },
            )
            manifest_digest = hashlib.sha256((run_dir / "run_manifest.json").read_bytes()).hexdigest()
            _write_json(
                run_dir / "RUN_COMPLETE.json",
                {
                    "format": "c170-latent-angle-complete",
                    "format_version": 1,
                    "status": "completed",
                    "config_sha256": digest,
                    "checkpoint_sha256": checkpoint_digest,
                    "manifest_sha256": manifest_digest,
                    "selected_epoch": 1,
                    "test_metrics_computed": False,
                },
            )
            expected = {
                "variant": "full",
                "seed": 17,
                "max_epochs": 5,
                "patience": 5,
                "cpu_threads": 2,
                "device": "cuda",
                "dataset_path": "E:/data/dataset.npz",
                "dataset_sha256": "ab" * 32,
            }

            completed = validate_completed_run(run_dir, expected)

            self.assertEqual(completed["config_sha256"], digest)
            self.assertEqual(completed["checkpoint_sha256"], checkpoint_digest)
            with self.assertRaises(LauncherError):
                validate_completed_run(run_dir, {**expected, "seed": 29})


if __name__ == "__main__":
    unittest.main()
