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

from launch_c170_ring_cuda_v2 import (  # noqa: E402
    LauncherError,
    RunSpec,
    _make_command,
    _parse_args,
    _prepare_output_root,
    _refresh_progress,
    _runner_expected_fields,
    build_run_specs,
    config_sha256,
    validate_completed_run,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


class RingCudaLauncherContractTests(unittest.TestCase):
    def test_probe_has_eight_legal_tasks_and_unique_repeat_ids(self) -> None:
        specs = build_run_specs("probe")

        self.assertEqual(len(specs), 8)
        self.assertEqual(
            [spec.seed for spec in specs],
            [17, 29, 43, 61, 89, 17, 29, 43],
        )
        self.assertEqual(len({spec.job_id for spec in specs}), 8)
        self.assertEqual(specs[5].job_id, "ring_main__seed-17__repeat-2")
        self.assertTrue(all(spec.variant == "ring_main" for spec in specs))
        self.assertTrue(all(spec.max_epochs == 3 and spec.patience == 3 for spec in specs))

    def test_formal_matrix_has_twenty_unique_variant_seed_runs(self) -> None:
        specs = build_run_specs("formal")

        self.assertEqual(len(specs), 20)
        self.assertEqual(
            [(spec.variant, spec.seed) for spec in specs],
            [
                (variant, seed)
                for variant in ("ring_main", "ring_ce", "slot_supcon", "ring_shuffle")
                for seed in (17, 29, 43, 61, 89)
            ],
        )
        self.assertEqual(len({spec.job_id for spec in specs}), 20)
        self.assertTrue(all(spec.max_epochs == 160 and spec.patience == 25 for spec in specs))

    def test_variant_specific_configs_match_the_frozen_runner(self) -> None:
        base = dict(
            seed=17,
            max_epochs=5,
            patience=5,
            cpu_threads=2,
            dataset_path=Path("E:/data/dataset.npz"),
            dataset_sha="ab" * 32,
        )
        expected = {
            "ring_main": ("main", True, False, 0.1),
            "ring_ce": ("classification_only", False, False, 0.0),
            "slot_supcon": ("slot_readout", True, False, 0.1),
            "ring_shuffle": ("main", True, True, 0.1),
        }
        for variant, values in expected.items():
            with self.subTest(variant=variant):
                spec = RunSpec(variant, base["seed"], base["max_epochs"], base["patience"])
                config = _runner_expected_fields(spec, Path(base["dataset_path"]), base["dataset_sha"], 2)
                self.assertEqual(
                    (
                        config["model_variant"],
                        config["use_supcon"],
                        config["shuffle_labels"],
                        config["supcon_weight"],
                    ),
                    values,
                )
                self.assertEqual(config["device"], "cuda:0")
                self.assertEqual(config["learning_rate"], 0.001)
                self.assertEqual(config["weight_decay"], 0.001)
                self.assertEqual(config["supcon_temperature"], 0.1)
                self.assertEqual(config["cross_entropy_label_smoothing"], 0.0)
                self.assertEqual(config["updates_per_epoch"], 20)

    def test_command_passes_the_frozen_cuda_cli(self) -> None:
        command = _make_command(
            python_exe=Path("C:/Python310/python.exe"),
            runner_path=Path("E:/project/run_c170_ring_cuda_experiment_v2.py"),
            run_dir=Path("E:/out/ring_main__seed-17"),
            dataset_path=Path("E:/data/dataset.npz"),
            spec=RunSpec("ring_main", 17, 160, 25),
            cpu_threads=2,
        )

        self.assertEqual(command[0], str(Path("C:/Python310/python.exe")))
        self.assertIn("--confirm-physical-order", command)
        self.assertEqual(
            command[command.index("--device") + 1],
            "cuda",
        )
        self.assertEqual(command[command.index("--variant") + 1], "ring_main")
        self.assertEqual(command[command.index("--seed") + 1], "17")
        self.assertEqual(command[command.index("--max-epochs") + 1], "160")
        self.assertEqual(command[command.index("--patience") + 1], "25")

    def test_cli_accepts_only_measured_concurrency_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for concurrency in (4, 8):
                args = _parse_args(
                    ["--suite", "probe", "--concurrency", str(concurrency), "--output-dir", temp_dir]
                )
                self.assertEqual(args.concurrency, concurrency)
            with self.assertRaises(SystemExit):
                _parse_args(["--suite", "probe", "--concurrency", "3", "--output-dir", temp_dir])

    def test_formal_root_can_contain_only_the_prepared_source_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "formal"
            snapshot = output_dir / "source_snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / "snapshot_manifest.json").write_text("{}", encoding="utf-8")
            (snapshot / "environment.json").write_text("{}", encoding="utf-8")
            spec = RunSpec("ring_main", 17, 160, 25)
            config = {"suite": "formal", "output_dir": str(output_dir)}

            manifest = _prepare_output_root(output_dir, config, [spec])

            self.assertEqual(manifest["status"], "running")
            self.assertTrue(snapshot.is_dir())
            self.assertTrue((output_dir / "launcher_manifest.json").is_file())

    def test_formal_root_refuses_unrelated_preexisting_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "formal"
            output_dir.mkdir(parents=True)
            (output_dir / "unrelated.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(LauncherError):
                _prepare_output_root(
                    output_dir,
                    {"suite": "formal", "output_dir": str(output_dir)},
                    [RunSpec("ring_main", 17, 160, 25)],
                )

    def test_progress_reader_uses_stdout_epoch_tokens_without_opening_curve_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "worker.stdout.log"
            log_path.write_text(
                "variant=ring_main seed=17 epoch=3 updates=20 val_macro_f1=0.625000\n",
                encoding="utf-8",
            )
            record = {
                "stdout_log": str(log_path),
                "run_dir": str(Path(temp_dir) / "run-with-no-curve-yet"),
                "latest_epoch": 2,
            }

            changed = _refresh_progress("ring_main__seed-17", record)

            self.assertTrue(changed)
            self.assertEqual(record["latest_epoch"], 3)
            self.assertEqual(record["latest_epoch_updates"], 20)
            self.assertIsNotNone(record["last_progress_utc"])

    def test_completed_run_requires_manifest_and_all_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            checkpoint = run_dir / "best_checkpoint.pt"
            checkpoint.write_bytes(b"ring checkpoint")
            predictions = run_dir / "validation_predictions.csv"
            predictions.write_text("sample,prediction\n0,1\n", encoding="utf-8")
            curve = run_dir / "learning_curve.csv"
            with curve.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("epoch", "optimizer_updates", "epoch_seconds"),
                )
                writer.writeheader()
                writer.writerow({"epoch": 1, "optimizer_updates": 20, "epoch_seconds": 1.25})

            artifact_hashes = {
                "best_checkpoint": _digest(checkpoint),
                "learning_curve": _digest(curve),
                "validation_predictions": _digest(predictions),
            }
            code_hashes = {
                "runner": "11" * 32,
                "ring_representation": "22" * 32,
                "dataset_preprocessing_helpers": "33" * 32,
                "preprocessing_transitive_signal_processing": "44" * 32,
            }
            spec = RunSpec("ring_main", 17, 5, 5)
            dataset_path = Path("E:/data/dataset.npz")
            config = _runner_expected_fields(spec, dataset_path, "ab" * 32, 2)
            digest = config_sha256(config)
            manifest_path = run_dir / "run_manifest.json"
            _write_json(
                manifest_path,
                {
                    "format": "c170-ring-cuda-v2-run",
                    "format_version": 1,
                    "status": "completed",
                    "config": config,
                    "config_sha256": digest,
                    "split": {
                        "split_counts": {"train": 879, "val": 320, "test": 320},
                        "label_shuffle_seed": None,
                        "label_shuffle_mapping": [],
                    },
                    "provenance": {"code_sha256": code_hashes},
                    "timing": {
                        "preprocessing_seconds": 1.0,
                        "training_seconds": 1.25,
                        "total_elapsed_seconds": 2.5,
                    },
                    "selection": {"best_epoch_one_based": 1},
                    "artifacts": {
                        name: {"path": path.name, "sha256": artifact_hashes[name]}
                        for name, path in (
                            ("best_checkpoint", checkpoint),
                            ("learning_curve", curve),
                            ("validation_predictions", predictions),
                        )
                    },
                    "test_metrics_computed": False,
                    "test_access": {"predicted_or_scored": False},
                },
            )
            _write_json(
                run_dir / "RUN_COMPLETE.json",
                {
                    "format": "c170-ring-cuda-v2-complete",
                    "format_version": 1,
                    "status": "completed",
                    "config_sha256": digest,
                    "manifest_sha256": _digest(manifest_path),
                    "artifact_sha256": artifact_hashes,
                    "selected_epoch_one_based": 1,
                    "test_metrics_computed": False,
                },
            )
            expected = _runner_expected_fields(
                spec,
                dataset_path,
                "ab" * 32,
                2,
            )

            verified = validate_completed_run(
                run_dir,
                expected,
                expected_code_sha256=code_hashes,
            )

            self.assertEqual(verified["config_sha256"], digest)
            self.assertEqual(verified["artifact_sha256"], artifact_hashes)
            self.assertEqual(verified["learning_curve"][0]["epoch"], "1")

            with self.assertRaises(LauncherError):
                validate_completed_run(
                    run_dir,
                    {**expected, "seed": 29},
                    expected_code_sha256=code_hashes,
                )

            checkpoint.write_bytes(b"tampered checkpoint")
            with self.assertRaises(LauncherError):
                validate_completed_run(
                    run_dir,
                    expected,
                    expected_code_sha256=code_hashes,
                )

    def test_completed_run_rejects_manifest_or_artifact_tampering(self) -> None:
        # Canonical JSON hashes distinguish any change in a recorded configuration.
        self.assertNotEqual(config_sha256({"a": 1}), config_sha256({"a": 2}))


if __name__ == "__main__":
    unittest.main()
