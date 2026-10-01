from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import run_c170_experiment as c170
import run_c170_ring_experiment as ring_experiment


def test_fixed_ring_plan_records_train_val_and_registered_test_distances() -> None:
    plan = ring_experiment.build_experiment_plan(speed_probe=True)

    assert plan["train_distance_mm"] == [0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160]
    assert plan["val_distance_mm"] == [5, 40, 90, 130]
    assert plan["test_distance_mm"] == [15, 25, 110, 150]
    assert 70 not in sum(
        (plan["train_distance_mm"], plan["val_distance_mm"], plan["test_distance_mm"]),
        [],
    )
    assert plan["variants"] == [
        "main_true",
        "main_block_permutation",
        "classification_only",
        "slot_readout",
    ]
    assert plan["seeds"] == [17]
    assert plan["max_epochs"] == 3
    assert plan["cpu_threads"] == 2


def test_ring_split_reuses_real_c170_distance_partition() -> None:
    dataset = c170.load_dataset(c170.DEFAULT_DATASET_PATH)
    plan = ring_experiment.build_experiment_plan(speed_probe=True)
    splits = c170.split_by_distance(
        dataset["distance_mm"],
        {
            "train": plan["train_distance_mm"],
            "val": plan["val_distance_mm"],
            "test": plan["test_distance_mm"],
        },
    )

    assert [len(splits[name]) for name in ("train", "val", "test")] == [879, 320, 320]
    assert np.array_equal(
        np.unique(dataset["distance_mm"][splits["val"]]),
        np.asarray(plan["val_distance_mm"]),
    )


def test_cli_only_accepts_validation_phase() -> None:
    with pytest.raises(SystemExit):
        ring_experiment.parse_args(["--phase", "final-test", "--output-dir", "unused"])


def test_experiment_names_map_to_reloadable_model_variants() -> None:
    assert ring_experiment._model_variant_for_experiment("main_true") == "main"
    assert ring_experiment._model_variant_for_experiment("main_block_permutation") == "main"
    assert ring_experiment._model_variant_for_experiment("classification_only") == "classification_only"
    assert ring_experiment._model_variant_for_experiment("slot_readout") == "slot_readout"
