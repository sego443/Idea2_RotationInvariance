from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


def ensemble_runner():
    return import_module("run_c170_validation_ensemble")


def test_ensemble_definitions_use_fixed_members_and_equal_probability_average() -> None:
    ensemble = ensemble_runner()

    assert ensemble.ensemble_definitions() == {
        "orbit_prob_5seed": ("orbit_prob",),
        "three_model_15seed": ("orbit", "orbit_prob", "harmonic"),
    }
    assert ensemble.ENSEMBLE_AGGREGATION == "equal_mean_softmax_probabilities"


def test_ensemble_output_directory_refuses_to_overwrite_existing_path(tmp_path: Path) -> None:
    ensemble = ensemble_runner()
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError):
        ensemble.create_output_directory(output_dir)


def test_real_c170_ensemble_checkpoint_reload_reproduces_validation_probabilities(
    tmp_path: Path,
) -> None:
    ensemble = ensemble_runner()
    experiment = ensemble.c170
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    val_indices = ensemble.validation_indices(dataset, [5, 40, 90, 130])[:12]
    raw_windows = dataset["X_raw"][val_indices]
    source_outputs = {
        "baseline": ensemble.DEFAULT_BASELINE_OUTPUT_DIR,
        "candidates": ensemble.DEFAULT_CANDIDATE_OUTPUT_DIR,
    }
    members = ensemble.load_member_checkpoints(
        source_outputs,
        "orbit_prob_5seed",
        "true_training_labels",
        experiment.DEFAULT_SEEDS,
    )

    original_probabilities = ensemble.predict_ensemble_probabilities(members, raw_windows)
    checkpoint_path = tmp_path / "orbit_prob_5seed.pt"
    ensemble.save_ensemble_checkpoint(
        checkpoint_path,
        "orbit_prob_5seed",
        "true_training_labels",
        members,
        experiment.dataset_sha256(experiment.DEFAULT_DATASET_PATH),
        [5, 40, 90, 130],
    )
    reloaded_members, payload = ensemble.load_ensemble_checkpoint(checkpoint_path)
    reloaded_probabilities = ensemble.predict_ensemble_probabilities(
        reloaded_members, raw_windows
    )

    np.testing.assert_allclose(reloaded_probabilities, original_probabilities, rtol=0, atol=0)
    np.testing.assert_allclose(reloaded_probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)
    assert len(payload["members"]) == 5
    assert payload["dataset_sha256"] == experiment.dataset_sha256(
        experiment.DEFAULT_DATASET_PATH
    )
    assert all(member["checkpoint_sha256"] for member in payload["members"])
    assert all(
        member["preprocessing"]["model_input_transform"] == "identity"
        for member in payload["members"]
    )


def test_legacy_baseline_orbit_checkpoint_uses_verified_identity_preprocessing() -> None:
    ensemble = ensemble_runner()
    source_outputs = {
        "baseline": ensemble.DEFAULT_BASELINE_OUTPUT_DIR,
        "candidates": ensemble.DEFAULT_CANDIDATE_OUTPUT_DIR,
    }

    members = ensemble.load_member_checkpoints(
        source_outputs,
        "three_model_15seed",
        "true_training_labels",
        ensemble.c170.DEFAULT_SEEDS,
    )
    legacy_orbit_members = [member for member in members if member["model_kind"] == "orbit"]

    assert len(legacy_orbit_members) == 5
    assert all(member["architecture"] == ensemble.c170.model_architecture("orbit") for member in legacy_orbit_members)
    assert all(
        member["preprocessing"]["model_input_transform"] == "identity"
        for member in legacy_orbit_members
    )
