from importlib import import_module
from pathlib import Path
import csv
import json

import numpy as np
import pytest
import torch


def runner():
    return import_module("run_c170_experiment")


def test_load_dataset_excludes_duplicate_rows_and_keeps_source_indices() -> None:
    experiment = runner()

    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    with np.load(experiment.DEFAULT_DATASET_PATH) as archive:
        duplicate_indices = np.flatnonzero(archive["is_duplicate"])

    assert len(dataset["X_raw"]) == 1599
    assert dataset["duplicate_rows_excluded"] == 1
    assert not dataset["is_duplicate"].any()
    np.testing.assert_array_equal(
        dataset["sample_index"],
        np.setdiff1d(np.arange(1600), duplicate_indices),
    )
    assert len(np.unique(dataset["sample_index"])) == len(dataset["sample_index"])


def test_split_by_distance_keeps_full_positions_and_excludes_unassigned_position() -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    distances = dataset["distance_mm"]
    assignments = {
        "train": [0, 10],
        "val": [5],
        "test": [15],
    }

    splits = experiment.split_by_distance(distances, assignments)

    assert set(distances[splits["train"]]) == {0, 10}
    assert set(distances[splits["val"]]) == {5}
    assert set(distances[splits["test"]]) == {15}
    assert all(70 not in distances[indices] for indices in splits.values())
    experiment.assert_no_distance_crossing(distances, splits)


def test_split_by_distance_rejects_a_position_assigned_twice() -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    assignments = {"train": [0, 10], "val": [5], "test": [10]}

    with pytest.raises(ValueError, match="assigned to multiple splits"):
        experiment.split_by_distance(dataset["distance_mm"], assignments)


def test_cli_requires_explicit_distance_positions_and_physical_order_confirmation() -> None:
    experiment = runner()

    args = experiment.parse_args(
        [
            "--phase",
            "validation-only",
            "--output-dir",
            "outputs/c170-test-run",
            "--train-distance-mm",
            "0",
            "10",
            "--val-distance-mm",
            "5",
            "--test-distance-mm",
            "15",
            "--confirm-physical-order",
        ]
    )

    assert args.train_distance_mm == [0, 10]
    assert args.val_distance_mm == [5]
    assert args.test_distance_mm == [15]
    assert args.confirm_physical_order
    experiment.validate_phase_args(args)


def test_cli_requires_explicit_candidate_selection_for_new_models() -> None:
    experiment = runner()
    args = experiment.parse_args(
        [
            "--phase",
            "validation-only",
            "--output-dir",
            "outputs/c170-candidate-smoke",
            "--train-distance-mm",
            "0",
            "--val-distance-mm",
            "5",
            "--test-distance-mm",
            "10",
            "--confirm-physical-order",
            "--model-kinds",
            "harmonic",
            "orbit_prob",
        ]
    )

    assert args.model_kinds == ["harmonic", "orbit_prob"]
    experiment.validate_phase_args(args)


def test_final_test_cli_rejects_position_arguments() -> None:
    experiment = runner()
    args = experiment.parse_args(
        ["--phase", "final-test", "--output-dir", "outputs/c170-test-run"]
    )

    experiment.validate_phase_args(args)
    args.test_distance_mm = [15]
    with pytest.raises(ValueError, match="does not accept position arguments"):
        experiment.validate_phase_args(args)


def test_user_selected_positions_support_confirmed_mixed_maps_and_group_shuffle() -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    assignments = {
        "train": [0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160],
        "val": [5, 40, 90, 130],
        "test": [15, 25, 110, 150],
    }

    splits = experiment.split_by_distance(dataset["distance_mm"], assignments)
    experiment.assert_recording_groups_intact(dataset)
    selected_indices = np.concatenate([splits[name] for name in experiment.SPLIT_NAMES])
    mappings = experiment.validate_channel_maps(
        dataset["channel_map"][selected_indices], confirm_physical_order=True
    )
    shuffled = experiment.shuffle_training_labels_by_recording(
        dataset["labels"],
        dataset["distance_mm"],
        dataset["block"],
        splits["train"],
        seed=10017,
    )

    assert 70 not in assignments["train"]
    assert set(mappings) == {"0,1,2,3", "0,2,4,6"}
    assert set(dataset["distance_mm"][splits["train"]]) == set(assignments["train"])
    for position in assignments["train"]:
        position_blocks = np.unique(dataset["block"][dataset["distance_mm"] == position])
        shuffled_block_labels = [
            np.unique(shuffled[dataset["block"] == block]).item() for block in position_blocks
        ]
        assert sorted(shuffled_block_labels) == list(range(4))


def test_mixed_channel_maps_require_explicit_physical_order_confirmation() -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    maps = np.unique(dataset["channel_map"], axis=0)

    with pytest.raises(ValueError, match="--confirm-physical-order"):
        experiment.validate_channel_maps(maps)

    confirmed = experiment.validate_channel_maps(maps, confirm_physical_order=True)

    assert confirmed == {
        "0,1,2,3": ["S0", "S1", "S2", "S3"],
        "0,2,4,6": ["S0", "S1", "S2", "S3"],
    }


def test_group_label_shuffle_preserves_four_direction_blocks_per_position() -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    distances = dataset["distance_mm"]
    blocks = dataset["block"]
    labels = dataset["labels"]
    train_indices = np.flatnonzero(distances == 20)

    shuffled = experiment.shuffle_training_labels_by_recording(
        labels, distances, blocks, train_indices, seed=17
    )

    assert not np.array_equal(shuffled[train_indices], labels[train_indices])
    np.testing.assert_array_equal(shuffled[~np.isin(np.arange(len(labels)), train_indices)], labels[~np.isin(np.arange(len(labels)), train_indices)])
    for position in np.unique(distances[train_indices]):
        position_blocks = np.unique(blocks[distances == position])
        block_labels = [np.unique(shuffled[blocks == block]).item() for block in position_blocks]
        assert sorted(block_labels) == list(range(4))
    for block in np.unique(blocks):
        block_labels = shuffled[blocks == block]
        assert len(np.unique(block_labels)) == 1


def test_group_label_shuffle_moves_every_block_when_one_recording_has_19_segments() -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    distances = dataset["distance_mm"]
    blocks = dataset["block"]
    labels = dataset["labels"]
    train_indices = np.flatnonzero(distances == 20)

    shuffled = experiment.shuffle_training_labels_by_recording(
        labels, distances, blocks, train_indices, seed=17
    )

    for block in (17, 18, 19, 20):
        assert np.unique(shuffled[blocks == block]).item() != np.unique(labels[blocks == block]).item()
    shuffled_training_block_labels = [
        np.unique(shuffled[blocks == block]).item() for block in (17, 18, 19, 20)
    ]
    assert sorted(shuffled_training_block_labels) == [0, 1, 2, 3]
    assert np.count_nonzero(blocks == 19) == 19
    np.testing.assert_array_equal(shuffled[distances != 20], labels[distances != 20])


@pytest.mark.parametrize("model_kind", ["raw", "orbit", "harmonic", "orbit_prob"])
def test_checkpoint_reload_preserves_predictions_and_run_metadata(tmp_path: Path, model_kind: str) -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    assignments = {"train": [0], "val": [5], "test": [10]}
    splits = experiment.split_by_distance(dataset["distance_mm"], assignments)
    _, sensor_scales = experiment.prepare_model_inputs(dataset["X_raw"], splits)
    train_val_indices = np.concatenate([splits["train"], splits["val"]])
    channel_map_confirmation = experiment.validate_channel_maps(
        dataset["channel_map"][train_val_indices]
    )
    torch.manual_seed(23)
    model = experiment.build_model(model_kind)
    payload = experiment.create_checkpoint_payload(
        model=model,
        model_kind=model_kind,
        experiment_name="true_training_labels",
        seed=17,
        label_shuffle_seed=None,
        best_epoch=4,
        best_val_macro_f1=0.75,
        best_val_loss=0.6,
        max_epochs=10,
        patience=3,
        sensor_scales=sensor_scales,
        channel_map_confirmation=channel_map_confirmation,
        split_distance_mm={"train": [0], "val": [5], "test": [10]},
        dataset_path=experiment.DEFAULT_DATASET_PATH,
        dataset_sha256_value=experiment.dataset_sha256(experiment.DEFAULT_DATASET_PATH),
        selected_for_final_test=True,
        selected_seed=17,
    )
    checkpoint_path = tmp_path / f"{model_kind}.pt"
    experiment.save_checkpoint(checkpoint_path, payload)
    reloaded_model, reloaded_payload = experiment.load_checkpoint(checkpoint_path)
    windows = dataset["X_raw"][:4]
    normalized = experiment.prepare_checkpoint_inputs(windows, reloaded_payload["preprocessing"])

    original_predictions = experiment.predict_temporal_model(model, normalized)
    reloaded_predictions = experiment.predict_temporal_model(reloaded_model, normalized)

    np.testing.assert_array_equal(reloaded_predictions, original_predictions)
    assert reloaded_payload["class_order"] == ["UP", "DOWN", "LEFT", "RIGHT"]
    assert reloaded_payload["training_hyperparameters"]["seed"] == 17
    assert reloaded_payload["selection_basis"]["validation_macro_f1"] == 0.75
    assert reloaded_payload["architecture"] == experiment.model_architecture(model_kind)
    assert reloaded_payload["preprocessing"]["model_input_transform"] == (
        "harmonic_c4_invariants" if model_kind == "harmonic" else "identity"
    )


def test_new_candidate_paths_are_c4_invariant_on_real_c170_samples() -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    windows = dataset["X_raw"][:24]
    filtered = experiment.filter_windows(windows, sample_rate=250)
    normalized, _ = experiment.calibrate_windows(filtered)

    harmonic = experiment.prepare_candidate_inputs(normalized, "harmonic")
    shifted_harmonic = experiment.prepare_candidate_inputs(
        np.roll(normalized, shift=1, axis=2), "harmonic"
    )
    np.testing.assert_allclose(harmonic, shifted_harmonic, rtol=1e-6, atol=1e-6)
    assert harmonic.shape == (24, 500, 3)

    model = experiment.build_model("orbit_prob")
    model.eval()
    inputs = torch.from_numpy(normalized).permute(0, 2, 1)
    shifted_inputs = torch.roll(inputs, shifts=1, dims=1)
    with torch.no_grad():
        actual = model(inputs)
        shifted = model(shifted_inputs)
        per_shift_probabilities = torch.stack(
            [
                torch.softmax(
                    model.classifier(torch.roll(inputs, shifts=shift, dims=1)), dim=1
                )
                for shift in range(4)
            ]
        )
    torch.testing.assert_close(actual, shifted, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        torch.softmax(actual, dim=1), per_shift_probabilities.mean(dim=0),
        rtol=1e-6, atol=1e-6,
    )


def test_new_candidates_train_on_real_c170_with_validation_only_outputs(tmp_path: Path) -> None:
    experiment = runner()
    output_dir = tmp_path / "c170-new-candidate-smoke"
    assignments = {"train": [0], "val": [5], "test": [10]}

    result = experiment.run_validation_phase(
        experiment.DEFAULT_DATASET_PATH,
        output_dir,
        assignments,
        confirm_physical_order=True,
        seeds=[17],
        max_epochs=1,
        patience=1,
        model_kinds=["harmonic", "orbit_prob"],
    )

    config = result["config"]
    assert config["phase"] == "validation-only"
    assert config["model_kinds"] == ["harmonic", "orbit_prob"]
    assert config["split_distance_mm"] == assignments
    assert config["dataset_sha256"] == experiment.dataset_sha256(
        experiment.DEFAULT_DATASET_PATH
    )
    assert set(config["candidate_selection"]["selected_checkpoints"]) == {
        "true_training_labels:harmonic",
        "true_training_labels:orbit_prob",
        "shuffled_training_labels:harmonic",
        "shuffled_training_labels:orbit_prob",
    }
    assert {row["split"] for row in result["metrics"]["runs"]} == {"train", "val"}
    assert not (output_dir / "final_test_started.json").exists()
    assert not (output_dir / "final_test_metrics.json").exists()
    for checkpoint_path in sorted((output_dir / "checkpoints").glob("*.pt")):
        model, payload = experiment.load_checkpoint(checkpoint_path)
        assert payload["dataset_sha256"] == config["dataset_sha256"]
        assert model.training is False


def test_checkpoint_saving_refuses_to_overwrite_an_existing_file(tmp_path: Path) -> None:
    experiment = runner()
    checkpoint_path = tmp_path / "existing.pt"
    original_bytes = b"preserve this checkpoint"
    checkpoint_path.write_bytes(original_bytes)

    with pytest.raises(FileExistsError):
        experiment.save_checkpoint(checkpoint_path, {})

    assert checkpoint_path.read_bytes() == original_bytes


def test_metrics_report_per_position_and_channel_map() -> None:
    experiment = runner()
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    indices = np.concatenate(
        [np.flatnonzero(dataset["distance_mm"] == distance) for distance in (5, 90)]
    )
    labels = dataset["labels"][indices]
    predictions = (labels + 1) % len(experiment.DIRECTIONS)
    distances = dataset["distance_mm"][indices]
    maps = dataset["channel_map"][indices]

    metrics = experiment.evaluation_metrics(labels, predictions, distances, maps)

    assert metrics["overall"]["prediction_unique_class_count"] == 4
    assert metrics["overall"]["macro_f1"] < 1.0
    assert set(metrics["by_distance_mm"]) == {"5", "90"}
    assert set(metrics["by_distance_mm_and_channel_map"]["5"]) == {"0,2,4,6"}
    assert set(metrics["by_distance_mm_and_channel_map"]["90"]) == {"0,1,2,3"}


def test_validation_only_and_final_test_are_separate_and_final_test_is_one_shot(tmp_path: Path) -> None:
    experiment = runner()
    output_dir = tmp_path / "c170-validation-smoke"
    assignments = {"train": [0], "val": [5], "test": [10]}

    validation_result = experiment.run_validation_phase(
        experiment.DEFAULT_DATASET_PATH,
        output_dir,
        assignments,
        confirm_physical_order=False,
        seeds=[17],
        max_epochs=1,
        patience=1,
    )

    validation_metrics = validation_result["metrics"]
    validation_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert validation_config["phase"] == "validation-only"
    assert validation_config["split_distance_mm"] == assignments
    dataset_sha256 = experiment.dataset_sha256(experiment.DEFAULT_DATASET_PATH)
    assert validation_config["dataset_sha256"] == dataset_sha256
    assert set(validation_config["candidate_selection"]["selected_checkpoints"]) == {
        "true_training_labels:raw",
        "true_training_labels:orbit",
        "shuffled_training_labels:raw",
        "shuffled_training_labels:orbit",
    }
    assert {row["split"] for row in validation_metrics["runs"]} == {"train", "val"}
    assert all("10" not in row["by_distance_mm"] for row in validation_metrics["runs"])
    with (output_dir / "validation_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as predictions_file:
        validation_predictions = list(csv.DictReader(predictions_file))
    assert {row["split"] for row in validation_predictions} == {"train", "val"}
    assert all(row["distance_mm"] != "10" for row in validation_predictions)
    checkpoint_paths = list((output_dir / "checkpoints").glob("*.pt"))
    assert len(checkpoint_paths) == 4
    for checkpoint_path in checkpoint_paths:
        _, checkpoint_payload = experiment.load_checkpoint(checkpoint_path)
        assert checkpoint_payload["dataset_sha256"] == dataset_sha256

    config_path = output_dir / "config.json"
    original_config_bytes = config_path.read_bytes()
    altered_config = dict(validation_config)
    altered_config["dataset_sha256"] = "0" * 64
    config_path.write_text(json.dumps(altered_config, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset.npz SHA-256"):
        experiment.run_final_test_phase(output_dir)
    assert not (output_dir / "final_test_started.json").exists()
    config_path.write_bytes(original_config_bytes)

    final_result = experiment.run_final_test_phase(output_dir)

    assert {row["split"] for row in final_result["metrics"]["runs"]} == {"test"}
    assert all(set(row["by_distance_mm"]) == {"10"} for row in final_result["metrics"]["runs"])
    with (output_dir / "final_test_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as predictions_file:
        final_predictions = list(csv.DictReader(predictions_file))
    dataset = experiment.load_dataset(experiment.DEFAULT_DATASET_PATH)
    test_sample_count = int(np.count_nonzero(dataset["distance_mm"] == 10))
    assert len(final_predictions) == len(final_result["metrics"]["runs"]) * test_sample_count
    final_prediction_bytes = (output_dir / "final_test_predictions.csv").read_bytes()
    with pytest.raises(FileExistsError):
        experiment.run_final_test_phase(output_dir)
    assert (output_dir / "final_test_predictions.csv").read_bytes() == final_prediction_bytes


def test_output_directory_creation_fails_if_path_already_exists() -> None:
    experiment = runner()

    with pytest.raises(FileExistsError):
        experiment.create_output_directory(experiment.PROJECT_DIR / "tests")
