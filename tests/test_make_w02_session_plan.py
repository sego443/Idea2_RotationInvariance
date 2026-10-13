from collections import Counter
from importlib import import_module
import json

import numpy as np
import pytest


def generator():
    return import_module("scripts.data.make_w02_session_plan")


def test_sequences_are_balanced_with_bounded_runs() -> None:
    module = generator()
    rng = np.random.default_rng(0)
    for _ in range(200):
        sequence = module.balanced_sequence(rng, 3)
        assert Counter(sequence) == {"U": 3, "D": 3, "L": 3, "R": 3}
        assert all(len(set(sequence[i:i + 3])) > 1 for i in range(len(sequence) - 2))


def test_tape_positions_cover_every_stratum_once() -> None:
    module = generator()
    for n_wears in (6, 12, 17):
        positions = module.stratified_tape_positions(np.random.default_rng(1), n_wears)
        width = module.TAPE_CIRCUMFERENCE_MM / n_wears
        assert sorted(int(m // width) for m in positions) == list(range(n_wears))
        assert all(m % module.TAPE_RESOLUTION_MM == 0 and 0 <= m < 170 for m in positions)


def test_band_angle_follows_the_s3_s0_midpoint_reading() -> None:
    module = generator()
    # S0 is 21.25 mm past the S3/S0 midpoint; the dorsal midline is at 10 mm.
    assert module.band_angle_deg(10 - 21.25 + 170) == pytest.approx(0.0)
    assert module.band_angle_deg(0) == pytest.approx(11.25 * 360 / 170)
    assert module.band_angle_deg(31.25) == pytest.approx(90.0)
    assert module.band_angle_deg(158.75) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        module.band_angle_deg(170)


def test_plan_structure_is_deterministic_and_unique() -> None:
    module = generator()
    rows = module.build_plan("P01", "A", 1, 6, 6, 2, 3, seed=7)
    assert rows == module.build_plan("P01", "A", 1, 6, 6, 2, 3, seed=7)
    assert len(rows) == 6 * 7
    assert len({row["prompt"] for row in rows}) == len(rows)
    for wear in range(1, 7):
        wear_rows = [row for row in rows if row["wear"] == wear]
        assert [row["recording"] for row in wear_rows] == ["REST", "T1", "T2", "T3", "T4", "T5", "T6"]
        assert len({row["scheduled_tape_mm"] for row in wear_rows}) == 1
        tape = wear_rows[0]["scheduled_tape_mm"]
        assert f"_W{wear:02d}_M{tape:03d}_" in wear_rows[0]["prompt"]
        assert wear_rows[0]["scheduled_angle_deg"] == round(module.band_angle_deg(tape))
        for row in wear_rows[1:]:
            assert Counter(row["prefix"]) == {"U": 2, "D": 2, "L": 2, "R": 2}
            assert Counter(row["body"]) == {"U": 3, "D": 3, "L": 3, "R": 3}
            assert row["sequence"] == row["prefix"] + row["body"] and row["n_prefix"] == 8
            assert row["prompt"].endswith(f"_C{row['prefix']}_S{row['body']}")


def test_invalid_arguments_are_rejected() -> None:
    module = generator()
    with pytest.raises(ValueError):
        module.build_plan("P01", "OTHER", 1, 24, 3, 2, 3, seed=0)
    with pytest.raises(ValueError):
        module.build_plan("P_01", "A", 1, 24, 3, 2, 3, seed=0)
    with pytest.raises(ValueError):
        module.stratified_tape_positions(np.random.default_rng(0), 3)


def test_part_b_continues_wear_numbering() -> None:
    rows = generator().build_plan("P01", "B", 1, 6, 6, 2, 3, seed=3, first_wear=7)
    assert sorted({row["wear"] for row in rows}) == list(range(7, 13))
    assert rows[0]["prompt"].startswith("P01_BD1_W")


def test_widget_prompts_use_action_strings_and_independent_filenames() -> None:
    module = generator()
    rows = module.build_plan("P01", "A", 1, 6, 6, 2, 3, seed=7)
    document = json.loads(json.dumps(module.build_widget_prompts(rows)))
    assert document["version"] == 1
    assert len(document["prompts"]) == len(rows)
    for row, prompt in zip(rows, document["prompts"]):
        assert prompt["filename_label"] == row["filename_label"]
        assert "_C" not in prompt["filename_label"]
        assert set(prompt) == {"filename_label", "bpm", "prepare_seconds", "actions", "finish_seconds"}
        assert all(isinstance(action, str) and action for action in prompt["actions"])
        duration = prompt["prepare_seconds"] + len(prompt["actions"]) * 60 / prompt["bpm"] + prompt["finish_seconds"]
        assert duration == row["expected_duration_s"]
        if row["kind"] == "rest":
            assert len(prompt["actions"]) == 1 and duration == 20
        else:
            assert prompt["bpm"] == 20 and duration == 66
            assert prompt["actions"] == [f"{module.DIRECTION_CODES[code]} - return to centre" for code in row["sequence"]]
