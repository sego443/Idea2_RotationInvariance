from collections import Counter
from importlib import import_module

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


def test_angles_cover_every_stratum_once() -> None:
    module = generator()
    angles = module.stratified_angles(np.random.default_rng(1), 24)
    assert sorted(angle // 15 for angle in angles) == list(range(24))
    assert all(angle % module.ANGLE_RESOLUTION_DEG == 0 for angle in angles)


def test_plan_structure_is_deterministic_and_unique() -> None:
    module = generator()
    rows = module.build_plan("P01", "A", 1, 24, 3, 2, 3, seed=7)
    assert rows == module.build_plan("P01", "A", 1, 24, 3, 2, 3, seed=7)
    assert len(rows) == 24 * 4
    assert len({row["prompt"] for row in rows}) == len(rows)
    for wear in range(1, 25):
        wear_rows = [row for row in rows if row["wear"] == wear]
        assert [row["recording"] for row in wear_rows] == ["REST", "T1", "T2", "T3"]
        assert len({row["scheduled_angle_deg"] for row in wear_rows}) == 1
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
        module.stratified_angles(np.random.default_rng(0), 3)


def test_part_b_continues_wear_numbering() -> None:
    rows = generator().build_plan("P01", "B", 1, 10, 3, 2, 3, seed=3, first_wear=11)
    assert sorted({row["wear"] for row in rows}) == list(range(11, 21))
    assert rows[0]["prompt"].startswith("P01_BD1_W")
