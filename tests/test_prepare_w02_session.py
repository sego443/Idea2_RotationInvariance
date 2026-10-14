import csv
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


def module():
    return import_module("scripts.data.prepare_w02_session")


def test_file_lookup_uses_independent_label_and_preserves_rejected_files(tmp_path: Path) -> None:
    label = "P01_AD1_W01_M030_T1"
    accepted = tmp_path / f"OpenBCI-RAW-2026-10-20_10-00-00_{label}.txt"
    rejected = tmp_path / f"OpenBCI-RAW-2026-10-20_09-00-00_{label}.txt.rejected"
    accepted.touch()
    rejected.touch()
    assert module().find_file([tmp_path], label) == accepted
    assert rejected.exists()
    with pytest.raises(ValueError, match="invalid filename label"):
        module().find_file([tmp_path], "*")
    second = tmp_path / f"OpenBCI-RAW-2026-10-20_09-30-00_{label}.txt"
    second.touch()
    with pytest.raises(ValueError, match="exactly one"):
        module().find_file([tmp_path], label)
    assert module().find_file([tmp_path], label, second.name) == second
    with pytest.raises(ValueError, match="does not match"):
        module().find_file([tmp_path], label, "OpenBCI-RAW-2026-10-20_09-30-00_P01_AD1_W01_M030_T2.txt")


def write_recording(path: Path, n_strokes: int, seed: int, silent: tuple[int, ...] = ()) -> None:
    """Synthetic OpenBCI raw file: EXG 0-3 active with offset and noise, one burst starting 0.5 s into each 3 s slot
    after 3 s of rest; slots listed in silent get no burst."""
    rng = np.random.default_rng(seed)
    rows = 250 * (3 + 3 * n_strokes + 3)
    exg = np.zeros((rows, 8))
    exg[:, :4] = 1000 + rng.normal(0, 5, (rows, 4))
    t = np.arange(250) / 250
    for k in range(n_strokes):
        if k in silent:
            continue
        start = 250 * (3 + 3 * k) + 125
        exg[start:start + 250, :4] += 400 * np.sin(2 * np.pi * 3 * t)[:, None] * np.hanning(250)[:, None]
    header = ["%OpenBCI Raw EXG Data", "%Number of channels = 8", "%Sample Rate = 250 Hz", "%Board = Cyton"]
    columns = ["Sample Index"] + [f"EXG Channel {i}" for i in range(8)]
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(header) + "\n")
        file.write(", ".join(columns) + "\n")
        for index, values in enumerate(exg):
            file.write(", ".join([str(index)] + [f"{v:.3f}" for v in values]) + "\n")


def write_plan(path: Path, rows: list[dict]) -> None:
    fields = ["prompt", "actual_tape_mm", "accepted"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_labels_follow_the_grid_and_flags_only_truncated_recordings(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    good, faint, short, rest = ("P01_AD1_W01_M030_T1_CUDLR_SRLDU", "P01_AD1_W01_M030_T2_CUDLR_SRLDU",
                                "P01_AD1_W01_M030_T3_CUDLR_SRLDU", "P01_AD1_W01_M030_REST")
    write_recording(session / f"OpenBCI-RAW-2026-10-20_10-00-00_{good}.txt", 8, 0)
    write_recording(session / f"OpenBCI-RAW-2026-10-20_10-01-00_{faint}.txt", 8, 1, silent=(4,))
    write_recording(session / f"OpenBCI-RAW-2026-10-20_10-02-00_{short}.txt", 6, 3)
    write_recording(session / f"OpenBCI-RAW-2026-10-20_10-03-00_{rest}.txt", 0, 2)
    plan = tmp_path / "plan.csv"
    write_plan(plan, [{"prompt": p, "actual_tape_mm": "31.25", "accepted": "yes"} for p in (rest, good, faint, short)])

    manifest = module().prepare([session], [plan], tmp_path / "out")

    assert manifest["n_strokes"] == 16
    assert manifest["n_flagged_recordings"] == 1
    with np.load(tmp_path / "out" / "dataset.npz") as archive:
        assert archive["X_raw"].shape == (16, 500, 4)
        np.testing.assert_array_equal(archive["label"], [0, 1, 2, 3, 3, 2, 1, 0] * 2)
        np.testing.assert_array_equal(archive["slot_has_peak"], [True] * 12 + [False] + [True] * 3)
        offsets = archive["slot_offset_s"]
        assert ((offsets >= 0) & (offsets < 3)).all()
        np.testing.assert_allclose(np.delete(offsets, 12), 1.0, atol=0.1)
        assert set(archive["scheduled_tape_mm"]) == {30}
        assert set(archive["actual_tape_mm"]) == {31.25}
        assert set(archive["actual_angle_deg"]) == {90}
        assert set(archive["scheduled_angle_deg"]) == {round((30 + 11.25) * 360 / 170)}
        np.testing.assert_array_equal(archive["is_prefix"], ([True] * 4 + [False] * 4) * 2)
        assert archive["rest_X_raw"].shape[0] >= 1
    assert "flagged_too_short" in (tmp_path / "out" / "recordings.csv").read_text()
    with pytest.raises(FileExistsError):
        module().prepare([session], [plan], tmp_path / "out")
