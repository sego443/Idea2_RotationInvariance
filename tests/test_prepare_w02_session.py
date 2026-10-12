import csv
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest


def module():
    return import_module("scripts.data.prepare_w02_session")


def write_recording(path: Path, n_strokes: int, seed: int) -> None:
    """Synthetic OpenBCI raw file: EXG 0-3 active with offset and noise, one burst every 3 s after 3 s of rest."""
    rng = np.random.default_rng(seed)
    rows = 250 * (3 + 3 * n_strokes + 3)
    exg = np.zeros((rows, 8))
    exg[:, :4] = 1000 + rng.normal(0, 5, (rows, 4))
    t = np.arange(250) / 250
    for k in range(n_strokes):
        start = 250 * (3 + 3 * k)
        exg[start:start + 250, :4] += 400 * np.sin(2 * np.pi * 3 * t)[:, None] * np.hanning(250)[:, None]
    header = ["%OpenBCI Raw EXG Data", "%Number of channels = 8", "%Sample Rate = 250 Hz", "%Board = Cyton"]
    columns = ["Sample Index"] + [f"EXG Channel {i}" for i in range(8)]
    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(header) + "\n")
        file.write(", ".join(columns) + "\n")
        for index, values in enumerate(exg):
            file.write(", ".join([str(index)] + [f"{v:.3f}" for v in values]) + "\n")


def write_plan(path: Path, rows: list[dict]) -> None:
    fields = ["prompt", "actual_angle_deg", "accepted"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_segments_labels_in_prompt_order_and_flags_count_mismatch(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    good, bad, rest = "P01_AD1_W01_A090_T1_CUDLR_SRLDU", "P01_AD1_W01_A090_T2_CUDLR_SRLDU", "P01_AD1_W01_A090_REST"
    write_recording(session / f"OpenBCI-RAW-2026-10-20_10-00-00_{good}.txt", 8, 0)
    write_recording(session / f"OpenBCI-RAW-2026-10-20_10-01-00_{bad}.txt", 7, 1)
    write_recording(session / f"OpenBCI-RAW-2026-10-20_10-02-00_{rest}.txt", 0, 2)
    plan = tmp_path / "plan.csv"
    write_plan(plan, [{"prompt": p, "actual_angle_deg": "95", "accepted": "yes"} for p in (rest, good, bad)])

    manifest = module().prepare([session], [plan], tmp_path / "out")

    assert manifest["n_strokes"] == 8
    assert manifest["n_flagged_recordings"] == 1
    with np.load(tmp_path / "out" / "dataset.npz") as archive:
        assert archive["X_raw"].shape == (8, 500, 4)
        np.testing.assert_array_equal(archive["label"], [0, 1, 2, 3, 3, 2, 1, 0])
        assert set(archive["actual_angle_deg"]) == {95}
        np.testing.assert_array_equal(archive["is_prefix"], [True] * 4 + [False] * 4)
        assert archive["rest_X_raw"].shape[0] >= 1
    with pytest.raises(FileExistsError):
        module().prepare([session], [plan], tmp_path / "out")
