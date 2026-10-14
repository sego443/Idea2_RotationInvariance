import csv
from pathlib import Path

import pytest

from scripts.data.complete_w02_plan import complete


def write_plan(path: Path, labels: list[str]) -> None:
    fields = ["row", "scheduled_tape_mm", "filename_label", "actual_tape_mm", "actual_filename", "accepted"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"row": i, "scheduled_tape_mm": "80", "filename_label": label} for i, label in enumerate(labels, 1))


def test_fills_single_takes_and_requires_a_choice_for_duplicates(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    rest, t1 = "P01_BD1_W07_M080_REST", "P01_BD1_W07_M080_T1"
    for name in (f"OpenBCI-RAW-2026-10-14_17-03-27_{rest}.txt", f"OpenBCI-RAW-2026-10-14_16-56-20_{t1}.txt",
                 f"OpenBCI-RAW-2026-10-14_17-03-50_{t1}.txt", "OpenBCI-RAW-2026-10-14_16-51-00.txt"):
        (session / name).touch()
    plan = tmp_path / "plan.csv"
    write_plan(plan, [rest, t1])

    with pytest.raises(ValueError, match="expected one take"):
        complete(plan, [session], tmp_path / "out.csv", {})
    rows = complete(plan, [session], tmp_path / "out.csv", {t1: f"OpenBCI-RAW-2026-10-14_17-03-50_{t1}.txt"})

    assert [r["actual_filename"] for r in rows] == [f"OpenBCI-RAW-2026-10-14_17-03-27_{rest}.txt",
                                                    f"OpenBCI-RAW-2026-10-14_17-03-50_{t1}.txt"]
    assert {r["accepted"] for r in rows} == {"yes"} and {r["actual_tape_mm"] for r in rows} == {"80"}
    with pytest.raises(FileExistsError):
        complete(plan, [session], tmp_path / "out.csv", {})
