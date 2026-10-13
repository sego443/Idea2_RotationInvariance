"""Fill the W02 P01 Part A plan from the files on disk, following protocol section 6.

Each planned row records the first take of its recording. Every later take is appended as a retry row whose
retry_of is the previous take's row. Files ending in .rejected are accepted=no; the single non-rejected take is
accepted=yes. actual_tape_mm is the M field of the filename (operator statement, 2026-10-14: M is the actual
tape reading). Rejection reasons and strap holes were not recorded. Files without a filename label are listed
on stdout and not entered. The source plan is not modified; the output file must not exist.
"""
import csv
import re
from pathlib import Path

root = Path("data/raw/w02_data/P01_AD1")
plan = Path("data/w02_plans/P01_AD1_timed/w02_P01_AD1_plan.csv")
out = plan.with_name("w02_P01_AD1_plan_completed.csv")
assert not out.exists(), out
rows = list(csv.DictReader(plan.open(encoding="utf-8", newline="")))
fields = list(rows[0])
name_re = re.compile(r"OpenBCI-RAW-(?P<time>\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)_(?P<label>.+?)\.txt(?P<rej>\.rejected)?$")
takes, unlabeled = {}, []
for path in sorted(root.glob("*/OpenBCI-RAW-*")):
    match = name_re.match(path.name)
    if match is None:
        unlabeled.append(path)
        continue
    takes.setdefault(match["label"], []).append((match["time"], path.name, bool(match["rej"])))
unknown = set(takes) - {row["filename_label"] for row in rows}
assert not unknown, unknown

retries = []
next_row = len(rows) + 1
for row in rows:
    ordered = sorted(takes[row["filename_label"]])
    assert [rej for _, _, rej in ordered] == [True] * (len(ordered) - 1) + [False], row["filename_label"]
    tape = int(re.search(r"_M(\d{3})_", row["filename_label"])[1])
    assert tape == int(row["scheduled_tape_mm"])
    previous = row
    for index, (_, name, rejected) in enumerate(ordered):
        take = row if index == 0 else {**row, "row": str(next_row), "retry_of": previous["row"]}
        if index:
            next_row += 1
            retries.append(take)
        take.update(actual_tape_mm=str(tape), actual_filename=name, accepted="no" if rejected else "yes",
                    notes="rejected at acquisition; reason not recorded" if rejected else "")
        previous = take

with out.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows + retries)
print(f"planned rows {len(rows)}, retry rows {len(retries)}, accepted {sum(r['accepted'] == 'yes' for r in rows + retries)}")
for path in unlabeled:
    print("UNLABELED", path.parent.name, path.name, path.stat().st_size)
