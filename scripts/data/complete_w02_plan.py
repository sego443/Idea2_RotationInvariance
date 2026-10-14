"""Fill a generated W02 plan with its accepted takes from the files on disk.

Every planned row receives the single labeled take of its recording (rejected takes are deleted and get no rows).
A label with more than one take on disk is an error unless --choose names the take to use; the unchosen takes are
listed on stdout and not entered. actual_tape_mm is the M field of the filename (operator statement, 2026-10-14:
M is the actual tape reading). Files without a filename label are listed on stdout and not entered. The source
plan is not modified; the output file must not exist.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

NAME_RE = re.compile(r"OpenBCI-RAW-\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_(?P<label>.+)\.txt$")


def complete(plan: Path, session_dirs: list[Path], output: Path, choices: dict[str, str]) -> list[dict]:
    if output.exists():
        raise FileExistsError(f"Output file already exists: {output}")
    with plan.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    takes: dict[str, list[str]] = {}
    for path in sorted(p for d in session_dirs for p in d.glob("OpenBCI-RAW-*.txt")):
        match = NAME_RE.match(path.name)
        if match is None:
            print("UNLABELED", path.parent.name, path.name, path.stat().st_size)
            continue
        takes.setdefault(match["label"], []).append(path.name)
    labels = {row["filename_label"] for row in rows}
    if unknown := set(takes) - labels:
        raise ValueError(f"files with labels outside the plan: {sorted(unknown)}")
    if unused := set(choices) - labels:
        raise ValueError(f"--choose labels outside the plan: {sorted(unused)}")
    for row in rows:
        label = row["filename_label"]
        found = takes.get(label, [])
        if label in choices:
            if choices[label] not in found:
                raise ValueError(f"chosen file for {label} not found: {choices[label]}")
            name = choices[label]
        elif len(found) == 1:
            name = found[0]
        else:
            raise ValueError(f"{label}: expected one take, found {found}; pass --choose {label}=<file>")
        for other in sorted(set(found) - {name}):
            print("NOT ENTERED", label, other)
        tape = int(re.search(r"_M(\d{3})_", label)[1])
        if tape != int(row["scheduled_tape_mm"]):
            raise ValueError(f"{label}: M field differs from the scheduled tape reading")
        row.update(actual_tape_mm=str(tape), actual_filename=name, accepted="yes")
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--choose", action="append", default=[], metavar="LABEL=FILE")
    args = parser.parse_args()
    choices = dict(item.split("=", 1) for item in args.choose)
    rows = complete(args.plan, args.session_dir, args.output, choices)
    print(f"{len(rows)} planned rows filled: {args.output}")


if __name__ == "__main__":
    main()
