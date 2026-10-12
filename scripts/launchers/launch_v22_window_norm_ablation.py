"""Launch the v22 window/normalization ablation grid (exploratory; B2 signed CNN, same-group episodes).

Variants (window, norm): the v21 baseline, five window variants under the shared scale, three normalizations on
the full window and two outbound-valley combinations. Each runs on fixed_split and w01_transfer with seeds 17/29/43.
Matched block-permuted nulls (seed 17, null seed 17) run for every variant on fixed_split and for the key variants
on w01_transfer. Real runs go first, in priority order.

--resume skips runs that completed; an incomplete run directory is preserved and the run is repeated as
<name>__retry<k>.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKBONE = "b2_signed_cnn"
SEEDS = (17, 29, 43)
NULL_SEED = 17
VARIANTS = (
    ("full", "shared"), ("outbound_valley", "shared"), ("full", "window_global"), ("full", "group_sensor"),
    ("full", "window_sensor"), ("prefix_250", "shared"), ("return_only", "shared"), ("prefix_300", "shared"),
    ("prefix_200", "shared"), ("outbound_valley", "window_global"), ("outbound_valley", "group_sensor"),
)
W01_NULL_VARIANTS = VARIANTS[:4]
PROTOCOLS = ("fixed_split", "w01_transfer")


def planned_runs() -> list[tuple[str, list[str]]]:
    runs = []
    for window, norm in VARIANTS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                runs.append((f"{protocol}__{window}__{norm}__s{seed}",
                             ["--protocol", protocol, "--window", window, "--norm", norm, "--seed", str(seed)]))
    for window, norm in VARIANTS:
        for protocol in PROTOCOLS:
            if protocol == "w01_transfer" and (window, norm) not in W01_NULL_VARIANTS:
                continue
            runs.append((f"{protocol}__{window}__{norm}__s{NULL_SEED}__null{NULL_SEED}",
                         ["--protocol", protocol, "--window", window, "--norm", norm, "--seed", str(NULL_SEED),
                          "--null-seed", str(NULL_SEED)]))
    return runs


def run_one(job: tuple[Path, list[str], str, int]) -> int:
    out, arguments, device, threads = job
    command = [sys.executable, "-m", "scripts.runners.run_v22_window_norm_ablation", "--backbone", BACKBONE,
               *arguments, "--device", device, "--cpu-threads", str(threads), "--output-dir", str(out)]
    with out.with_suffix(".log").open("w") as handle:
        return subprocess.run(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cpu", help="Torch device for every run, e.g. cuda or cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    jobs = []
    for name, arguments in planned_runs():
        attempts = [a for a in (args.output_root / name, *sorted(args.output_root.glob(f"{name}__retry*"))) if a.is_dir()]
        if any((a / "RUN_COMPLETE.json").exists() for a in attempts):
            continue
        out = args.output_root / (name if not attempts else f"{name}__retry{len(attempts)}")
        jobs.append((out, arguments, args.device, args.cpu_threads))
    with (args.output_root / "commands.jsonl").open("a") as file:
        for out, arguments, _, _ in jobs:
            file.write(json.dumps({"output_dir": str(out), "arguments": arguments, "device": args.device}) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run_one, jobs))
    failed = [str(out) for (out, *_), code in zip(jobs, codes) if code != 0]
    (args.output_root / ("GRID_FAILED.json" if failed else "GRID_COMPLETE.json")).write_text(
        json.dumps({"launched": len(jobs), "failed": failed}, indent=2))
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {failed}")


if __name__ == "__main__":
    main()
