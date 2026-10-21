"""Launch the W02 causal-stream exploration families in parallel (development wears only).

Each job runs `python -m <module> --output-dir <root>/<job> [extra args]` with one BLAS thread, writing stdout and
stderr to <root>/<job>.log and the exit code to <root>/<job>.exit.json. The root directory must not exist.
Works on Windows and POSIX. Example:

    python -m scripts.analysis.launch_w02_stream_families --root outputs/w02_explore_20261015/stream_families_v1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

JOBS = {
    "A_adapt": ("scripts.analysis.explore_w02_stream_adapt", []),
    "B_features": ("scripts.analysis.explore_w02_stream_features", []),
    "C_fewer_b0.1": ("scripts.analysis.explore_w02_stream_fewer", ["--beta", "0.1"]),
    "C_fewer_b0.05": ("scripts.analysis.explore_w02_stream_fewer", ["--beta", "0.05"]),
    "D_feedback_b0.1": ("scripts.analysis.explore_w02_stream_feedback", ["--beta", "0.1"]),
    "D_feedback_b0.05": ("scripts.analysis.explore_w02_stream_feedback", ["--beta", "0.05"]),
    "E_drift": ("scripts.analysis.explore_w02_stream_drift", []),
    "F_usage": ("scripts.analysis.explore_w02_stream_usage", []),
    "G_robust": ("scripts.analysis.explore_w02_stream_robust", []),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", nargs="*", choices=list(JOBS), default=list(JOBS))
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running = {}
    for job in args.jobs:
        module, extra = JOBS[job]
        command = [sys.executable, "-u", "-m", module, "--output-dir", str(args.root / job), *extra]
        if args.dataset is not None:
            command += ["--dataset", str(args.dataset)]
        log = (args.root / f"{job}.log").open("x")
        running[job] = (subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log,
                        time.time(), command)
        print(f"started {job}: {' '.join(command)}", flush=True)
    while running:
        for job, (process, log, start, command) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            record = {"job": job, "command": command, "exit_code": code, "elapsed_seconds": round(time.time() - start, 1),
                      "finished_utc": datetime.now(timezone.utc).isoformat()}
            (args.root / f"{job}.exit.json").write_text(json.dumps(record, indent=2))
            print(f"finished {job}: exit {code} after {record['elapsed_seconds']} s", flush=True)
            del running[job]
        time.sleep(2)


if __name__ == "__main__":
    main()
