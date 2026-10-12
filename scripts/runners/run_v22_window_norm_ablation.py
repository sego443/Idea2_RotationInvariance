"""One v22 window/normalization ablation run: the v21 wear-calibrated recipe with a different input preprocessing.

Training, episodes, evaluation modes (ZERO, CAL, UNSUP, CALNB) and outputs are the v21 runner's
(scripts/runners/run_v21_wear_calibrated.py, imported unchanged); only the input changes:
--window selects the visible time span and --norm the amplitude normalization (see src/v22_window_norm.py).
`--window full --norm shared` reproduces the v21 preprocessing. Only the fixed_split and w01_transfer development
protocols are offered; every result is exploratory (C170 held-out positions and W01 were seen before).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.runners import run_v21_wear_calibrated as base
from src import v21_data as data_module
from src import v21_wear_calibrated as v21
from src import v22_window_norm as wn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = ("fixed_split", "w01_transfer")


def prepare(part: dict, window: str, norm: str, shared_scale: float | None) -> tuple[np.ndarray, float | None, dict]:
    values, mask, stats = wn.apply_window(part["X"], window)
    if norm == "shared" and shared_scale is None:
        shared_scale = wn.fit_shared_scale(values, mask)
    return wn.normalize(values, mask, norm, part["wear_group"], shared_scale), shared_scale, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--backbone", choices=v21.BACKBONES, required=True)
    parser.add_argument("--window", choices=wn.WINDOWS, required=True)
    parser.add_argument("--norm", choices=wn.NORMS, required=True)
    parser.add_argument("--episodic-weight", type=float, default=1.0)
    parser.add_argument("--episode-support", choices=("same", "neighbor"), default="same")
    parser.add_argument("--augment", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--null-seed", type=int, default=None, help="Train on block-permuted labels (matched null)")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; must not exist")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.episodic_weight < 0:
        raise ValueError("epochs, batch size and episodic weight are out of range")

    sources = [Path(__file__), PROJECT_ROOT / "scripts" / "runners" / "run_v21_wear_calibrated.py",
               PROJECT_ROOT / "src" / "v21_wear_calibrated.py", PROJECT_ROOT / "src" / "v21_data.py",
               PROJECT_ROOT / "src" / "v22_window_norm.py"]
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): base.sha256(p) for p in sources}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    started = time.monotonic()

    train, evaluations = base.split_protocol(args.protocol)
    neighbors = data_module.nearest_groups(train)
    train_values, shared_scale, train_stats = prepare(train, args.window, args.norm, None)
    eval_values, window_stats = {}, {"train": train_stats}
    for name, part in evaluations.items():
        eval_values[name], _, window_stats[name] = prepare(part, args.window, args.norm, shared_scale)
    train_labels = data_module.block_permuted_labels(train, args.null_seed) if args.null_seed is not None else train["y"]

    model = v21.WearCalibratedClassifier(args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_train = torch.as_tensor(train_values, device=device)
    y_train = torch.as_tensor(train_labels, device=device)
    curve = []
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_values))
        totals = {"ce": 0.0, "episode": 0.0, "steps": 0}
        for start in range(0, len(order), args.batch_size):
            index = torch.as_tensor(order[start:start + args.batch_size], device=device)
            batch = x_train[index]
            if args.augment:
                batch = v21.augment(batch, generator)
            loss = F.cross_entropy(model(batch), y_train[index])
            totals["ce"] += float(loss.detach())
            if args.episodic_weight > 0:
                picks, episodes = base.episode_batch(train, train_labels, rng, args.episode_support, neighbors)
                if episodes:
                    pick_index = torch.as_tensor(picks, device=device)
                    episode_values = x_train[pick_index]
                    if args.augment:
                        episode_values = v21.augment(episode_values, generator)
                    episode = v21.support_query_loss(model, episode_values, y_train[pick_index], episodes, base.TRAIN_TAU)
                    loss = loss + args.episodic_weight * episode
                    totals["episode"] += float(episode.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["steps"] += 1
        curve.append({"epoch": epoch + 1, "ce": totals["ce"] / totals["steps"],
                      "episode": totals["episode"] / totals["steps"], "elapsed_s": round(time.monotonic() - started, 1)})

    metrics, all_rows = {}, []
    for name, part in evaluations.items():
        cal_source = None
        if args.protocol == "fixed_split":
            cal_source = {f"C170_P{t:03d}": f"C170_P{s:03d}" for t, s in base.FIXED_SPLIT_CAL_SOURCE.items()}
        metrics[name], rows = base.evaluate(model, eval_values[name], part, device, cal_source)
        all_rows += [{"evaluation": name, **row} for row in rows]

    with (args.output_dir / "learning_curve.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    with (args.output_dir / "predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone, "window": args.window, "norm": args.norm,
                "shared_scale": shared_scale, "classes": data_module.CLASSES}, args.output_dir / "checkpoint.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest = {
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "status": "exploratory v22 window/normalization ablation on existing data",
        "train_groups": sorted(set(train["group"].tolist())),
        "n_train": int(len(train_values)),
        "evaluations": {name: {"groups": sorted(set(part["group"].tolist())), "n_scored": int((~part["is_calibration"]).sum())}
                        for name, part in evaluations.items()},
        "window_statistics": window_stats,
        "shared_scale": shared_scale,
        "cal_shots_per_direction": base.CAL_SHOTS,
        "taus": base.TAUS,
        "train_tau": base.TRAIN_TAU,
        "data_sha256": {"c170": base.sha256(data_module.C170_PATH),
                        **({"w01": base.sha256(data_module.W01_PATH)} if args.protocol != "fixed_split" else {})},
        "source_sha256": source_hashes,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "machine": platform.machine(),
                        "pid": os.getpid()},
        "runtime_s": round(time.monotonic() - started, 1),
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "RUN_COMPLETE.json").write_text(json.dumps({"complete": True, "runtime_s": manifest["runtime_s"]}))
    summary = {name: {mode: round(m["macro_f1"], 4) for mode, m in modes.items()} for name, modes in metrics.items()}
    print(json.dumps({"output": str(args.output_dir), "runtime_s": manifest["runtime_s"], "macro_f1": summary}, indent=1))


if __name__ == "__main__":
    main()
