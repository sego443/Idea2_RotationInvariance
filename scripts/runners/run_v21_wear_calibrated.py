"""Train and evaluate one v21 wear-calibrated run on existing data (exploratory development protocols).

Protocols:
  fixed_split   train C170 F2-train positions; evaluate the 8 held-out positions (old validation + exposed registered test)
  w01_transfer  train all C170 (including 70 mm); evaluate W01, a different session/day with 12 angles
  all_current   train all C170 and W01; no evaluation (freeze candidate for the next acquisition)
  all_current_plus_w02  as all_current plus the requested W02 parts (--w02-dataset, --w02-parts); no evaluation
  w01_half      train all C170 + W01 angles 0/60/.../300; evaluate W01 angles 30/90/.../330 (stage-2 analog)

--init-checkpoint starts from a saved v21 checkpoint of the same backbone (fine-tuning).

fixed_split additionally scores CALNB modes: held-out positions calibrated with the calibration strokes of a different
held-out position 10-20 mm away (different recordings), a conservative proxy for cross-recording calibration.
Episodes (--episode-support): "same" draws support and query from one group; "neighbor" draws support from one of
the two nearest other groups of the same source. W02 groups always use CAL support and trial queries.

Every evaluated group is scored on repetitions > CAL_SHOTS in ZERO, CAL and UNSUP modes. CAL uses the first
CAL_SHOTS repetitions of each direction in the same group; in C170/W01 these share a recording block with the scored
strokes, so CAL gains here are upper bounds. Training uses fixed epochs; no held-out data selects checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from sklearn.metrics import f1_score
from torch.nn import functional as F

from src import v21_data as data_module
from src import v21_wear_calibrated as v21
from src.c170_shared_scale_preprocessing_v3 import calibrate_windows_shared_scale_v3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = ("fixed_split", "w01_transfer", "all_current", "all_current_plus_w02", "w01_half")
W01_HALF_TRAIN_DEG = (0, 60, 120, 180, 240, 300)
CAL_SHOTS = data_module.EXISTING_CAL_SHOTS
FIXED_SPLIT_CAL_SOURCE = {5: 15, 15: 5, 25: 15, 40: 25, 90: 110, 110: 90, 130: 150, 150: 130}
TAUS = (1.0, 4.0, 16.0)
TRAIN_TAU = 4.0
EPISODE_GROUPS = 4
EPISODE_QUERY_PER_CLASS = 3
SAMPLE_RATE = 250


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bandpass(windows: np.ndarray) -> np.ndarray:
    sos = signal.butter(3, [0.5, 40], btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return signal.sosfiltfilt(sos, windows, axis=1).astype(np.float32)


def split_protocol(protocol: str, w02_dataset: Path | None = None, w02_parts: tuple[str, ...] = ()) -> tuple[dict, dict[str, dict]]:
    c170 = data_module.load_c170()
    if protocol == "fixed_split":
        train = data_module.subset(c170, np.isin(c170["position"], data_module.C170_F2_TRAIN_MM))
        held_out = data_module.C170_OLD_VALIDATION_MM + data_module.C170_EXPOSED_TEST_MM
        evaluations = {"c170_heldout": data_module.subset(c170, np.isin(c170["position"], held_out))}
        return train, evaluations
    w01 = data_module.load_w01()
    if protocol == "w01_transfer":
        return c170, {"w01": w01}
    if protocol == "w01_half":
        in_train = np.isin(w01["position"], W01_HALF_TRAIN_DEG)
        return data_module.concatenate(c170, data_module.subset(w01, in_train)), {"w01_heldout": data_module.subset(w01, ~in_train)}
    if protocol == "all_current":
        return data_module.concatenate(c170, w01), {}
    if protocol == "all_current_plus_w02":
        if w02_dataset is None or not w02_parts:
            raise ValueError("all_current_plus_w02 requires --w02-dataset and --w02-parts")
        w02 = data_module.load_w02(w02_dataset)
        w02 = data_module.subset(w02, np.isin(w02["part"], w02_parts))
        return data_module.concatenate(c170, w01, w02), {}
    raise ValueError(f"unknown protocol: {protocol}")


def macro_f1(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(true, predicted, average="macro", labels=list(range(4))))


def summarize(true: np.ndarray, predicted: np.ndarray, groups: np.ndarray) -> dict:
    recall = [float(np.mean(predicted[true == c] == c)) if np.any(true == c) else None for c in range(4)]
    return {
        "macro_f1": macro_f1(true, predicted),
        "accuracy": float(np.mean(true == predicted)),
        "recall": dict(zip(data_module.CLASSES, recall)),
        "prediction_counts": dict(zip(data_module.CLASSES, np.bincount(predicted, minlength=4).tolist())),
        "per_group_f1": {str(g): macro_f1(true[groups == g], predicted[groups == g]) for g in np.unique(groups)},
    }


@torch.no_grad()
def embed_all(model: v21.WearCalibratedClassifier, values: np.ndarray, device: torch.device) -> torch.Tensor:
    model.eval()
    chunks = [model.embed(torch.as_tensor(values[i:i + 256], device=device)) for i in range(0, len(values), 256)]
    return torch.cat(chunks)


@torch.no_grad()
def evaluate(model, values: np.ndarray, part: dict, device: torch.device,
             cal_source: dict[str, str] | None = None) -> tuple[dict, list[dict]]:
    embeddings = embed_all(model, values, device)
    prototypes = model.global_prototypes()
    labels = torch.as_tensor(part["y"], device=device)
    scored = ~part["is_calibration"]
    predictions = {"zero": v21.cosine_logits(embeddings, prototypes).argmax(1).cpu().numpy()}
    modes = ["cal", "unsup"] + (["calnb"] if cal_source else [])
    for mode in modes:
        for tau in TAUS:
            predictions[f"{mode}_tau{tau:g}"] = np.full(len(values), -1)
    for group in np.unique(part["group"]):
        in_group = part["group"] == group
        support = np.flatnonzero(in_group & part["is_calibration"])
        targets = np.flatnonzero(in_group & scored)
        neighbor = np.flatnonzero((part["group"] == cal_source[group]) & part["is_calibration"]) if cal_source else None
        for tau in TAUS:
            calibrated = v21.calibrated_prototypes(prototypes, embeddings[support], labels[support], tau)
            predictions[f"cal_tau{tau:g}"][targets] = v21.cosine_logits(embeddings[targets], calibrated).argmax(1).cpu().numpy()
            if cal_source:
                calibrated = v21.calibrated_prototypes(prototypes, embeddings[neighbor], labels[neighbor], tau)
                predictions[f"calnb_tau{tau:g}"][targets] = v21.cosine_logits(embeddings[targets], calibrated).argmax(1).cpu().numpy()
            for index in targets:
                others = targets[targets != index]
                refined = v21.transductive_prototypes(prototypes, embeddings[others], tau)
                predictions[f"unsup_tau{tau:g}"][index] = int(v21.cosine_logits(embeddings[index:index + 1], refined).argmax())
    metrics = {mode: summarize(part["y"][scored], pred[scored], part["group"][scored]) for mode, pred in predictions.items()}
    rows = []
    for index in np.flatnonzero(scored):
        row = {"source": part["source"][index], "sample_index": int(part["sample_index"][index]),
               "group": part["group"][index], "block": int(part["block"][index]),
               "repetition": int(part["repetition"][index]), "true": int(part["y"][index])}
        row.update({mode: int(pred[index]) for mode, pred in predictions.items()})
        rows.append(row)
    return metrics, rows


def _take(rng: np.random.Generator, members: np.ndarray, count: int) -> list[int]:
    return list(rng.choice(members, size=min(len(members), count), replace=False)) if len(members) else []


def episode_batch(train: dict, labels: np.ndarray, rng: np.random.Generator, support_mode: str,
                  neighbors: dict[str, list[str]]) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Sample sub-episodes; returns the picked training indices and (support, query) positions into that pick list."""
    groups = train["group"]
    chosen = rng.choice(np.unique(groups), size=min(EPISODE_GROUPS, len(np.unique(groups))), replace=False)
    picks, episodes = [], []
    for group in chosen:
        support, query = [], []
        w02 = train["source"][groups == group][0] == "W02"
        support_group = group
        if not w02 and support_mode == "neighbor":
            support_group = neighbors[group][int(rng.integers(0, min(2, len(neighbors[group]))))]
        for label in range(4):
            if w02:
                support += _take(rng, np.flatnonzero((groups == group) & (labels == label) & train["is_calibration"]), CAL_SHOTS)
                query += _take(rng, np.flatnonzero((groups == group) & (labels == label) & ~train["is_calibration"]),
                               EPISODE_QUERY_PER_CLASS)
            elif support_group == group:
                members = rng.permutation(np.flatnonzero((groups == group) & (labels == label)))
                support += list(members[:CAL_SHOTS])
                query += list(members[CAL_SHOTS:CAL_SHOTS + EPISODE_QUERY_PER_CLASS])
            else:
                support += _take(rng, np.flatnonzero((groups == support_group) & (labels == label)), CAL_SHOTS)
                query += _take(rng, np.flatnonzero((groups == group) & (labels == label)), EPISODE_QUERY_PER_CLASS)
        if not support or not query:
            continue
        offset = len(picks)
        picks += support + query
        episodes.append((np.arange(offset, offset + len(support)),
                         np.arange(offset + len(support), offset + len(support) + len(query))))
    return np.asarray(picks, dtype=int), episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=PROTOCOLS, required=True)
    parser.add_argument("--backbone", choices=v21.BACKBONES, required=True)
    parser.add_argument("--episodic-weight", type=float, default=1.0)
    parser.add_argument("--episode-support", choices=("same", "neighbor"), default="same")
    parser.add_argument("--augment", type=int, choices=(0, 1), default=1)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Fine-tune from this v21 checkpoint")
    parser.add_argument("--w02-dataset", type=Path, default=None)
    parser.add_argument("--w02-parts", nargs="+", choices=("A", "B"), default=())
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

    sources = [Path(__file__), PROJECT_ROOT / "src" / "v21_wear_calibrated.py", PROJECT_ROOT / "src" / "v21_data.py"]
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in sources}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    started = time.monotonic()

    train, evaluations = split_protocol(args.protocol, args.w02_dataset, tuple(args.w02_parts))
    neighbors = data_module.nearest_groups(data_module.subset(train, train["source"] != "W02")) if (train["source"] != "W02").any() else {}
    initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False) if args.init_checkpoint else None
    # Fine-tuning keeps the starting checkpoint's train-fitted scale so inputs match what the encoder learned.
    train_values, shared_scale = calibrate_windows_shared_scale_v3(
        bandpass(train["X"]), scale=None if initial is None else initial["shared_scale"])
    eval_values = {name: calibrate_windows_shared_scale_v3(bandpass(part["X"]), scale=shared_scale)[0]
                   for name, part in evaluations.items()}
    train_labels = data_module.block_permuted_labels(train, args.null_seed) if args.null_seed is not None else train["y"]

    model = v21.WearCalibratedClassifier(args.backbone).to(device)
    init_sha256 = None
    if initial is not None:
        if initial["backbone"] != args.backbone:
            raise ValueError("init checkpoint backbone differs from --backbone")
        model.load_state_dict(initial["state_dict"])
        init_sha256 = sha256(args.init_checkpoint)
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
                picks, episodes = episode_batch(train, train_labels, rng, args.episode_support, neighbors)
                if episodes:
                    pick_index = torch.as_tensor(picks, device=device)
                    episode_values = x_train[pick_index]
                    if args.augment:
                        episode_values = v21.augment(episode_values, generator)
                    episode = v21.support_query_loss(model, episode_values, y_train[pick_index], episodes, TRAIN_TAU)
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
            cal_source = {f"C170_P{target:03d}": f"C170_P{source:03d}" for target, source in FIXED_SPLIT_CAL_SOURCE.items()}
        metrics[name], rows = evaluate(model, eval_values[name], part, device, cal_source)
        all_rows += [{"evaluation": name, **row} for row in rows]

    with (args.output_dir / "learning_curve.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    if all_rows:
        with (args.output_dir / "predictions.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone, "shared_scale": shared_scale,
                "classes": data_module.CLASSES}, args.output_dir / "checkpoint.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    manifest = {
        "arguments": {key: str(value) for key, value in vars(args).items()},
        "status": "exploratory development run on existing data",
        "train_groups": sorted(set(train["group"].tolist())),
        "n_train": int(len(train_values)),
        "evaluations": {name: {"groups": sorted(set(part["group"].tolist())), "n_scored": int((part["repetition"] > CAL_SHOTS).sum())}
                        for name, part in evaluations.items()},
        "shared_scale": shared_scale,
        "init_checkpoint_sha256": init_sha256,
        "cal_shots_per_direction": CAL_SHOTS,
        "taus": TAUS,
        "train_tau": TRAIN_TAU,
        "data_sha256": {"c170": sha256(data_module.C170_PATH),
                        **({"w01": sha256(data_module.W01_PATH)} if args.protocol != "fixed_split" else {})},
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
