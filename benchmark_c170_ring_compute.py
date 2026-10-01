from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "src"))

import run_c170_experiment as c170
from c170_ring_representation import (
    RingRepresentationClassifier,
    build_balanced_batches,
    cross_position_supervised_contrastive_loss,
)
from run_c170_ring_experiment import (
    CONTRASTIVE_TEMPERATURE,
    CONTRASTIVE_WEIGHT,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C170 ring-model device and thread benchmark")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-repeats", type=int, default=6)
    parser.add_argument("--validation-repeats", type=int, default=3)
    return parser.parse_args(argv)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _median_seconds(action, repeats: int, device: torch.device) -> tuple[float, list[float]]:
    timings = []
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter()
        action()
        _synchronize(device)
        timings.append(time.perf_counter() - started)
    return float(np.median(timings)), timings


def _training_update(
    model: RingRepresentationClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    distances: np.ndarray,
    blocks: np.ndarray,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, projection = model.forward_with_projection(waveforms)
    ce_loss = criterion(logits, labels)
    contrastive_loss = cross_position_supervised_contrastive_loss(
        projection,
        labels,
        distances,
        blocks,
        temperature=CONTRASTIVE_TEMPERATURE,
    )
    total_loss = ce_loss + CONTRASTIVE_WEIGHT * contrastive_loss
    if not torch.isfinite(total_loss):
        raise RuntimeError("A real C170 training batch produced non-finite loss")
    total_loss.backward()
    optimizer.step()


def _benchmark_device(
    *,
    device: torch.device,
    cpu_threads: int,
    state_dict: dict[str, torch.Tensor],
    train_inputs: torch.Tensor,
    train_labels: torch.Tensor,
    train_distances: np.ndarray,
    train_blocks: np.ndarray,
    val_inputs: torch.Tensor,
    training_repeats: int,
    validation_repeats: int,
) -> dict[str, object]:
    torch.set_num_threads(cpu_threads)
    model = RingRepresentationClassifier("main")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    device_train_inputs = train_inputs.to(device)
    device_train_labels = train_labels.to(device)
    device_val_inputs = val_inputs.to(device)

    for _ in range(2):
        _training_update(
            model,
            optimizer,
            criterion,
            device_train_inputs,
            device_train_labels,
            train_distances,
            train_blocks,
        )
    _synchronize(device)

    train_median, train_timings = _median_seconds(
        lambda: _training_update(
            model,
            optimizer,
            criterion,
            device_train_inputs,
            device_train_labels,
            train_distances,
            train_blocks,
        ),
        training_repeats,
        device,
    )

    model.eval()
    with torch.no_grad():
        val_logits = model(device_val_inputs)
    if val_logits.shape != (len(val_inputs), 4):
        raise RuntimeError(f"Invalid C170 validation output shape: {tuple(val_logits.shape)}")

    def validation_forward() -> None:
        with torch.no_grad():
            model(device_val_inputs)

    val_median, val_timings = _median_seconds(
        validation_forward,
        validation_repeats,
        device,
    )
    return {
        "device": str(device),
        "cpu_threads": cpu_threads,
        "train_update_repeats": training_repeats,
        "train_update_seconds_median": train_median,
        "train_update_seconds_each": train_timings,
        "validation_forward_repeats": validation_repeats,
        "validation_forward_seconds_median": val_median,
        "validation_forward_seconds_each": val_timings,
        "validation_window_count": int(len(val_inputs)),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.training_repeats < 1 or args.validation_repeats < 1:
        raise ValueError("Benchmark repeats must be positive")
    default_threads = torch.get_num_threads()
    dataset_path = c170.DEFAULT_DATASET_PATH
    dataset_hash = c170.dataset_sha256(dataset_path)
    dataset = c170.load_dataset(dataset_path)
    splits = c170.split_by_distance(
        dataset["distance_mm"],
        {
            "train": (0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160),
            "val": (5, 40, 90, 130),
            "test": (15, 25, 110, 150),
        },
    )
    normalized, _ = c170.prepare_model_inputs(dataset["X_raw"], splits)
    batches = build_balanced_batches(
        splits["train"],
        dataset["labels"],
        dataset["distance_mm"],
        dataset["block"],
        seed=17,
    )
    batch_indices = batches[0]
    train_offsets = np.searchsorted(splits["train"], batch_indices)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    reference_model = RingRepresentationClassifier("main")
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in reference_model.state_dict().items()
    }
    results: list[dict[str, object]] = []
    plan: dict[str, object] = {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_hash,
        "sample_count_after_duplicate_exclusion": int(len(dataset["labels"])),
        "duplicate_rows_excluded": int(dataset["duplicate_rows_excluded"]),
        "train_sample_count": int(len(splits["train"])),
        "validation_sample_count": int(len(splits["val"])),
        "train_batch_size": int(len(batch_indices)),
        "train_batch_sample_index_sha256": hashlib.sha256(
            np.asarray(dataset["sample_index"][batch_indices], dtype="<i8").tobytes()
        ).hexdigest(),
        "model_kind": "main",
        "parameter_count": int(sum(value.numel() for value in state_dict.values())),
        "cpu_count": int(os.cpu_count() or 0),
        "torch_version": torch.__version__,
        "torch_default_threads_before_benchmark": int(default_threads),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "warmup_training_updates_per_configuration": 2,
        "metrics_computed": False,
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = output_dir / "benchmark.json"

    configurations = [(torch.device("cpu"), threads) for threads in (1, 2, 4)]
    if torch.backends.mps.is_available():
        configurations.append((torch.device("mps"), 1))
    for device, threads in configurations:
        if device.type == "mps":
            torch.set_num_threads(1)
        record = _benchmark_device(
            device=device,
            cpu_threads=threads,
            state_dict=state_dict,
            train_inputs=torch.from_numpy(normalized["train"][train_offsets]),
            train_labels=torch.from_numpy(dataset["labels"][batch_indices].astype(np.int64)),
            train_distances=dataset["distance_mm"][batch_indices],
            train_blocks=dataset["block"][batch_indices],
            val_inputs=torch.from_numpy(normalized["val"]),
            training_repeats=args.training_repeats,
            validation_repeats=args.validation_repeats,
        )
        results.append(record)
        with config_path.open("w", encoding="utf-8") as output:
            json.dump(plan, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return plan


def main(argv: Sequence[str] | None = None) -> None:
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
