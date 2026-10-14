"""Score the src/w02_cov_fewshot.py recipes on W02 wears.

--mode dev     leave-one-wear-out on the development dataset (recipes that need training wears use the other
               development wears). Exploratory; may be rerun.
--mode sealed  the sealed dataset, with all development wears as training wears. Run once: the output directory
               must not exist and EVALUATION_STARTED.json is created exclusively before any sealed stroke is read.
Each recipe reports macro-F1, accuracy, per-class recall, prediction counts, per-wear macro-F1, confusion matrix and
a 95% wear-cluster bootstrap interval (10,000 resamples, seed 20261015). Calibrated recipes also report a matched
null (support labels permuted within each support set, seeds 0-4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from src import v21_data, w02_cov_fewshot as recipes

SEALED_WEARS = {"W02_P01_B_W07", "W02_P01_B_W10", "W02_P01_B_W12"}
CALIBRATED = ("rec_nc", "rec1_borrowed", "wear_logreg")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", labels=list(range(4))))


def report(data, predicted) -> dict:
    scored = ~data["is_calibration"]
    y, p, wears = data["y"][scored], predicted[scored], data["wear_group"][scored]
    rng = np.random.default_rng(20261015)
    unique = np.unique(wears)
    members = {w: np.flatnonzero(wears == w) for w in unique}
    boot = [macro_f1(y[i], p[i]) for i in (np.concatenate([members[w] for w in rng.choice(unique, len(unique))])
                                           for _ in range(10_000))]
    return {"macro_f1": macro_f1(y, p), "accuracy": float(np.mean(y == p)),
            "recall": {name: float(np.mean(p[y == c] == c)) for c, name in enumerate(v21_data.CLASSES)},
            "prediction_counts": dict(zip(v21_data.CLASSES, np.bincount(p, minlength=4).tolist())),
            "per_wear_f1": {w: macro_f1(y[wears == w], p[wears == w]) for w in unique},
            "confusion_true_rows": confusion_matrix(y, p, labels=list(range(4))).tolist(),
            "bootstrap_95ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "n_scored": int(scored.sum())}


def score(recipe, test, train_for):
    """train_for(wear) -> labeled training data for that test wear, or None."""
    predicted = np.full(len(test["y"]), -1)
    nulls = [np.full(len(test["y"]), -1) for _ in range(5)] if recipe in CALIBRATED else []
    for wear in np.unique(test["wear_group"]):
        mask = test["wear_group"] == wear
        part = v21_data.subset(test, mask)
        train = train_for(wear)
        predicted[mask] = recipes.predict(recipe, part, train)
        for seed, null in enumerate(nulls):
            null[mask] = recipes.predict(recipe, part, train, permute_seed=seed)
    return predicted, nulls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dev", "sealed"), required=True)
    parser.add_argument("--dev-dataset", type=Path, default=Path("outputs/w02_dev_20261015/dataset.npz"))
    parser.add_argument("--sealed-dataset", type=Path, default=Path("outputs/w02_sealed_20261015/dataset.npz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dev = v21_data.load_w02(args.dev_dataset)
    if set(np.unique(dev["wear_group"])) & SEALED_WEARS:
        raise RuntimeError("sealed wear present in the development dataset")
    started = {"started_utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
               "dev_dataset_sha256": sha256(args.dev_dataset),
               "recipe_source_sha256": sha256(Path(recipes.__file__)), "scorer_sha256": sha256(Path(__file__))}
    if args.mode == "sealed":
        started["sealed_dataset_sha256"] = sha256(args.sealed_dataset)
        with (args.output_dir / "EVALUATION_STARTED.json").open("x") as file:
            json.dump(started, file, indent=2)
        test = v21_data.load_w02(args.sealed_dataset)
        if set(np.unique(test["wear_group"])) != SEALED_WEARS:
            raise RuntimeError("sealed dataset does not hold exactly the sealed wears")
        train_for = lambda wear: dev  # noqa: E731
    else:
        (args.output_dir / "run.json").write_text(json.dumps(started, indent=2))
        test = dev
        train_for = lambda wear: v21_data.subset(dev, dev["wear_group"] != wear)  # noqa: E731
    results = {}
    for recipe in recipes.RECIPES:
        predicted, nulls = score(recipe, test, train_for)
        results[recipe] = report(test, predicted)
        results[recipe]["null_macro_f1"] = [macro_f1(test["y"][~test["is_calibration"]], n[~test["is_calibration"]])
                                            for n in nulls]
        np.save(args.output_dir / f"predictions_{recipe}.npy", predicted)
        r = results[recipe]
        print(f"{recipe:14s} F1 {r['macro_f1']:.3f} CI {[round(v, 3) for v in r['bootstrap_95ci']]} "
              f"min recall {min(r['recall'].values()):.2f} pred {list(r['prediction_counts'].values())} "
              f"null {[round(v, 3) for v in r['null_macro_f1']]}", flush=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
