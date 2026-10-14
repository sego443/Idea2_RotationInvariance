import numpy as np

from src import w02_cov_fewshot as recipes


def synthetic(n_recordings: int, seed: int) -> dict:
    """Recordings of 8 prefix + 12 body strokes; each class drives a different channel pair with a 3 Hz burst."""
    rng = np.random.default_rng(seed)
    t = np.arange(500) / 250
    burst = np.sin(2 * np.pi * 3 * t) * np.exp(-((t - 1) ** 2) / 0.05)
    X, y, group, calib = [], [], [], []
    for r in range(n_recordings):
        labels = np.concatenate([rng.permutation(np.repeat(np.arange(4), 2)), rng.permutation(np.repeat(np.arange(4), 3))])
        for k, label in enumerate(labels):
            window = rng.normal(0, 0.2, (500, 4))
            window[:, label] += burst
            window[:, (label + 1) % 4] -= 0.5 * burst
            X.append(window)
            y.append(label)
            group.append(f"W01_T{r + 1}")
            calib.append(k < 8)
    n = len(y)
    return {"X": np.array(X, dtype=np.float32), "y": np.array(y), "group": np.array(group),
            "wear_group": np.array(["W01"] * n), "is_calibration": np.array(calib), "position": np.zeros(n)}


def test_calibrated_recipes_separate_classes_and_null_does_not() -> None:
    data = synthetic(3, 0)
    body = ~data["is_calibration"]
    for recipe in ("rec_nc", "wear_logreg"):
        predicted = recipes.predict(recipe, data)
        assert np.all(predicted[data["is_calibration"]] == -1)
        assert np.mean(predicted[body] == data["y"][body]) > 0.9
    null = recipes.predict("rec_nc", data, permute_seed=1)
    assert np.mean(null[body] == data["y"][body]) < 0.9


def test_cyclic_canonical_rounds_half_up() -> None:
    windows = np.arange(4, dtype=float)[None, None, :].repeat(3, axis=0)
    out = recipes.cyclic_canonical(windows, np.array([0.0, 90.0, 225.0]))
    np.testing.assert_array_equal(out[:, 0, :], [[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]])
