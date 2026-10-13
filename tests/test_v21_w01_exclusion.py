import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import make_v21_freeze_manifest as freeze
from scripts.runners import run_v21_wear_calibrated as runner
from src import v21_data


def _fake(source: str, n: int) -> dict:
    return {"X": np.zeros((n, 500, 4), np.float32), "y": np.arange(n) % 4, "group": np.array([f"{source}_G"] * n),
            "position": np.zeros(n, int), "block": np.zeros(n, int), "repetition": np.ones(n, int),
            "sample_index": np.arange(n), "source": np.array([source] * n), "is_calibration": np.zeros(n, bool),
            "part": np.array(["A"] * n), "wear_group": np.array([f"{source}_G"] * n)}


def test_c170_protocols_never_load_w01(monkeypatch, tmp_path):
    monkeypatch.setattr(v21_data, "load_c170", lambda: _fake("C170", 8))
    monkeypatch.setattr(v21_data, "load_w02", lambda path: _fake("W02", 4))
    monkeypatch.setattr(v21_data, "load_w01", lambda: pytest.fail("W01 must not be loaded"))
    train, evaluations = runner.split_protocol("c170_all")
    assert set(train["source"]) == {"C170"} and evaluations == {}
    train, evaluations = runner.split_protocol("c170_plus_w02", tmp_path / "w02.npz", ("A",))
    assert set(train["source"]) == {"C170", "W02"} and evaluations == {}


@pytest.mark.parametrize("protocol", ["all_current", "all_current_plus_w02", "w01_half", "fixed_split"])
def test_freeze_refuses_runs_trained_with_w01_or_partial_data(tmp_path: Path, protocol: str):
    run = tmp_path / "run"
    run.mkdir()
    (run / "RUN_COMPLETE.json").write_text("{}")
    (run / "checkpoint.pt").write_bytes(b"x")
    (run / "run_manifest.json").write_text(json.dumps(
        {"arguments": {"protocol": protocol}, "source_sha256": {}, "data_sha256": {}}))
    with pytest.raises(ValueError):
        freeze.checkpoint_entry(run)
