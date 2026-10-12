import numpy as np
import pytest

from scripts.runners import run_v21_wear_calibrated as base
from src import v22_window_norm as wn
from src.c170_shared_scale_preprocessing_v3 import calibrate_windows_shared_scale_v3


def _windows(n=6, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(500)
    bursts = np.exp(-((t - 215) / 25.0) ** 2) + 0.8 * np.exp(-((t - 300) / 25.0) ** 2)
    carrier = np.sin(2 * np.pi * 8 * t / 250)
    gains = rng.uniform(0.5, 3.0, size=(n, 1, 4))
    return (bursts * carrier)[None, :, None] * gains + 0.05 * rng.standard_normal((n, 500, 4)) + 7.0


def test_full_shared_matches_v21_preprocessing():
    raw = _windows()
    expected, scale = calibrate_windows_shared_scale_v3(base.bandpass(raw))
    values, mask, _ = wn.apply_window(raw, "full")
    ours = wn.normalize(values, mask, "shared", np.zeros(len(raw)), wn.fit_shared_scale(values, mask))
    assert mask.all()
    assert wn.fit_shared_scale(values, mask) == pytest.approx(scale, rel=1e-5)
    np.testing.assert_allclose(ours, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("variant", ["prefix_200", "prefix_250", "prefix_300", "outbound_valley"])
def test_prefix_variants_ignore_later_samples(variant):
    raw = _windows()
    values, mask, _ = wn.apply_window(raw, variant)
    stop = mask.sum(axis=1)
    changed = raw.copy()
    for index, cut in enumerate(stop):
        changed[index, cut:] += 50.0 * np.random.default_rng(index).standard_normal((500 - cut, 4))
    if variant == "outbound_valley":
        return  # the oracle cut itself depends on the whole window; checked separately below
    np.testing.assert_array_equal(wn.apply_window(changed, variant)[0], values)
    assert not values[:, stop[0]:].any()


def test_outbound_valley_cuts_between_the_two_bursts():
    _, mask, stats = wn.apply_window(_windows(), "outbound_valley")
    cuts = mask.sum(axis=1)
    assert stats["two_bursts_found_fraction"] == 1.0
    assert np.all((cuts > 215) & (cuts < 300))


def test_return_only_hides_the_first_half():
    values, mask, _ = wn.apply_window(_windows(), "return_only")
    assert not mask[:, :250].any() and mask[:, 250:].all()
    assert not values[:, :250].any()


def test_normalizations_remove_the_intended_gains():
    raw = _windows(n=4)
    values, mask, _ = wn.apply_window(raw, "prefix_300")
    scaled = values * np.array([2.0, 0.5, 3.0, 1.5], dtype=np.float32)
    groups = np.array(["a", "a", "b", "b"])
    for kind in ("window_global", "window_sensor"):
        a = wn.normalize(values, mask, kind, groups)
        b = wn.normalize(values * 4.0, mask, kind, groups)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    # per-sensor gains shared by a whole group cancel under group_sensor but not under window_global
    np.testing.assert_allclose(wn.normalize(values, mask, "group_sensor", groups),
                               wn.normalize(scaled, mask, "group_sensor", groups), rtol=1e-5, atol=1e-6)
    assert not np.allclose(wn.normalize(values, mask, "window_global", groups),
                           wn.normalize(scaled, mask, "window_global", groups))
    # window_global keeps cross-channel ratios; window_sensor gives every visible channel unit RMS
    global_values = wn.normalize(values, mask, "window_global", groups)
    visible = np.abs(values) > 1e-3
    ratio = np.where(visible, global_values / np.where(visible, values, 1.0), np.nan)
    per_window = np.nanmedian(ratio.reshape(len(values), -1), axis=1)
    np.testing.assert_allclose(ratio, np.broadcast_to(per_window[:, None, None], ratio.shape)
                               * np.where(visible, 1.0, np.nan), rtol=1e-4)
    sensor = wn.normalize(values, mask, "window_sensor", groups)
    rms = np.sqrt((sensor**2).sum(axis=1) / mask.sum(axis=1, keepdims=True))
    np.testing.assert_allclose(rms, 1.0, rtol=1e-4)
    assert not sensor[~mask].any()


def test_shared_requires_scale_and_unknown_names_fail():
    values, mask, _ = wn.apply_window(_windows(n=2), "full")
    with pytest.raises(ValueError):
        wn.normalize(values, mask, "shared", np.zeros(2))
    with pytest.raises(ValueError):
        wn.apply_window(_windows(n=2), "suffix")
    with pytest.raises(ValueError):
        wn.normalize(values, mask, "per_class", np.zeros(2))
