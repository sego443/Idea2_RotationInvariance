# W02 sealed holdout and switch to exploratory development

Written 2026-10-15, before any W02 stroke was scored by any model.

## 1. Decision

The two one-shot evaluations of the v21 design (S1 on all 12 wears, S2 on Part B) are retired. The operator expects the frozen models to perform poorly and wants to develop on W02 data. Three Part B wears are sealed as an untouched holdout instead (option 2 of the 2026-10-15 discussion):

- Everything that follows on the development wears is **exploratory**: model and preprocessing choices may be tuned on them freely.
- The sealed wears are opened once, for one frozen recipe, after development ends. Only that score is a confirmatory result. It covers rewearing on a day that also supplied development data (Part B was recorded on 2026-10-14; Part A on 2026-10-13).

## 2. Sealed wears

`scripts/data/split_w02_sealed.py` drew one Part B wear from each 120° sector of the measured band angle with `numpy.random.default_rng(20261015)`:

| Sealed wear | Angle | Tape (mm) |
|---|---:|---:|
| W07 | 193° | 80 |
| W10 | 289° | 125 |
| W12 | 87° | 30 |

Every stroke and REST window of these wears is in `outputs/w02_sealed_20261015/dataset.npz` only (360 strokes, 216 body; 22 REST windows; SHA-256 `cb8e397b…a227`; directory made read-only). The development dataset `outputs/w02_dev_20261015/dataset.npz` holds W01–W06, W08, W09 and W11 (1,080 strokes; 63 REST windows; SHA-256 `f5ec8530…a764`). Both derive from `outputs/w02_day1_grid_20261015` (`9bf6f77d…c30e`). Split manifest: [`docs/evidence/w02_seal_20261015/split_manifest.json`](../evidence/w02_seal_20261015/split_manifest.json).

Every angle of a sealed wear also occurs in Part A (same tape readings), so the sealed test measures rewearing at a seen angle, not generalization to an unseen angle.

## 3. Frozen baseline on the development wears

The frozen S1 and S2 manifests are scored once each with the unchanged scorer, on the development dataset only (S1: Parts A and B → nine wears; S2: Part B → W08, W09, W11). These are exploratory baselines for the development work, not the retired one-shot tests. Results: [exploratory development record](2026-10-15-w02-exploratory-development.md) §1.
