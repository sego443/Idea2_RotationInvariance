# v21 development on existing data (exploratory)

Date: 2026-10-12. Machine: Apple M2 CPU, Python 3.14.6, PyTorch 2.14.1 (local `.tmp/venv`, not the historical CUDA environment). Data: C170 SHA-256 `d57a7fef…e349`, W01 segmented SHA-256 `29246cf4…9179` (matching the saved evidence).

All results are exploratory:
- The C170 registered-test positions were already exposed by F2 on 2026-09-29.
- C170 old validation has been reused many times.
- W01 was used in earlier W01-only studies. It had never scored a C170-trained model, but it still cannot serve as an untouched holdout.

Nothing here is a confirmatory result.

## Runs

| Root (`outputs/`) | Content | Status |
|---|---|---|
| `v21_dev_grid_20261012` | First grid | **Superseded**: launched before a calibration fix and before start-time source hashing; stopped |
| `v21_dev_grid_20261012_r2` + `_recovery1` | Grid 1 (24 runs) | 15 completed + 4 interrupted at the background time limit (preserved) + 9 rerun in the recovery root |
| `v21_crossrec_grid_20261012` | Grid 2 (12 runs) | Complete |
| `v21_stage2_analog_20261012` | Stage-2 analog (12 runs) | Complete |
| `v21_stage1_candidates_20261012` | Stage-1 freeze runs (10 real + 5 null, all data, no evaluation) | Complete; frozen in `docs/evidence/v21_freeze/v21_freeze_S1.json` |
| `c170_calibration_pilot_20261012` | Linear pilot | See the [pilot report](2026-10-12-c170-calibration-pilot.md) |

Each run records arguments, start-time source hashes, data hashes, a learning curve, a checkpoint, predictions and a completion marker. Summaries were produced with `scripts/analysis/summarize_v21_dev_grid.py`.

Common settings:
- 120 fixed epochs; augmentation on.
- CAL uses the first 2 repetitions of each direction in the scored group, which share that direction's recording. Scored strokes are the remaining repetitions.
- Values are means over 3 seeds; "vote" is the 3-seed majority-vote ensemble.

## Grid 1: calibration and cross-session transfer

Fixed split: train on F2's 11 training positions. The held-out positions were scored as two sets, old validation (val) and the exposed registered test (test).

| Config | ZERO val/test | CAL τ=1 val/test | UNSUP τ=1 val/test |
|---|---|---|---|
| B1, no episodes | 0.620 / 0.709 | 0.732 / 0.821 | 0.739 / 0.802 |
| B1, episodes | 0.654 / 0.745 | 0.809 / 0.830 | 0.816 / 0.827 |
| B2, no episodes | 0.471 / 0.657 | 0.764 / 0.859 | 0.531 / 0.723 |
| B2, episodes | 0.448 / 0.659 | 0.813 / 0.859 (vote 0.881 / 0.901) | 0.551 / 0.767 |

W01 transfer: train on all C170 and score all 12 W01 angles. This is a different session, day and angle frame.

| Config | ZERO | CAL τ=1 (vote) | UNSUP τ=1 |
|---|---|---|---|
| B1, no episodes | 0.229 | 0.330 (0.333) | 0.179 |
| B1, episodes | 0.230 | 0.840 (0.853) | 0.254 |
| B2, no episodes | 0.221 | 0.791 (0.869) | 0.221 |
| B2, episodes | 0.273 | 0.947 (0.986) | 0.274 |

## Grid 2: cross-recording calibration proxy

Fixed split, scored as one held-out set of 8 positions, with episodic weight 1.

CALNB calibrates each held-out position with the first 2 repetitions of another held-out position 10–20 mm away. That uses different recordings, made minutes apart, after the band was slid. It is a pessimistic proxy for W02's same-wear calibration across recordings.

| Config (episode support) | ZERO | CAL τ=1 | CALNB best τ (value) |
|---|---|---|---|
| B1, same | 0.704 | 0.816 | τ=16 (0.712) |
| B1, neighbour | 0.685 | 0.797 | τ=16 (0.694) |
| B2, same | 0.515 | 0.852 (vote 0.887) | τ=4 (0.541) |
| B2, neighbour | 0.551 | 0.786 | τ=1 (0.566) |

## Stage-2 analog: adding same-session data

Train on all of C170 + W01 angles 0/60/…/300, then score W01 angles 30/90/…/330 (3 seeds; same-recording episodes). The "no update" row is the C170-only W01-transfer models from grid 1, restricted to the same angles.

| Training | ZERO B2 / B1 | CAL τ=1 B2 / B1 | Runtime per run (4 parallel) B2 / B1 |
|---|---|---|---|
| No update (C170 only) | 0.267 / 0.267 | 0.977 / 0.834 | — |
| Fine-tune 30 epochs, lr 3e-4 | 0.352 / 0.277 | 0.991 / 0.882 | 7.7 / 2.1 min |
| From scratch 120 epochs | 0.413 / 0.379 | 0.986 / 0.925 | 31.4 / 7.7 min |

Same-session data from other angles lifts zero-calibration F1 only to about 0.4. Fine-tuning keeps in-recording calibration at its ceiling, so it is the fixed S2 recipe.

The Stage-1 runs took about 33 min each for B2 and 10 min for B1 (4 parallel). MPS is not usable, because adaptive pooling with non-divisible sizes is unsupported.

## Linear diagnostics (`.tmp/` scratch scripts, not maintained)

- **W01 vs C170:** the per-class amplitude ordering differs (C170: UP/DOWN loudest; W01: RIGHT/DOWN loudest). No channel roll or order reversal maps a C170 linear model onto W01; the best was 0.35.
- **Within W01**, leave-one-angle-out linear F1 is 0.36, or 0.44 with unsupervised per-angle channel normalization. Adding C170 data does not help (0.34).
- **Training-free nearest centroid** with 2 calibration strokes per direction from the same recording: W01 0.942, C170 0.832.
- **Neighbour-recording calibration, linear:** no calibration 0.551; same recording 0.850; neighbour recording 0.607 (nearest centroid 0.335).

## Interpretation

1. **Zero-calibration cross-session transfer is at chance** for every variant. This is the central obstacle for a pure "wear it any way" claim.
2. **Same-recording calibration is strong**, and episodic training makes it work across sessions (W01 0.95). In this data, however, every recording holds a single direction, so recording identity and direction are confounded. The training-free baseline (0.94 on W01) shows how much is recording-level template matching.
3. **Cross-recording calibration barely beats zero calibration** in the neighbour proxy, even with neighbour-support episodes.
4. **Consequence for W02.** Each trial recording carries its own mixed-direction calibration prefix. That keeps the strong within-recording consistency and removes the direction-recording confound. Cross-recording calibration (E1b), label-free adaptation (E2) and zero calibration (E3) remain secondary endpoints.
5. **Selected configurations:** B2 with same-recording episodes and τ=1 for in-recording calibration; B1 with same-recording episodes for the other modes.
