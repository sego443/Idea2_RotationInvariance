# W02 Part A and v21 Stage-1/Stage-2 freezes

Written 2026-10-14. Records what was collected for W02 Part A, how the plan was completed and segmented, how the Stage-1 and Stage-2 models were frozen, and every deviation from [protocol version 3](../data_collection/w02_rewear_protocol.md) and the [v21 design](../designs/2026-10-12-v21-wear-calibrated-classifier.md).

**Status:** everything before Part B collection is done. **No W02 stroke has been scored by any model, and no W02 performance is claimed.** Both evaluations (S1 on all 12 wears, S2 on Part B) remain one-shot and untouched.

## 1. Collection

- Participant P01, Part A, wears W01–W06, plan `data/w02_plans/P01_AD1_timed/` (the `_timed` variant loaded into the Sequence Collection widget).
- Recorded 2026-10-13, 17:58 (W01 session start) to 22:54 (last W06 recording). Raw files: `data/raw/w02_data/P01_AD1/OpenBCISession_2026-10-13_*` (six session folders, one per wear; not in Git).

| Wear | Tape reading at S3/S0 midpoint (mm) | Band angle (°) | REST → T6 |
|---|---:|---:|---|
| W01 | 80 | 193 | 18:01 → 18:12 |
| W02 | 160 | 3 | 20:45 → 20:52 |
| W03 | 95 | 225 | 21:05 → 21:13 |
| W04 | 125 | 289 | 21:26 → 21:41 |
| W05 | 5 | 34 | 21:52 → 22:15 |
| W06 | 30 | 87 | 22:25 → 22:53 |

Angles use the protocol §3 formula as implemented by `make_w02_session_plan.band_angle_deg`.

## 2. Completed plan

`data/w02_plans/P01_AD1_timed/w02_P01_AD1_plan_completed.csv` was derived from the files on disk by [`complete_plan_a.py`](../evidence/w02_partA_20261014/complete_plan_a.py). The generated plan file is unchanged.

- A file whose name carries a plan `filename_label` and does not end in `.rejected` is the accepted take (operator statement). There is exactly one such take per planned recording: 42 accepted (6 REST + 36 trial).
- The 49 `.rejected` files are entered as protocol §6 retry chains. The planned row holds the first take and each later take is appended with `retry_of` pointing to the previous take. In every chain the rejected takes precede the accepted take.
- `actual_tape_mm` is the M field of the filename. The operator stated that M is the actual reading, so actual equals scheduled for every wear.
- Not recorded: rejection reasons, strap hole, and the wrist/band circumference measurements (protocol §3.2).
- Seven files have no filename label and are not entered in the plan. They are kept unchanged:

| Session folder | File | Size (bytes) |
|---|---|---:|
| `OpenBCISession_2026-10-13_21-25-07` (W04) | `OpenBCI-RAW-2026-10-13_21-30-19.txt` | 11,066,169 |
| `OpenBCISession_2026-10-13_21-51-15` (W05) | `OpenBCI-RAW-2026-10-13_21-52-21.txt` | 465,475 |
| | `OpenBCI-RAW-2026-10-13_22-01-25.txt` | 13,029,818 |
| | `OpenBCI-RAW-2026-10-13_22-07-42.txt` | 523,836 |
| | `OpenBCI-RAW-2026-10-13_22-08-56.txt` | 2,270,312 |
| | `OpenBCI-RAW-2026-10-13_22-09-59.txt` | 1,724,746 |
| `OpenBCISession_2026-10-13_22-23-55` (W06) | `OpenBCI-RAW-2026-10-13_22-24-56.txt` | 2,598,643 |

## 3. Segmentation

```bash
python -m scripts.data.prepare_w02_session \
  --session-dir data/raw/w02_data/P01_AD1/OpenBCISession_2026-10-13_17-58-39 ... (all six folders) \
  --plan data/w02_plans/P01_AD1_timed/w02_P01_AD1_plan_completed.csv --output-dir outputs/w02_partA_20261014_r2
```

- 35 of 36 trial recordings gave exactly 20 strokes. One recording was flagged and not repaired: W02 T1 (`OpenBCI-RAW-2026-10-13_20-45-49_P01_AD1_W02_M160_T1.txt`), 19 detected. It contributes no strokes.
- 700 strokes (280 prefix, 420 body), 175 per class; 42 REST windows; channel map `0,1,2,3` in all 42 recordings.
- Dataset SHA-256 `c8a8c5e8f6107d6a69f9ba3aa967350ac697a451051f77a3529eaa227ed7dcc8`. Manifest, per-recording table and flag list: [`docs/evidence/w02_partA_20261014/`](../evidence/w02_partA_20261014/).
- Only stroke counts and integrity were inspected; no class-wise signal was plotted or classified.
- `outputs/w02_partA_20261014` is an earlier segmentation that used a first plan completion with the retry rows in the wrong direction. It produced the same 700 windows in another row order, was never used for training, and is kept with a `SUPERSEDED.txt` note.

## 4. Stage-1 and Stage-2 freezes

Both stages used the fixed recipe in design §5 without changes, on CPU (Apple M2, 4 workers × 2 threads).

| | S1 | S2 |
|---|---|---|
| Command | `launch_v21_stage1` (design §6) | `launch_v21_stage2 --w02-dataset outputs/w02_partA_20261014_r2/dataset.npz` |
| Output root (local, not in Git) | `outputs/v21_stage1_c170_20261014` | `outputs/v21_stage2_20261014` |
| Training data | C170, 1,599 windows, 20 position groups (protocol `c170_all`) | C170 + Part A, 2,299 windows, 55 groups (35 W02 recordings) (protocol `c170_plus_w02`) |
| Runs | B1 and B2 × seeds 17/29/43/61/89, 120 epochs; five B2 matched nulls | Every S1 run fine-tuned 30 epochs at lr 3e-4; nulls keep their null seed |
| Mean run time (B1 / B2 / B2 null) | 489 / 1,730 / 1,456 s | 192 / 634 / 527 s |
| Manifest | `docs/evidence/v21_freeze/v21_freeze_S1_c170.json` | `docs/evidence/v21_freeze/v21_freeze_S2.json` |
| Manifest SHA-256 | `d9c69214…2ebd` | `cf93ed88…bfd` |
| Commit | `9f7461f` | `00e17e7` (before any Part B recording) |
| Scored on | Body strokes of Parts A and B | Body strokes of Part B |

Checks made after the freezes:

- All 15 runs of each stage wrote `RUN_COMPLETE.json`. Their checkpoints match the manifest hashes.
- Every S1 run used protocol `c170_all` and every S2 run `c170_plus_w02`. The C170 data hash is `d57a7fef…9e349` and the S2 W02 data hash equals the Part A dataset above.
- All S1 and S2 runs share the S1 shared scale (11444.94). S2 kept it as specified.
- The S1 run source hashes (`run_v21_wear_calibrated.py`, `src/v21_wear_calibrated.py`, `src/v21_data.py`) equal the files committed in `ff1f682`.
- Endpoints, τ values, gate, bootstrap and ensemble rule are identical in S1 and S2. The scorer hash `2094004a…fa275` equals the current `scripts/analysis/evaluate_v21_frozen_w02.py`.
- No training metrics on W02 were computed or inspected.

Pipeline rehearsal (protocol §3.5), synthetic data only, 1 epoch:

- `outputs/w02_rehearsal_20261014` failed at segmentation. `rehearse_w02_pipeline.py` still named its synthetic files by `prompt`, while `prepare_w02_session.py` has looked files up by `filename_label` since the 2026-10-13 plan change. The fix renames the synthetic files and changes no frozen code. The scorer, the runner and the launchers are untouched.
- `outputs/w02_rehearsal_20261014_r2` ran plan generation, segmentation, S1 and S2 launchers, the freeze manifests and both one-shot evaluators to completion. Its scores are meaningless by design.
- The 23 v21/W02 unit tests pass.

## 5. Deviations

1. **S1 frozen after Part A was collected.** The design required the C170-only S1 manifest to be committed before the acquisition day. It was trained on 2026-10-14, after Part A was recorded and segmented. S1 uses C170 data only and no W02 data. Before the freeze, no W02 stroke had been classified, plotted class-wise or scored. The code S1 ran was last modified on 2026-10-12, before Part A started (file modification times), and was committed in `ff1f682` before S1 training started. S1's evaluation on Part A wears is therefore blind to scores but not pre-registered in time; report this alongside every S1 result.
2. **Not one day.** Part A was collected in the evening of 2026-10-13, and Part B will be collected on a later date. The plan files keep the `D1` day label. S2's test wears (Part B) come from a different day than its W02 training data (Part A).
3. **Missing acquisition records:** rejection reasons, strap hole, circumference measurements and the per-part reference photo were not recorded in the plan (the photo may exist outside the repository).
4. **Seven unlabeled recordings** (§2) are not part of the plan.
5. **Environment:** CPU training on the Apple M2 used `.tmp/venv` (Python 3.14.6, torch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1), the same versions recorded in the 2026-10-12 v21 development runs.
6. **Rehearsal script fix:** `scripts/validation/rehearse_w02_pipeline.py` was corrected after the freezes (§4). It is not part of any frozen model or of the scorer.

## 6. Remaining steps (Part B, not executed)

1. Collect Part B, wears W07–W12, from `data/w02_plans/P01_BD1_timed/` (protocol §5–§6). Record rejection reasons, strap hole and the reference photo this time. Keep the `D1` labels printed in the plan; the actual date is recorded here.
2. Complete the Part B plan the same way and segment Parts A and B together into a new directory (design §6).
3. Run the two one-shot evaluations, `v21_freeze_S1_c170.json` on all 12 wears and `v21_freeze_S2.json` on Part B. Run each exactly once. Report both with deviation 1 attached to S1.
