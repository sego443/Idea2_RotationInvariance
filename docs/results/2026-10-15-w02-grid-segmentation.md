# W02 grid segmentation and rejected-take deletion

Written 2026-10-15. Records the change of the W02 segmenter from peak counting to the prompt timing grid, its check on Part A, and the deletion of the rejected W02 takes.

**Status:** no W02 stroke has been scored by any model, and no W02 performance is claimed. Only stroke timing and label agreement were inspected; no class-wise signal was plotted or classified.

## 1. Segmentation change

W02 recordings are paced by the Sequence Collection widget (`bpm=20`, `prepare_seconds=3`, `finish_seconds=3`). `scripts/data/prepare_w02_session.py` now segments on that grid:

- The grid is anchored at the first sample of the file: 3 s of preparation, then one 3 s slot per prompt letter.
- Stroke k takes the label of the k-th prompt letter.
- Its 2 s window (500 samples) is centred on the maximum of the C170 localizer energy (0.5–12 Hz band-pass, MAD scaling, 75-sample envelope) inside slot k.
- Every slot gives a window. A recording is flagged only when it ends before its last slot.
- Per stroke, `slot_offset_s` and `slot_has_peak` (a C170 localizer peak, distance 525, prominence 1.5, lies in the slot) are written as diagnostics. They never drop a stroke.

The previous version required exactly as many localizer peaks as prompt letters and assigned labels by peak order. A missed stroke plus a spurious peak would have passed its count check with shifted labels.

Timing check on the 36 Part A trial files: 16,481–16,493 rows (66 s expected = 16,500), host-timestamp span 65.85–66.05 s, one sample-index discontinuity per file, no markers.

## 2. Part A comparison

Command (output not in Git):

```bash
python -m scripts.data.prepare_w02_session --session-dir <each of the six P01_AD1 session folders> \
  --plan data/w02_plans/P01_AD1_timed/w02_P01_AD1_plan_completed.csv --output-dir outputs/w02_partA_grid_20261015
```

| | Peak counting (`outputs/w02_partA_20261014_r2`) | Grid (`outputs/w02_partA_grid_20261015`) |
|---|---:|---:|
| Strokes (prefix / body) | 700 (280 / 420) | 720 (288 / 432) |
| Strokes per class | 175 | 180 |
| Flagged recordings | 1 (W02 T1, 19 peaks) | 0 |
| Dataset SHA-256 | `c8a8c5e8…dcc8` | `7499647551c603b5a3f56d0748d87f892d2061e54aa31fac3aafa96248bc5051` |

- The 700 strokes common to both have identical labels. 699 have identical centres. W06 T2 stroke 20 moved from 1.58 s to 0.99 s into its slot; both lie in the correct slot.
- W02 T1 (`OpenBCI-RAW-2026-10-13_20-45-49_P01_AD1_W02_M160_T1.txt`) now contributes 20 strokes. Slot 9, its first body stroke, has no localizer peak; it is the only slot in Part A without one.
- Window centres lie 0.50–2.06 s into their slots (1st/50th/99th percentile 0.68 / 1.18 / 1.79 s). None is within 0.1 s of a slot edge.
- Manifest and per-recording table: [`docs/evidence/w02_partA_grid_20261015/`](../evidence/w02_partA_grid_20261015/).

Checks: `tests/test_prepare_w02_session.py` passes, including a silent-slot case and a truncated-recording case. The synthetic rehearsal `outputs/w02_rehearsal_20261015` (1 epoch) ran plan generation, segmentation (480 strokes, 0 flagged), both launchers, the freeze manifests and both evaluators to completion. Its scores are meaningless by design.

## 3. Rejected takes deleted

On the user's instruction, all 72 `.rejected` files under `data/raw/w02_data/` were deleted on 2026-10-15: 49 from Part A (`P01_AD1`) and 23 from Part B (`P01_BD1`), 116 MB in total. Rejected takes are deleted without an inventory, hashes or reasons (project rule from 2026-10-15). The Part A completed plan still lists the Part A rejected filenames in its retry chains as historical rows.

## 4. Deviations for the one-shot evaluations

1. The frozen S2 models were fine-tuned on the peak-counting Part A dataset (`c8a8c5e8…`). Part B, and the Part A wears scored by S1, will be segmented with the grid method. For Part A the two methods differ in 20 W02 T1 strokes and one moved centre (§2).
2. The segmenter changed after both freezes and after Part B was recorded, before any W02 stroke was scored. The freeze manifests do not hash the segmenter.

## 5. Part B and the combined segmentation

Part B (wears W07–W12) was recorded on 2026-10-14, 16:53–23:20, in `data/raw/w02_data/P01_BD1/` (three session folders; not in Git).

Both plans were refilled with accepted takes only by `scripts/data/complete_w02_plan.py`, into `data/w02_plans/P01_<part>D1_timed/w02_P01_<part>D1_plan_accepted.csv`:

- Part A: the 42 entered files equal the accepted files of the 2026-10-14 completion. The 2026-10-14 file `w02_P01_AD1_plan_completed.csv` is kept because the frozen S2 dataset was built from it.
- Part B: W07 REST, T1 and T2 each had two non-rejected takes, 16:53–16:59 and 17:03–17:05. On the operator's choice, the later set was entered (`--choose`), making W07 one continuous REST→T6 run, 17:03–17:11. The three earlier takes and eight unlabeled files in Part B are not entered and are kept unchanged.
- `prepare_w02_session.py` now opens the plan's `actual_filename` when one is given, so extra takes with the same label are never read.

Combined segmentation (output not in Git; manifest and per-recording table in [`docs/evidence/w02_day1_grid_20261015/`](../evidence/w02_day1_grid_20261015/)):

```bash
python -m scripts.data.prepare_w02_session --session-dir <six P01_AD1 folders> --session-dir <three P01_BD1 folders> \
  --plan data/w02_plans/P01_AD1_timed/w02_P01_AD1_plan_accepted.csv \
  --plan data/w02_plans/P01_BD1_timed/w02_P01_BD1_plan_accepted.csv --output-dir outputs/w02_day1_grid_20261015
```

- 1,440 strokes (per part 288 prefix + 432 body), 360 per class, 85 REST windows, 0 flagged recordings. Dataset SHA-256 `9bf6f77d7add5d388e792ad88aef024643b825a4e669bec35e5282dcc17ac30e`.
- Part A strokes (files, positions, labels, centres, diagnostics) equal `outputs/w02_partA_grid_20261015` up to row order.
- Part B: 42 recordings, channel map `0,1,2,3` in all; trial files 16,475–16,496 rows; REST files 4,987–5,003 rows.
- Slots without a localizer peak: Part A 1 (W02 T1 position 9); Part B 3 (W08 T4 positions 12 and 17, W09 T4 position 12), all body strokes. Window centres lie 0.19–2.49 s into their slots in Part B (1st/50th/99th percentile 0.66 / 1.36 / 2.00 s); none within 0.1 s of a slot edge.

## 6. Remaining steps (not executed)

1. Run the two one-shot evaluations on `outputs/w02_day1_grid_20261015/dataset.npz` (design §6), once each, with these deviations and those of the 2026-10-14 record attached.
