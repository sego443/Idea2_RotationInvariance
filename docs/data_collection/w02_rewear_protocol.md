# W02 one-day rewear acquisition protocol

Version 3 (2026-10-12). Version 2 of the same day (20 wears × 3 trial recordings, pen strokes, degree tape under S0) and version 1 (separate CAL recording, three days) were replaced before any collection. Status: **proposed, not collected**.

Version 3 changes: (a) the movement is the C170/W01 in-air swipe with return to centre, not a pen stroke; (b) wear positions use the wrist tape already used for C170; (c) 12 wears with 6 trial recordings each, instead of 20 wears with 3, because doffing the band is slow while extra recordings within a wear are cheap. This document supersedes the outline in [EXPERIMENT_DESIGN.md §7](../EXPERIMENT_DESIGN.md#7-proposed-independent-follow-up-not-executed).

The hardware, firmware, gain and sample rate are unchanged from C170; only the protocol changes. Model and endpoints: [v21 design](../designs/2026-10-12-v21-wear-calibrated-classifier.md). Evidence behind each choice: [v21 development results](../results/2026-10-12-v21-development-on-existing-data.md).

## 1. Why the protocol changes

| C170/W01 property | Consequence | W02 fix |
|---|---|---|
| One recording per direction per position, always UP→DOWN→LEFT→RIGHT | Direction is confounded with recording and time. Models and calibration exploit recording-level similarity (2-stroke nearest-centroid inside a W01 recording: 0.94 F1) | Every recording interleaves all four directions in a random, balanced order |
| Positions/angles collected in increasing order | Angle is confounded with time | Wear angles are stratified over 360° and visited in random order |
| Band slid along the wrist | Few independent wear episodes | Every wear is a full doff/don |
| Calibration across recordings barely helps (neighbour-recording proxy: +0.00 to +0.05 F1) | A separate calibration recording is a weak lever | Every trial recording **starts with its own labeled 8-stroke calibration prefix** |
| No measured angle | Angle cannot be stratified | Wrist-tape reading per wear, converted to the S0 band angle (§3) |

The prefix design matches the intended use, "start recording → 8 calibration strokes → write → stop". Because the prefix and the scored strokes contain all four directions inside the same recording, recording-level similarity cannot stand in for direction.

## 2. Supported claims

The claim covers one participant on unchanged hardware: four-direction recognition of presegmented strokes **after a complete rewear at an arbitrary measured angle**, on a different day from all training data. The primary condition uses an 8-stroke in-recording calibration. Secondary results cover cross-recording calibration, unlabeled adaptation and no calibration. Every evaluated model is frozen before the wears it is scored on. Cross-person evidence requires §8. Continuous detection, handwriting and open vocabulary are out of scope.

## 3. Preparation (before the day)

1. Keep the C170 S0–S3 labels and EXG mapping. Sensors are equally spaced around the band. Record the active EXG columns.
2. Measure and record the wrist circumference at the band line and the band circumference at the chosen strap hole.
3. Position reference (the C170 tape, no hardware change). The band is worn on the **right wrist**, circumference 170 mm.
   - A wrist tape marked 0–17 cm runs around the wrist at the band line. The **1 cm mark lies on the dorsal midline** (middle of the back of the hand side). Readings increase from there toward the little finger, across the palm toward the thumb, and back to the dorsal side.
   - S0→S1→S2→S3 increase along the tape. The sensors are equally spaced, 170/4 = 42.5 mm apart.
   - The band position is the **tape reading under the midpoint between S3 and S0**. Write it in mm.
   - Software converts it to the **band angle**: the angle of S0 from the dorsal midline, in the tape direction. S0 sits 21.25 mm past the S3/S0 midpoint, so

     band angle = ((reading + 21.25 − 10) mod 170) × 360 / 170.

     Examples: 158.75 mm → 0° (S0 on the dorsal midline); 0 mm → 24°; 31.25 mm → 90°.
   - Viewed from the hand toward the elbow, increasing readings run counter-clockwise on the right wrist. The W01 task document describes its angles as increasing clockwise in that view; the two frames have not been reconciled, so W01 angles and W02 band angles are not compared or pooled by value.
4. Fix the setup: same wrist, chair, posture and forearm support as C170. Use one strap hole for all wears unless comfort requires otherwise.
5. **Train and commit the C170-only Stage-1 freeze manifest** (`docs/evidence/v21_freeze/v21_freeze_S1_c170.json`, design §6). W01 is not used for training; the older `v21_freeze_S1.json` (C170 + W01) is superseded. Run `python -m scripts.validation.rehearse_w02_pipeline --output-dir outputs/<new dir>` on the acquisition laptop to confirm that the whole pipeline runs.
6. On the morning of the day, generate both plans with new seeds:

   ```bash
   python -m scripts.data.make_w02_session_plan --part A --day 1 --n-wears 6 --seed <new seed> --output-dir data/w02_plans/P01_AD1
   python -m scripts.data.make_w02_session_plan --part B --day 1 --n-wears 6 --first-wear 7 --seed <new seed> --output-dir data/w02_plans/P01_BD1
   ```

   Load each `_prompts.json` into the OpenBCI GUI Sequence Collection widget and print the matching plan CSVs. The `filename_label` column supplies the recording label, such as `P01_AD1_W03_M085_T2` (M085 = 8.5 cm at the S3/S0 midpoint). The JSON `actions` array contains all 20 movements in order, with `bpm=20`, `prepare_seconds=3` and `finish_seconds=3`. Each displayed movement includes the instruction to return to centre. The first eight movements are the calibration prefix; the following twelve are scored. These roles are recorded in the W02 plan CSV. The CSV `prompt` column contains the W02 sequence identifier used by segmentation; file lookup uses `filename_label`.

   REST uses one `"Hold still in centre"` action at `bpm=3` with zero preparation and finishing time, giving 20 seconds. The generic prompt format is documented in [Sequence Collection](../../../OpenBCI_GUI/SEQUENCE_COLLECTION.md).

## 4. Day schedule and staged freezes

| Step | Content | Est. time |
|---|---|---|
| Part A | Wears 1–6 | ~75 min |
| Midday break | Segment Part A; run `launch_v21_stage2` (fine-tunes the frozen S1 models on C170 + Part A and writes the S2 manifest, ~35 min); commit it | ~60 min |
| Part B | Wears 7–12 | ~75 min |
| After Part B | Segment Part B; run the one-shot S1 evaluation (all 12 wears) and the S2 evaluation (Part B) | ~15 min |

Each wear has one REST recording and six trial recordings (T1–T6). Each trial recording has 20 strokes: an 8-stroke prefix (2 per direction) and a 12-stroke body (3 per direction), each in random order. That gives 72 scored strokes per wear and 864 for the day. One wear takes about 12 minutes (2 min doff/don, 1 min settling, 20 s REST, six recordings of 66 s with short pauses and manual review).

The wear is the unit of analysis, so 12 wears (6 tested by S2) give a much wider wear-level interval than 20 would; extra recordings narrow only the within-wear error. They also give E1b five other recordings of the same wear as calibration support. If time allows, add wears, not recordings: rerun the generator with a larger `--n-wears` before the day.

If Stage 2 cannot be frozen during the break, skip it rather than delaying Part B or improvising. Part B still tests Stage 1. Before the scheduled evaluations, open signals only for the integrity checks in §6. Segmenting Part A for training involves no class-wise inspection.

## 5. One wear

1. Remove the band completely, shake out the hand and wait 30 s.
2. Put the band on and turn it until the midpoint between S3 and S0 is at the scheduled tape reading. Tighten to the recorded hole. Write the actual reading (nearest 5 mm, in mm) into `actual_tape_mm` and the hole into the plan. Do not convert it to an angle by hand.
3. Sit in the fixed posture and wait 60 s.
4. **REST** (20 s): hold the hand still in the centre (start) posture; make no movements.
5. **T1–T6**: follow the prompt letters, prefix first, then body, without pausing between them. For each recording:
   - Start recording and stay still in the centre posture for 3 s.
   - With a 20-BPM cue (one beat every 3 s), make one movement per beat: the same **in-air swipe** as in C170 and W01 (U=up, D=down, L=left, R=right), immediately followed by the **return to centre** at a natural, steady speed, without a pause at the far end. Then hold still in the centre posture until the next beat.
   - Keep the swipe extent and speed as in C170. Do not switch to swipe-and-hold: the frozen models were trained on swipe-and-return, and the return carries class information (v22 ablation).
   - Stay still for 3 s after the last movement. The widget stops recording automatically. **Stop** interrupts it early.
   - After the file closes, choose **Accept** to proceed or **Reject / Redo** to repeat the same recording. Starting another recording or loading another prompt file requires this decision.
6. Do not adjust the band, touch the sensors or move the cables within a wear. Any adjustment ends the wear. Mark its remaining recordings rejected and redo the whole wear at the end of its part, with the same scheduled angle and a new wear number.

## 6. Integrity checks and retries

After each recording, check only integrity:

- the file is present;
- the duration is within ±30% of the plan;
- four active EXG channels, without flatline, disconnection or saturation;
- 20 strokes in the printed order (operator self-report).

Do not classify or plot class-wise signals.

The widget renames the rejected file to `.rejected`.

## 7. Data handling

Use session folders named `data/OpenBCISession_<date>_W02_P01_<part>D1/`, together with the completed plan CSVs, the generator manifests and one reference photo per part showing S0, S3 and the wrist tape at the band line. Accepted raw files are never edited. Segment with:

```bash
python -m scripts.data.prepare_w02_session --session-dir <folder> --plan <completed plan CSV> --output-dir <new dir>
```

Segmentation follows the widget timing grid, anchored at the first sample of the file: 3 s of preparation, then one 3 s slot per prompt letter. Stroke k takes the label of the k-th prompt letter. Its 2 s window is centred on the maximum of the C170 localizer energy inside slot k. Every slot gives a window. A recording is flagged only when it ends before its last slot. Each stroke records its offset in the slot and whether the slot contains a C170 localizer peak; these are diagnostics and never drop a stroke (2026-10-15 change; earlier segmentations required exactly 20 localizer peaks).

## 8. Optional cross-person extension

If Stage 1 or Stage 2 passes its gate, up to two more participants can each perform 6–12 wears with newly generated plans. They are scored once by the unchanged frozen model of the passing stage, reported separately and never pooled with P01.

## 9. Safety

Stop for pain, numbness or irritation. Do not tighten the band to improve the signal. Rest between recordings without touching the band.
