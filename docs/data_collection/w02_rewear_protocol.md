# W02 one-day rewear acquisition protocol

Version 2 (2026-10-12); version 1 (separate CAL recording, three days) was replaced the same day. Status: **proposed, not collected**. This document supersedes the outline in [EXPERIMENT_DESIGN.md §7](../EXPERIMENT_DESIGN.md#7-proposed-independent-follow-up-not-executed).

The hardware, firmware, gain and sample rate are unchanged from C170; only the protocol changes. Model and endpoints: [v21 design](../designs/2026-10-12-v21-wear-calibrated-classifier.md). Evidence behind each choice: [v21 development results](../results/2026-10-12-v21-development-on-existing-data.md).

## 1. Why the protocol changes

| C170/W01 property | Consequence | W02 fix |
|---|---|---|
| One recording per direction per position, always UP→DOWN→LEFT→RIGHT | Direction is confounded with recording and time. Models and calibration exploit recording-level similarity (2-stroke nearest-centroid inside a W01 recording: 0.94 F1) | Every recording interleaves all four directions in a random, balanced order |
| Positions/angles collected in increasing order | Angle is confounded with time | Wear angles are stratified over 360° and visited in random order |
| Band slid along the wrist | Few independent wear episodes | Every wear is a full doff/don |
| Calibration across recordings barely helps (neighbour-recording proxy: +0.00 to +0.05 F1) | A separate calibration recording is a weak lever | Every trial recording **starts with its own labeled 8-stroke calibration prefix** |
| No measured angle | Angle cannot be stratified | Measured angle per wear |

The prefix design matches the intended use, "start recording → 8 calibration strokes → write → stop". Because the prefix and the scored strokes contain all four directions inside the same recording, recording-level similarity cannot stand in for direction.

## 2. Supported claims

The claim covers one participant on unchanged hardware: four-direction recognition of presegmented strokes **after a complete rewear at an arbitrary measured angle**, on a different day from all training data. The primary condition uses an 8-stroke in-recording calibration. Secondary results cover cross-recording calibration, unlabeled adaptation and no calibration. Every evaluated model is frozen before the wears it is scored on. Cross-person evidence requires §8. Continuous detection, handwriting and open vocabulary are out of scope.

## 3. Preparation (before the day)

1. Keep the C170 S0–S3 labels and EXG mapping. Sensors are equally spaced around the band. Record the active EXG columns.
2. Measure and record the wrist circumference at the band line and the band circumference at the chosen strap hole.
3. Reference marks (no hardware change):
   - a washable skin mark on the dorsal wrist midline at the band line;
   - a paper angle tape (0–355° in 5° ticks) around the band, with 0° under S0.
   
   0° places S0 on the dorsal midline; clockwise increases, viewed from the hand toward the elbow.
4. Fix the setup: same wrist, pen, paper template, chair and forearm support. Use one strap hole for all wears unless comfort requires otherwise.
5. **Commit the Stage-1 freeze manifest** (`docs/evidence/v21_freeze/v21_freeze_S1.json`, design §6). Run `python -m scripts.validation.rehearse_w02_pipeline --output-dir outputs/<new dir>` on the acquisition laptop to confirm that the whole pipeline runs.
6. On the morning of the day, generate both plans with new seeds:

   ```bash
   python -m scripts.data.make_w02_session_plan --part A --day 1 --n-wears 10 --seed <new seed> --output-dir data/w02_plans/P01_AD1
   python -m scripts.data.make_w02_session_plan --part B --day 1 --n-wears 10 --first-wear 11 --seed <new seed> --output-dir data/w02_plans/P01_BD1
   ```

   Load each `_prompts.txt` into the OpenBCI GUI Sequence Collection widget and print the plan CSVs. A prompt such as `P01_AD1_W03_A135_T2_CRULDDRUL_SDLURRUDLLDRU` encodes participant, part/day, wear, scheduled angle and recording, followed by the calibration prefix (C) and the scored body (S).

## 4. Day schedule and staged freezes

| Step | Content | Est. time |
|---|---|---|
| Part A | Wears 1–10 | ~75 min |
| Midday break | Segment Part A; run `launch_v21_stage2` (fine-tunes the frozen S1 models on C170 + W01 + Part A and writes the S2 manifest, ~35 min); commit it | ~60 min |
| Part B | Wears 11–20 | ~75 min |
| After Part B | Segment Part B; run the one-shot S1 evaluation (all 20 wears) and the S2 evaluation (Part B) | ~15 min |

Each wear has one REST recording and three trial recordings (T1–T3). Each trial recording has 20 strokes: an 8-stroke prefix (2 per direction) and a 12-stroke body (3 per direction), each in random order. That gives 36 scored strokes per wear and 720 for the day.

If Stage 2 cannot be frozen during the break, skip it rather than delaying Part B or improvising. Part B still tests Stage 1. Before the scheduled evaluations, open signals only for the integrity checks in §6. Segmenting Part A for training involves no class-wise inspection.

## 5. One wear

1. Remove the band completely, shake out the hand and wait 30 s.
2. Put the band on and turn it until the angle tape reads the scheduled angle at the skin mark. Tighten to the recorded hole. Write the actual angle (nearest 5°) and hole into the plan.
3. Sit in the fixed posture and wait 60 s.
4. **REST** (20 s): hold the pen at the start point; make no strokes.
5. **T1, T2, T3**: follow the prompt letters, prefix first, then body, without pausing between them. For each recording:
   - Start recording and stay still for 3 s.
   - With a 20-BPM cue (one beat every 3 s), draw one ~4 cm straight stroke per beat (U=up, D=down, L=left, R=right on the paper template). Lift the pen without a return line, and move back to the start point between beats.
   - Stay still for 3 s after the last stroke, then stop.
6. Do not adjust the band, touch the sensors or move the cables within a wear. Any adjustment ends the wear. Mark its remaining recordings rejected and redo the whole wear at the end of its part, with the same scheduled angle and a new wear number.

## 6. Integrity checks and retries

After each recording, check only integrity:

- the file is present;
- the duration is within ±30% of the plan;
- four active EXG channels, without flatline, disconnection or saturation;
- 20 strokes in the printed order (operator self-report).

Do not classify or plot class-wise signals.

Reject and repeat a recording for a wrong or skipped letter, an extra stroke, a return line, an interruption, speech, a cough, an impact or a failed channel. Keep the rejected file (rename it to `.rejected`), set `accepted=no` with a reason, and append a retry row with `retry_of`. Repeating a recording within the same wear is allowed, because the band is not touched. Stop if a channel fails in two consecutive recordings; after repair, start a new wear from REST.

## 7. Data handling

Use session folders named `data/OpenBCISession_<date>_W02_P01_<part>D1/`, together with the completed plan CSVs, the generator manifests and one reference photo per part showing S0, the skin mark and the angle tape. Raw and rejected files are never edited. Segment with:

```bash
python -m scripts.data.prepare_w02_session --session-dir <folder> --plan <completed plan CSV> --output-dir <new dir>
```

Segmentation must detect exactly 20 strokes per trial recording. Any other count flags the recording without repairing it. Labels come from the prompt letters.

## 8. Optional cross-person extension

If Stage 1 or Stage 2 passes its gate, up to two more participants can each perform 8–10 wears with newly generated plans. They are scored once by the unchanged frozen model of the passing stage, reported separately and never pooled with P01.

## 9. Safety

Stop for pain, numbness or irritation. Do not tighten the band to improve the signal. Rest between recordings without touching the band.
