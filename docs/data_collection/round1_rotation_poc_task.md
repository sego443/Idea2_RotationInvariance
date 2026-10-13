# Round-one rotation-invariance data collection task

Version: 2026-08-27. Operator: the participant. This is a preliminary proof of concept. Collect data without inspecting model results or changing test angles.

> **Correction added 2026-10-12 (participant report).** The movements were performed as an in-air swipe in each direction followed by a return to centre, the same movement as in C170, not as pen strokes on paper. The pen, paper and "no return line" wording below records the plan as written, not the executed movement. The angle direction stated below (clockwise viewed from the hand toward the elbow) has not been reconciled with the C170/W02 wrist tape, which increases counter-clockwise in that view on the right wrist.

## Deliverables and scope

Collect one participant, one wrist, one wearing episode, and one session. Do not remove or retighten the band. Use angles `0/30/60/90/120/150/180/210/240/270/300/330 degrees`, four short paper strokes `up/down/left/right`, and five consecutive repetitions per angle-direction combination. This yields 48 accepted movement recordings and 240 strokes, plus one start and one end rest recording: 50 accepted files. Preserve rejected takes separately. Estimated duration: 45–60 minutes.

The question is whether a model trained on some angles recognizes other, untrained angles within this participant/session/wear. This cannot establish cross-person, cross-day, rewearing, or arbitrary continuous-angle performance.

## Preparation

1. Permanently label the physical sensors S0/S1/S2/S3. Connect them to EXG Channel 0/1/2/3, respectively. Keep connectors, gain, and sample rate fixed. Record all channels in the OpenBCI raw format; analysis retains EXG 0–3. Confirm 250 Hz in the interface.
2. Record the wrist side. Define zero degrees with S0 on the dorsal wrist midline. Viewed from the hand toward the elbow, clockwise increases angle. Mark the zero reference on the circular fixture and band. Photograph the initial installation with S0, wrist position, and angle scale visible. Rotate only the band/fixture; keep paper, chair, and forearm orientation fixed.
3. Use the same pen, paper, and direction template. Rest the forearm naturally on the table. Paper-up is up, toward the participant is down, and paper-left/right define left/right. Draw a roughly 4 cm straight stroke from the template start point. Lift the pen without drawing a return line. Practice each direction two or three times without recording.

## One movement recording

Each CSV row corresponds to one raw file containing five strokes of one direction at one angle. A 60 BPM metronome is recommended:

1. Start recording and remain still for 3 seconds.
2. Complete the first stroke within 1 second.
3. Lift the pen, slowly move to the next start point, and remain still for at least 2 seconds.
4. Repeat for the remaining four strokes.
5. Remain still for 2 seconds after the fifth stroke, then stop recording.

Target duration is approximately 25 seconds; accept 20–35 seconds. Do not speak, adjust the strap, touch sensors, or move cables during recording.

## Fixed acquisition sequence

Use [round1_rotation_poc_plan.csv](round1_rotation_poc_plan.csv) and load [round1_rotation_poc_prompts.txt](round1_rotation_poc_prompts.txt) in the OpenBCI GUI Sequence Collection widget. TXT and CSV rows correspond exactly. `B001_A000_UP_R5` means block B001, zero degrees, up, five repetitions; filenames contain labels automatically.

The historical W01 protocol advances angles from 0 to 330 degrees in order, with up/down/left/right at each angle. Do not rearrange W01 acquisition; shuffle samples in the training loader. The sequential order is a documented confound that the proposed W02 design addresses.

1. Follow block_id order.
2. B000 is a 20-second still baseline at zero degrees.
3. B001–B048 are movement files; rotate after all four directions at an angle.
4. Rest for 2 minutes after every three angles without loosening the band.
5. Return to zero degrees after B048 and collect the 20-second B049 end baseline.
6. Immediately enter each complete generated filename in actual_filename.
7. Do not rename, overwrite, or delete raw recordings.

## Frozen splits

- Train: 0/60/120/180/240/300 degrees.
- Validation: 30/150/270 degrees.
- Test: 90/210/330 degrees.

Do not change splits after acquisition. Test angles never select preprocessing, architecture, or epochs. Do not collect irregular challenges at 1/55/123/203/317 degrees until the model and analysis are frozen.

## Immediate quality checks and retries

After each angle, verify its four nonempty files, 20–35 second durations, exactly five intended strokes, and signals on EXG 0–3 without sustained flatlining, disconnection, or obvious saturation. Confirm the angle scale, tightness, connectors, and four CSV filenames. Check integrity and hardware only; do not classify or inspect training/test scores.

Reject the entire file and repeat all five strokes for a wrong angle/direction/count, interrupted stroke or return line, strap/sensor/forearm adjustment, empty or out-of-range duration, failed channel, or disturbance such as impact, coughing, or speech. Preserve the rejected original; set accepted=no and explain in notes. Append a retry row with block_id R001/R002/..., retry_of referencing the original, and accepted=yes for the accepted retry.

Stop if the same channel fails in two consecutive files. After repair, restart from the first failed file. If repair requires removing or retightening the band, end W01 and identify the repaired acquisition as a separate wear.

## Safety, handoff, and completion

Stop for pain, numbness, irritation, or discomfort. Do not progressively tighten the band to increase signal. Keep participant, wrist, pen, paper, chair height, and support fixed. Short breaks are allowed while preserving the wear.

Provide the complete session folder (suggested name OpenBCISession_rotation_poc_round1_P01_W01_YYYY-MM-DD), completed CSV, and reference photo. Record participant P01, wrist side, wear W01, actual sample rate, start/end times, adjustments, rejected/retry counts, and anomalies or discomfort.

Completion requires all 12 angles, four accepted direction files per angle, five valid strokes per file, 48 movement files/240 strokes, two rest files, and a unique filename for every accepted CSV row. Preserve raw/rejected files with reasons and separate wear identities. These are requirements; the [actual W01 report](../results/2026-08-27-round1-rotation-invariance-feasibility.md) documents deviations.
