# v21 wear-calibrated direction classifier

Version 2 (2026-10-12). Version 1 of the same day (separate CAL recording, DEV/TST days, a cross-validation grid on W02 DEV) was replaced after the existing-data experiments in [v21 development results](../results/2026-10-12-v21-development-on-existing-data.md).

Status:
- implemented and smoke/rehearsal tested;
- **W01 is excluded from training (amendment, 2026-10-12, before any W02 collection).** W01 was a preliminary, non-rigorous feasibility collection. The first Stage-1 freeze, `docs/evidence/v21_freeze/v21_freeze_S1.json`, was trained on C170 + W01; it is superseded and kept unchanged as a historical record. Stage 1 must be retrained on C170 only with `scripts/launchers/launch_v21_stage1.py` and re-frozen as `docs/evidence/v21_freeze/v21_freeze_S1_c170.json`, which does not exist yet. Recipe, seeds, endpoints and τ values are unchanged. The W01 results in §1 were used to choose the design and remain exploratory development evidence; they are not training data for any frozen model;
- **no W02 data exists**, and no W02 result is claimed.

v20 is an unrelated earlier proposal. The protocol is in [W02 one-day protocol](../data_collection/w02_rewear_protocol.md).

## 1. Evidence that shaped the design

The full numbers are in the development results.

- Frozen F2 failed its registered test (0.675). Zero-calibration transfer across positions is wear-specific, and across sessions (C170 → W01) it is at chance (0.19–0.27) for every v21 variant.
- Calibration helps strongly when the calibration strokes come from the same recording as the scored strokes:
  - C170 held-out positions: 0.85 (B2) vs 0.52 without calibration;
  - W01 cross-session: 0.95 vs 0.27.
- Calibration from a different recording helps little. Calibrating from a neighbouring position's recordings gave +0.00 to +0.05 over zero calibration.
- In C170/W01 every recording holds one direction, so same-recording calibration partly exploits recording identity: a training-free nearest-centroid reaches 0.94 on W01. W02 removes that confound by mixing directions inside every recording, so in-recording calibration must rely on direction-specific waveform.
- Episodic (prototypical) training is required for calibration to transfer across sessions: W01 CAL 0.33 → 0.84 (B1) and 0.79 → 0.95 (B2). Neighbour-recording episodes did not improve cross-recording calibration.

## 2. Pre-registered endpoints

Scored strokes are the 12 body strokes of every trial recording. Prefix strokes are never scored. Unit of analysis: the wear.

| Endpoint | Calibration input | Model family | τ | Role |
|---|---|---|---:|---|
| **E1** | Labeled 8-stroke prefix of the same recording | B2 signed CNN | 1 | **Primary** |
| E1b | Prefixes of the other recordings of the same wear (five under protocol version 3) | B1 complex-Gram | 16 | Secondary (cross-recording) |
| E2 | Same-recording prefix with labels ignored, plus earlier body strokes (causal) | B1 | 1 | Secondary (label-free) |
| E3 | None | B1 | — | Secondary (strict zero calibration) |
| E4 | Same-recording prefix, nearest centroid on raw features, no training | — | — | Training-free comparator |

**Gate** (E1): macro-F1 ≥ 0.80 and every class recall ≥ 0.65.

**Reported for every endpoint:**
- macro-F1 and accuracy;
- recalls and confusion matrix;
- prediction counts;
- per-wear F1 and per-90°-quadrant F1 of the measured angle;
- 95% wear-cluster bootstrap interval (10,000 resamples, seed 20261012);
- E1 of five matched-null B2 models (training labels permuted by recording block within every group).

The ensemble rule is majority vote over the seeds of the endpoint's family, with ties going to the lowest class ID. The secondary endpoints are reported regardless of outcome and are never used to rescue E1.

## 3. Two staged confirmatory tests on one day

| Stage | Frozen model trained on | Frozen when | Scored on |
|---|---|---|---|
| S1 | All C170 (1,599); W01 excluded | Before the acquisition day | All 12 wears (864 body strokes) |
| S2 | C170 + W02 Part A (6 wears, 720 strokes incl. prefixes) | During the midday break | Part B (6 wears, 432 body strokes) |

Wear counts follow [protocol version 3](../data_collection/w02_rewear_protocol.md) (amended 2026-10-12, before any collection): 12 wears × 6 trial recordings, replacing 20 wears × 3. The frozen S1 manifest, its scorer and the endpoints do not depend on the number of wears or recordings.

Each stage is a separate one-shot test of a model frozen before its test wears exist. Both stages are reported. A gate pass is claimed only for the stage that passes, with the other stage reported alongside; this is two pre-specified tests, not a choice made after seeing results. S2 is expected to be stronger, because Part A provides mixed-direction recordings with proper prefix → body episodes.

## 4. Model

Input: 0.5–40 Hz third-order SOS zero-phase filter and per-window demeaning. A shared scalar (median absolute value) is fitted on the training data of each run. The encoder outputs a 64-d L2-normalized embedding:

- **B1:** the v11 complex-Gram descriptor (C4-invariant by construction) plus a linear projection.
- **B2:** a signed four-channel temporal CNN (asinh input; stem plus three dilated residual blocks; temporal descriptor plus per-channel log RMS) with a linear projection.

Heads: learned class prototypes g_c with cosine logits at scale 16.
- **Calibration:** p_c = normalize(Σ support embeddings of class c + τ·g_c). Classes without support keep g_c.
- **Unlabeled refinement:** two soft k-means steps.

## 5. Training recipe (fixed)

- **Objective:** cosine-softmax cross-entropy on shuffled batches of 64, plus one episodic loss per step (weight 1).
  - Each episode takes 4 groups. Support is 2 strokes per direction, query is 3 per direction, and the training τ is 4.
  - C170 groups draw support and query from the same group ("same" mode).
  - W02 groups draw support from a recording's prefix and queries from its body.
- **Augmentation:**
  - per-channel and global gain ±6 dB;
  - zero-padded shift ±200 ms;
  - time warp ±10%;
  - 1% RMS Gaussian noise.
- **Optimizer:** AdamW, lr 1e-3, weight decay 1e-3, 120 fixed epochs, no early stopping.
- **S1:** families B1 and B2, seeds 17/29/43/61/89 each, plus five B2 matched nulls (`--null-seed` = seed), protocol `c170_all`. `scripts/launchers/launch_v21_stage1.py` runs this and writes the S1 manifest.
- **S2:** fine-tune every S1 run (10 real, 5 null; nulls keep their null seed) for 30 epochs at lr 3e-4 on C170 + W02 Part A (protocol `c170_plus_w02`), keeping each starting checkpoint's shared scale. `scripts/launchers/launch_v21_stage2.py` runs this and writes the S2 manifest.
  - On the W01 analog (C170 + 6 W01 angles, scored on the other 6), fine-tuning matched from-scratch training on in-recording calibration (B2: 0.991 vs 0.986) at about a quarter of the cost.
  - The S2 set needs about 35 min on the M2 CPU (4 workers), so it fits the break.

## 6. Freeze and evaluation commands

```bash
# S1 (C170 only): train, freeze and commit before the day (CUDA: add --device cuda)
python -m scripts.launchers.launch_v21_stage1 --output-root outputs/v21_stage1_c170_<date> \
  --s1-freeze-output docs/evidence/v21_freeze/v21_freeze_S1_c170.json
git add docs/evidence/v21_freeze/v21_freeze_S1_c170.json && git commit -m "Freeze v21 Stage 1 (C170 only)"

# Midday break: segment Part A, then fine-tune and freeze S2; commit before Part B
python -m scripts.data.prepare_w02_session --session-dir <Part A folder> --plan <completed Part A plan> --output-dir outputs/w02_partA_<date>
python -m scripts.launchers.launch_v21_stage2 --s1-freeze docs/evidence/v21_freeze/v21_freeze_S1_c170.json \
  --w02-dataset outputs/w02_partA_<date>/dataset.npz --output-root outputs/v21_stage2_<date> \
  --s2-freeze-output docs/evidence/v21_freeze/v21_freeze_S2.json
git add docs/evidence/v21_freeze/v21_freeze_S2.json && git commit -m "Freeze v21 Stage 2"

# After Part B: segment A+B together, then one-shot evaluations
python -m scripts.data.prepare_w02_session --session-dir <A folder> --session-dir <B folder> --plan <A plan> --plan <B plan> --output-dir outputs/w02_day1_<date>
python -m scripts.analysis.evaluate_v21_frozen_w02 --freeze-manifest docs/evidence/v21_freeze/v21_freeze_S1_c170.json --w02-dataset outputs/w02_day1_<date>/dataset.npz --output-dir outputs/v21_S1_evaluation_<date>
python -m scripts.analysis.evaluate_v21_frozen_w02 --freeze-manifest docs/evidence/v21_freeze/v21_freeze_S2.json --w02-dataset outputs/w02_day1_<date>/dataset.npz --output-dir outputs/v21_S2_evaluation_<date>
```

Run these commands from the project root on the machine that holds the Stage-1 output root. Checkpoints are local and not in Git; their hashes are in the manifest. If the acquisition laptop is a different machine, copy that directory over unchanged.

The evaluator checks the scorer hash and the checkpoint hashes, and writes an exclusive `EVALUATION_STARTED.json` before scoring. Rerunning it on the same wears produces no new evidence.

## 7. Risks

| Risk | Status / mitigation |
|---|---|
| In-recording calibration on mixed recordings is weaker than the single-direction-recording estimates | Unknown until W02. E4 shows how much is template matching; S2 adds proper training episodes |
| S1 transfer from 2026-08/09 sessions to the new day | Zero-shot cross-session is at chance; S1 depends on calibration, and S2 adds same-day data |
| Drift between prefix and late body strokes | 20-stroke recordings (~66 s); per-position effects reportable from saved predictions |
| Small test (12 or 6 wears) | Wear-cluster bootstrap; optional extra participants only after a pass |
| Primary relies on calibration | The claim is stated as calibration-assisted; E1b/E2/E3 quantify the gap |
