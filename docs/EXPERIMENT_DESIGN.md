# Experimental design

Updated 2026-10-01 from local code, metadata, dated protocols, and saved run artifacts. Executed experiments and proposed follow-up are distinguished below.

## 1. Research question and transformation levels

Can four circumferential piezoelectric channels retain signed movement-direction information while reducing sensitivity to wristband wearing position? Each example is a presegmented two-second movement; output class order is UP/DOWN/LEFT/RIGHT. The project seeks a useful temporal representation h(t), rather than collapsing four sensors to a scalar.

| Transformation | Meaning | Evidence required |
|---|---|---|
| C4 channel shift | Reindex four already measured sensor slots | Algebraic architecture argument and synthetic/input-output check |
| Ideal harmonic phase | Apply a stipulated phase action to a representation | Explicit transformation assumptions and numerical checks |
| Physical wearing-position change | Move sensors to different anatomical/contact locations | Real held-position/angle acquisition |
| Rewearing | Remove and refit the band | Independent wear/session held out after model freeze |
| Cross-day/person | Change day or participant | Corresponding grouped independent acquisition |

The first two do not establish the last three. Sensor mismatch, contact pressure, anatomy, and acquisition drift can break an idealized relation between these transformations.

## 2. Executed acquisitions

### W01, 2026-08-27

One nominal participant/session/wear, 12 angles in 30-degree steps, four paper strokes, five repetitions per combination: 48 movement files and 240 windows. Two rest baselines were planned. EXG 0–3 at 250 Hz. The task-coordinate directions are fixed as the band rotates. A roughly 4 cm stroke, lifting the pen without a return stroke, is specified in the collection protocol. Correction (2026-10-12, from the participant): W01 and C170 were both performed as the same in-air swipe followed by a return to centre, so each window holds an outward and a return burst. The pen-stroke wording does not describe the executed movement.

Frozen angles: train 0/60/120/180/240/300, validation 30/150/270, test 90/210/330 degrees, giving 120/60/60 movements. Angles/directions were collected sequentially, so acquisition time is a confound. Thirty-one movement files violate the 20-second minimum, and some wear metadata is absent. The [dated report](results/2026-08-27-round1-rotation-invariance-feasibility.md) discloses these deviations.

### C170, 2026-09-26

The nominal band circumference is 170 mm; distance_mm is a collection-position label. Recorded positions are 0,5,10,15,20,25,30 and 40–160 mm in 10 mm increments. Twenty positions × four directions × 20 output repetitions yields 1,600 windows from 80 complete recordings. Fifty-six short prompt recordings are excluded. The one synthesized B019 repetition means 1,599 original windows and one marked copy.

X_raw shape is `(1600,500,4)`, sampled at 250 Hz. Metadata includes position, direction, recording block, repetition, source repetition, duplicate flag, and original EXG columns. Forty-one recordings use columns 0,2,4,6 and 39 use 0,1,2,3. Both are confirmed to correspond to S0/S1/S2/S3. The transition occurs at B057/B058 within 70 mm.

| Split | Positions (mm) | Windows | Independent position groups |
|---|---|---:|---:|
| Train | 0,10,20,30,50,60,80,100,120,140,160 | 879 | 11 |
| Old validation | 5,40,90,130 | 320 | 4 |
| Registered test | 15,25,110,150 | 320 | 4 |
| Excluded | Entire 70 mm, plus copied B019 window | 81 | — |

Keep complete positions together. Exclude the copy in model helpers without modifying the NPZ. A position group contains four direction recordings; within-recording repetitions are correlated. Do not treat 320 test windows as 320 independent wearing episodes.

## 3. Preprocessing and sampler

Segmentation localizes centers using a third-order 0.5–12 Hz Butterworth bandpass, 75-sample smoothed envelope, minimum peak separation 525 samples, prominence 1.5, and first usable center row 376. Extract 500-sample raw windows. Keep the first 20 detected movements; if 19 are found, mark the prescribed copy of movement 10 as repetition 20. Record every source row interval and original-file hash.

Model inputs use third-order 0.5–40 Hz SOS zero-phase filtering and temporal demeaning. Early models use training-fitted per-sensor scales; v3 and many later models use a training-only shared scalar to restore scale/permutation commutation. Sensor-specific normalization remains a deliberate later ablation, not an equivalent preprocessing choice. No validation/test fitting is permitted.

The formal ring/latent/neural sampler groups by original position and direction and uses each real training window once per epoch: 19 batches of 44 and a final batch of 43. Position weights use normalized circular Voronoi lengths on the training-position coordinates; each sample has w_position/(4*n_group), with all sample weights summing to one. Shuffle targets do not alter groups, index order, or weights. Read each version's actual manifest for deviations from this common protocol.

## 4. Model and ablation families

| Family | Hypothesis or control | Main scope |
|---|---|---|
| Raw CNN | Signed temporal baseline without explicit rotation pooling | W01 and early C170 |
| Orbit CNN | Average predictions/features over four cyclic sensor shifts | Discrete C4 handling |
| Harmonic magnitude | Ring spectral magnitudes discard phase | Baseline invariance/information tradeoff |
| Independent LSTM + graph attention | Per-sensor encoding with temporal node interaction | Exploratory W01 re-evaluation |
| Latent angle | Match movement-conditioned periodic templates; marginalize nominal angle | MLP/no-angle/full/no-RMS/null controls |
| Ring v2/v3 | Shared temporal sensor encoder, directed equivariant messages, invariant late readout | CE/SupCon/slot/null and scale controls |
| Harmonic v4/v5 | Phase modeling and Nyquist handling | Signed/phase representation tests |
| Temporal phase v6 | Temporal invariant features, including learned-power variants | Phase/information retention |
| KRR v7–v10 and DTW | C4 kernel/template alternatives | Non-neural comparison |
| Complex orbit/Gram v11/v12 | Complex invariant interactions; domain-adversarial extension | Feature mechanism/domain tests |
| Sensor adapters v13 | Per-sensor gain, sensor FIR+gain, shared FIR | Sensor-difference calibration |
| Normalization v14 | Median, RMS, window-RMS, bounded gain | Scale sensitivity |
| Phase kernel v15, multiscale v16 | Alternate kernels, multiple temporal scales/Nyquist | Further exploratory alternatives |
| Regularization v17 | Six weight-decay/dropout combinations | Capacity/generalization control |
| OOF fusion v18 | Training-position crossfit for fusion fitting | Reduce old-validation fitting dependency |
| Temporal Gram v19 | gz/gzn/hybrid representations at L64/L128 | Temporal interaction retention |

Consult versioned source/designs for formulas and hyperparameters. A completed artifact directory does not imply every hypothesis succeeded. v20 is an unimplemented/unlaunched proposal in the September 30 closeout.

## 5. Matched controls and inference freeze

Within each training position, permute the four complete direction recording blocks, preserving class-block counts. Validation/test targets remain true. Apply the same preprocessing, architecture, epochs, seeds, selection policy, and fusion aggregation to true and null conditions. Log the actual permutation and prediction diversity. A low null score is evidence of label dependence, not sufficient proof of physical robustness.

Frozen F2 comprises complex_gram5, learned_power10, and v7 group_true. Each member first chooses argmax of raw neural logits or raw KRR scores. Each neural family takes majority of member labels; ties use lowest class ID. The two family winners and the KRR result cast three equally weighted votes; ties again use lowest ID. F2 is a **hard-label hierarchical vote**, not equal probability averaging. The saved rule revision corrects an earlier description while reproducing all 320 historical validation rows.

## 6. Outcomes and preregistered decisions

Primary acceptance: overall macro-F1 >=0.80 and every class recall >=0.65, with null/control and diversity review. The later improvement target 0.90 is a separate research aspiration, not a revision of the acceptance gate.

Report accuracy, macro-F1, class precision/recall/F1, confusion matrix, counts/diversity, per-position and position×channel-map metrics, seed summaries and selected epochs. Keep mean-seed scores, family ensemble scores, and fused-system scores distinct. Use recording/position groups for uncertainty estimates; seed SD measures optimization variability, not biological generalization uncertainty.

Repeated old-validation selection is exploratory. The C170 test was opened for frozen F2 on 2026-09-29. Later candidates have no new independent test. Synthetic invariance checks and same-session scores do not establish rewearing or arbitrary-angle performance.

## 7. Proposed independent follow-up, not executed

Superseded on 2026-10-12 by the [W02 rewear protocol](data_collection/w02_rewear_protocol.md) and the [v21 design](designs/2026-10-12-v21-wear-calibrated-classifier.md); the original outline is kept below.


Collect a new W02 rewear after freezing model/checkpoints/scales/classes/seeds/scorer and their hashes. Keep participant/hardware fixed if isolating rewearing; randomize or balance angle order and directions. Use the same 12 measured/reference angles, four directions, and five repetitions per combination, start/end rest and rest/RMS every three angles. Record actual angle, wrist, wear, hardware/firmware/gain/channel identity, times, interruptions, and prespecified exclusion/retry rules. Keep interrupted/repaired wears separate.

Score W02 separately without refitting, tuning, or pooling it into C170. If it motivates any change, label it development and collect W03. Add irregular-angle, cross-day, or cross-participant studies only with corresponding frozen plans and untouched grouped holdouts. W02 and irregular challenges are not claimed as collected by this release.
