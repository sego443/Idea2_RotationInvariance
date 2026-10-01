# Temporal movement classifier with latent wearing angle

Historical design, 2026-09-28: not implemented or validated at the time of writing. Later implementation/results are reported separately. Input: two seconds, 250 Hz, four channels. Output: UP/DOWN/LEFT/RIGHT. Inference uses neither true angle nor movement labels. Directions are fixed in task coordinates, subject to collection-record verification, and are not redefined with band rotation.

## Evidence and mechanism

Existing mean validation macro-F1: orbit 0.671794, probability-averaged orbit 0.688593; best seed 0.77501. None meets F1 0.80/recall 0.65. Existing ring adjacency and cross-position SupCon are not contributions of this design. Three-epoch probes cannot establish fully trained capacity. C4 channel permutation and physical continuous rotation require assumptions about sensors/contact fields.

Learn movement response as a function of nominal position, retain signed temporal information, and marginalize unknown angle in a joint distribution.

## Encoder

Retain established slices, 0.5–40 Hz filter, and four training-only physical sensor scales. Both confirmed channel maps correspond to S0–S3. channel_map supports ordering and stratified audits only; block, time, and repetition are not predictor inputs.

Input `(B,500,4)`. Shared sensor Conv1d(1,16,kernel=9,stride=2,padding=4) yields `(250,16)` per sensor. Three residual temporal blocks use depthwise kernel 5, pointwise 16→16, GELU, and per-time-feature LayerNorm, dilation 1/2/4 with length-preserving padding. Concatenate four sensors into 64 features per time step; linear 64→32; AdaptiveAvgPool1d(16); flatten to 512. Add four calibrated waveform log-RMS features standardized with training-only mean/std to obtain 516-dimensional h(X). Remove RMS in an ablation. No cyclic invariant averaging is applied inside this encoder.

## Continuous templates and joint inference

For each movement y:

\[
\mu_y(\theta)=A_{y,0}+\sum_{m=1}^{2}[A^c_{y,m}\cos(m\theta)+A^s_{y,m}\sin(m\theta)].
\]

Four classes × five coefficients × 516 features gives 10,320 parameters. M=2 is a fixed smoothness assumption to be tested, not a raw-waveform bandlimit claim. Set J=64 uniform integration nodes theta_j=2*pi*j/J. Node density does not imply acquisition-angle precision. L2-normalize h and templates to u and v:

\[
s_{yj}=16u(X)^\top v_y(\theta_j),\quad
p(y,\theta_j|X)=\frac{e^{s_{yj}}}{\sum_c\sum_k e^{s_{ck}}},\quad p(y|X)=\sum_jp(y,\theta_j|X).
\]

Predict argmax p(y|X). The angle posterior sums over y and may be multimodal; no unique angle estimate is required before correction. Marginalization alone does not ensure physical invariance and cannot create information where class response trajectories overlap.

## Objectives and sampling

Nominal alpha_i=2*pi*distance_mm/170 is weak supervision, not independently measured physical angle. Action loss is -log p(y_i|X_i). Set q_ij proportional to exp[8*cos(theta_j-alpha_i)] and angle loss to -sum_j q_ij log p(theta_j|X_i,y_i). Bandwidth 8 is a fixed loss setting, not measurement uncertainty. True action appears only in training loss.

Let Delta=2*pi/J with periodic indices:

\[
L_{smooth}=\frac{1}{4J}\sum_y\sum_j\left\|\frac{v_y(\theta_{j+1})-2v_y(\theta_j)+v_y(\theta_{j-1})}{\Delta^2}\right\|^2,
\qquad L=L_{action}+0.2L_{angle}+0.001L_{smooth}.
\]

Regularize normalized templates to prevent evasion through shrinking coefficients. AdamW lr/weight_decay 0.001; max 160 epochs, patience 25; seeds 17/29/43/61/89. Shuffle each of 44 original position-direction groups without replacement. Nineteen batches have one sample from each group (44); the last has 43, excluding B019's 19-sample group. Randomize batch order. Each epoch uses all 879 unique windows once: 19*44+43. Exclude the copied window; do not average waveforms across positions.

Training-position circular Voronoi lengths define normalized position weights. Each sample has w_position/(4*n_group), n_group 19 or 20. Derive weights solely from coordinates, record them, and report whole-epoch weighted losses. The later execution plan specifies batch-update scaling.

## Ablations and diagnostics

- A: same 516-dimensional encoder, 516→20→4 MLP head (~10,424 parameters), controlling capacity.
- B: full structure, angle weight 0, smoothness 0.001.
- C: full design, angle 0.2, smoothness 0.001.
- D: full design without four RMS features, 512-dimensional templates.
- Shuffle: C with complete direction-block permutation within training positions; validation labels remain true.

Use the same split and five seeds. On C's same checkpoint, diagnose known nominal angle by q-weighted p(y|X,theta). This validation-angle-assisted score is not deployment performance or a theoretical bound and does not select models. If C does not improve A/B or conditioning is ineffective, do not claim that the mechanism succeeds.

Report overall/per-position F1, class recalls, predicted class count, channel-map strata, posterior entropy, and movement-stratified nominal-angle errors. Inspect 90 mm DOWN and 40 mm RIGHT without changing labels/splits. Verify periodic continuity, probability normalization, 64→128-node validation-only numerical stability, reload consistency, and absence of test metrics.

## Split and claim boundaries

Train 0,10,20,30,50,60,80,100,120,140,160 mm; validation 5,40,90,130; registered test 15,25,110,150. Exclude 70 mm and the B019 copy: 879/320/320. Preserve raw data.

The historical test gate requires five-seed validation mean F1 >=0.80 and each mean recall >=0.65, with null and position checks. Freeze checkpoints/preprocessing/scales/seeds/inference before any test. Group uncertainty by recording or position; 20 repetitions are not 20 independent wear conditions. Repeated validation selection and one-session confounding require new rewearing data. At design time, original-session access restrictions limited rest/source audits but not validation on processed data; hardware causality was unconfirmed.

The candidate contribution is action-conditioned continuous angle templates, joint matching, and nominal-position supervision for four-channel contact sensing. Latent marginalization, periodic bases, continuous pose, and prototype matching have precedents; novelty is not established and the architecture does not remove time/position confounding.

## Historical implementation scope and sources

Use separate model and runner; preserve existing ring experiments. Log each epoch, save checkpoints/timings immediately, and record data/code hashes, coefficients, preprocessing, classes, and integration rules. Separate training and final test. This document originally delivered design only.

The historical evidence review recorded these sources:

1. Jojic and Frey, [Topographic Transformation as a Discrete Latent Variable](https://papers.nips.cc/paper/1729-topographic-transformation-as-a-discrete-latent-variable.pdf), NeurIPS 1999: joint class/transform modeling and marginalization, primarily with known discrete operators.
2. Hinton, Krizhevsky, and Wang, [Transforming Auto-encoders](https://www.cs.toronto.edu/~hinton/absps/transauto6.pdf), 2011: continuous pose and known-transform supervision.
3. Dym, Lawrence, and Siegel, [Equivariant Frames and the Impossibility of Continuous Canonicalization](https://arxiv.org/html/2402.16077), ICML 2024: Proposition 2.7 concerns the standard SO(d) action on all R^(d×n), n>=d>=2; it does not make every angle-estimation task impossible.

These historical literature notes were translated, not freshly re-reviewed. Source modules: [rotation_poc.py](../../src/rotation_poc.py), [ring representation](../../src/c170_ring_representation.py). See [C170 acceptance record](../results/2026-09-27-c170-exploratory-validation-acceptance.md).
