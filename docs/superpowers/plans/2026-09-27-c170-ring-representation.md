# C170 temporal ring representation implementation plan

Historical approved design/plan, 2026-09-27. Checklist items below describe planned work at that date; they do not certify later completion. Later v2/v3 runners supersede some sampling/training details.

Goal: preserve temporal relationships in a C4 ring representation, add cross-position supervised contrastive loss, and expose a separate validation-only runner. Python/NumPy/SciPy/PyTorch; the historical sibling-project virtual environment is replaced by a standalone environment for reproduction.

Mechanism precedents: [Cohen and Welling 2016](https://arxiv.org/abs/1602.07576) and [Khosla et al. 2020](https://papers.nips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf). These support discrete group equivariance/SupCon, not novelty of this combination or continuous-angle generalization. This translation does not constitute a new literature review.

## Constraints and review questions

Scope only this project; preserve run_c170_experiment.py behavior and original C170 data. Reuse loading, split, channel-order checks, filtering, calibration, metric/prediction helpers; record SHA-256. Train distances 0,10,20,30,50,60,80,100,120,140,160; validation 5,40,90,130; register test 15,25,110,150; exclude 70. Physical order S0–S3. Validation-only computes no test metrics. Historical task uses real processed C170, no synthetic samples or original-session access, and only three-epoch speed measurements rather than formal five-seed training.

Review node equivariance/logit invariance, independent +1/-1 parameters, length preservation/dilations, same-direction/different-position/different-block positives, denominator exclusions, shuffled targets in both CE/SupCon, and per-seed curves/predictions/position metrics/checkpoint hashes/reload checks.

## Architecture

1. `(B,500,4)`→`(4B,1,500)`. Shared Conv1d 1→16 (k9,s2,p4), 16→32 (k7,s2,p3), 32→32 (k5,s1,p2), each GroupNorm/GELU. Restore `(B,4,32,125)`; striding compresses samples but subsequent ring layers preserve ordered feature-time positions.
2. Two residual ring blocks preserve shape. Shared self pointwise transform and separately parameterized +1/-1 temporal convolutions, kernel3/dilation1 then2. Cyclic node rolls, per-node GroupNorm/GELU, residual addition preserve C4 equivariance; no reflections.
3. main/classification_only have identical invariant readout: node mean/max each `(B,32,125)`, ordered adjacent pairs `(B,4,64,125)` encoded by shared temporal convolution to `(B,4,32,125)` then edge mean. Concatenate `(B,96,125)`. Only loss differs (main CE/SupCon, classification_only CE).
4. Conv1d 96→64 (k5,p2), AdaptiveAvgPool1d(8), flatten, four-class head. A training projection head produces normalized 32-dimensional features. forward accepts only waveforms; separate method returns projection.
5. slot_readout flattens ordered nodes to `(B,128,125)` and uses the same kind of terminal temporal classifier, controlling the full readout route rather than claiming equal parameter count.

## Loss and original sampler proposal

Main objective CE+.1 SupCon, temperature .1, normalized projections. Positives require same training-target direction, different distance and block; assert blocks explicitly. Negatives have different targets. Same-position/same-target pairs enter neither positive set nor denominator. Average each anchor's positive log-probabilities then anchors.

Original batches draw one real window from each of 11×4 groups (44), yielding ten cross-position positives and 33 negatives per anchor, 20 batches/epoch. The original text incorrectly identified B019 as 120 mm LEFT; metadata places it at 20 mm LEFT. The original proposal would repeat one real sample from its 19-window group in the last batch. The later v2 protocol instead uses 19×44+43 without replacement; use that later protocol for reproduction.

Original classification_only/slot_readout contrastive weights are 0. main_block_permutation swaps complete direction blocks inside training positions and uses permuted targets in both losses; true targets are used only for validation.

## Original budget

Variants main_true/main_block_permutation/classification_only/slot_readout, five seeds 17/29/43/61/89. Original plan uses AdamW lr .003, weight_decay .001, CE smoothing .02, max160/patience25; validation F1 then CE selects checkpoint, seed order breaks aggregate ties. Later v2 uses lr .001 and smoothing0.

Twenty updates per epoch: <=3,200/seed and <=64,000 for four×five matrix, not a claim of equal budget with old candidates. Initial speed measurements run four variants×one seed×three epochs with full real train/validation, saving each candidate immediately. Formal runs required a separate launch.

## Planned tasks

### Task 1: model, loss, sampler

Files src/c170_ring_representation.py and tests/test_c170_ring_representation.py. Interfaces: RingRepresentationClassifier, encode_nodes, build_cross_position_masks, cross_position_supervised_contrastive_loss, build_balanced_batches, projection_diagnostics (effective rank/class-center distance).

- [ ] Write real-data shape/output tests and observe initial failure.
- [ ] Implement shared temporal encoder, independent directed messages, invariant/slot readouts, masks/loss, balanced sampling and diagnostics.
- [ ] Verify node equivariance, classifier invariance, parameter independence and pair coverage.

### Task 2: separate validation-only runner

Files run_c170_ring_experiment.py and tests/test_run_c170_ring_experiment.py. CLI phase only validation-only, dataset/exclusive output, speed-probe/formal budgets. Reuse dataset_sha256, existing evaluation/metric/prediction helpers. Checkpoint records format/version, kind/architecture/state, classes/hyperparameters/preprocessing/selection/channel confirmation, splits, data path/hash.

- [ ] Test that test positions appear only as configuration metadata and metrics contain train/validation only; observe initial failure.
- [ ] Implement four variants, shuffle, early stopping, complete curves/checkpoints/reload/data hashes/position metrics/immediate saves.
- [ ] Run four variants×one seed×three-epoch speed measurement, saving configuration, timings and budget estimates.

### Task 3: review and evidence

- [ ] Record dataset hash/counts, parameter/update budgets, elapsed time and formal upper-bound estimate.
- [ ] Verify saved/reloaded predictions agree and retain full curves/per-position metrics.
- [ ] Describe C4 as normalized-input discrete index behavior; position holdout does not establish continuous rotation or rewearing.

The originally proposed docs/results/2026-09-27-c170-ring-representation-validation.md is not present in the current tree; do not treat a proposed filename as evidence. See [later protocol review](../../results/2026-09-28-c170-ring-v2-runner-protocol-review.md) and [current results](../../EXPERIMENT_RESULTS.md).
