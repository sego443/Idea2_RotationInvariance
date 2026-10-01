# C170 ring cross-position CUDA experiment plan

Historical status (2026-09-28): runner implementation and real one-epoch CUDA smoke authorized; formal 20 runs required a later launch. No test scoring. Reuse the established latent-angle split and preprocessing; later execution is reported separately.

## Scope and fixed matrix

Create run_c170_ring_cuda_experiment_v2.py separately from latent-angle code/results. Preserve src/c170_ring_representation.py and existing datasets/runners. Reuse original data/split/block-shuffle helpers and prepare_model_inputs; filter/calibration from rotation_poc.py. Record every computational dependency's SHA-256. This runner itself is single-run, not a launcher.

Four variants × seeds 17/29/43/61/89:

- ring_main: main invariant readout, CE + .1 cross-position SupCon.
- ring_ce: same invariant readout, CE only.
- slot_supcon: ordered slot_readout, CE + .1 SupCon.
- ring_shuffle: ring_main with complete recording-block direction permutation within training positions; validation true.

Temperature .1 from the unchanged helper; no hyperparameter search. Same seed/epoch means identical batch indices across arms; original position-direction groups define membership and weights, while shuffle changes targets only.

AdamW lr=.001, weight_decay=.001; max 160 epochs/patience 25. Highest validation macro-F1 selects a run checkpoint, ties to lower validation CE. CE label smoothing 0. Each of 20 updates uses 20*sum(w_i*CE_i), w_i=w_position/(4*n_original_group), total sample weight 1. SupCon adds .1 times its scalar batch mean; no batch-weight renormalization. Record selected-checkpoint training-target F1 and weighted CE.

## Split and test access

Train 0,10,20,30,50,60,80,100,120,140,160; validation 5,40,90,130; registered test 15,25,110,150 mm. Exclude all 70 mm and the copied B019 window, not its complete block. Counts 879/320/320. Sampler uses 19×44+43 unique windows once per epoch. Train-only sensor scales and established Butterworth/demeaning apply to train/validation.

Legacy load_dataset decodes the entire NPZ and finite-checks all arrays, including test waveforms. After integrity checking, test does not enter preprocessing, scale fitting, model input, prediction, statistics, scoring, or checkpoint selection. Preserve this distinction rather than claiming zero test reads.

## Artifacts and gate

New exclusive run directory: canonical config/hash, data/source hashes, split metadata, shuffle map, atomically updated learning_curve.csv, validation predictions, best_checkpoint.pt, selected training metrics, run manifest, and RUN_COMPLETE.json only after artifacts complete. Log variant/seed, timings/updates/losses, F1/CE/recalls/diversity, position/channel strata, SupCon, selected/actual epochs.

ring_main gate: five-seed mean validation F1 >=.80 and every mean recall >=.65. Report all variants/nulls and position confusion. Test remains separately gated, with no automatic evaluation.

## Historical implementation sequence

1. Contract tests for sampler, variant/loss mapping, block shuffle, metrics, and no test scoring; observe failure before implementation.
2. Reuse the unchanged model/loss and validated helpers; hash dependencies.
3. Run tests/syntax and real CUDA one-epoch smoke for all four arms; inspect completion marker, checkpoint, CSV/manifest hashes, 879 unique windows and test_metrics_computed=false.
4. Freeze runner and report smoke evidence before formal launch.
5. Formal source_snapshot copies dependencies, plan/design, and environment. A separate fail-closed analyzer verifies exactly 20 arm/seed runs and marker/config/artifact/source hashes.
6. Analyzer reads only saved train/validation artifacts, not dataset/test; report seed statistics, strata, confusion, paired differences, curves, epochs, training metrics and gate.

Reject wrong group sizes (43×20 plus one×19), duplicates/omissions or differing sampler indices; missing cross-position positives; altered shuffle grouping/weights or validation labels; invalid splits/finite values/channel order; existing output or bad hashes; or premature completion markers. Test preprocessing, prediction, or scoring is outside this plan.
