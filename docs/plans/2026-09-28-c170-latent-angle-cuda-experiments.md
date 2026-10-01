# C170 latent-angle CUDA experiment plan

Historical status (2026-09-28): train/validation authorized; final test not authorized by this plan. Design: [latent-angle classifier](../designs/2026-09-28-latent-angle-temporal-classifier.md). Later completion is documented separately.

## Scope and matrix

Create a separate runner and Windows launcher, reusing dataset/split/channel/preprocessing helpers and the latent-angle model. Preserve raw/processed data and old runners/results. Full NPZ loading decodes test waveforms and metadata; training computes no test statistics, fitted scale, predictions, or metrics. Do not call the legacy final-test interface or choose selected_best_seed_for_test.

25 runs: mlp/no_angle/full/no_rms/shuffle × seeds 17/29/43/61/89. MLP uses the 516→20→4 head and no angle/smoothness terms. no_angle uses angle weight 0 and smoothness .001. full uses .2/.001. no_rms removes four RMS features. shuffle uses full losses with complete training-block direction permutation within each position; validation remains true.

All arms share split, seed, per-seed/per-epoch batch indices, max 160 epochs and patience 25. Select highest validation F1, then lowest CE. Record actual epochs, updates, curves, and wall time. Save each finished run immediately, including configuration, data/code fingerprints, logs, and checkpoint, in a new output directory; refuse overwrites.

## Data and objective

Train 0,10,20,30,50,60,80,100,120,140,160 mm; validation 5,40,90,130; unscored registered test 15,25,110,150. Exclude 70 mm and copied B019 window; counts 879/320/320. Reuse dataset and integrity helpers from run_c170_experiment.py and train/validation preprocessing from run_c170_ring_experiment.py.

Third-order Butterworth SOS 0.5–40 Hz with sosfiltfilt, then per-window/channel temporal demeaning. Fit four sensor scales and RMS mean/std only on training waveforms; reuse on validation. channel_map supports ordering/strata, not prediction; block/time/repetition are not inputs.

Group sampling/weights by original distance and direction, unaffected by target shuffling. Independent sampler RNG derives from seed/epoch. Use 19 batches of 44 and one of 43, exactly 879 unique training windows once; exclude B019's 19-sample group from the last batch. No cyclic 880-sample padding.

Circular Voronoi position weights on a 170 mm circumference yield sample weight w_position/(4*n_group); all weights sum to 1. Each of 20 updates uses:

```text
L_batch = 20 * sum(w_i * (L_action_i + lambda_angle * L_angle_i)) + .001 * L_smooth
```

Do not renormalize by batch weight sum. MLP omits angle/smoothness. Report whole-epoch weighted data loss, mean smoothness, and updates.

## CUDA and throughput

Historical runtime: Python310/python.exe under the operator's local Programs/Python directory, PyTorch 1.13.1+cu117, cuda:0, RTX 2080 Ti; reported visible memory 22,528 MiB. Prefer FP32, without upgrading PyTorch. Record driver/build/device, peak GPU memory, CPU/RAM, wall time, and throughput.

Smoke and probe use real train/validation. Compare 2/4 concurrent workers on identical full-model seeds 17,29,43,61,89,101,113,127, five epochs/patience five, 2–4 CPU threads per process. Extra seeds are probe-only. Treat first epoch as startup and compare last-three steady epochs, total time, and updates. Choose concurrency after reviewing speed and memory, before the formal matrix.

## Artifacts, gate, and boundaries

Log per-epoch overall/position F1, recalls, prediction diversity, channel-map strata, angle entropy/error and per-seed nulls. Checkpoint includes state, sensor/RMS scales, classes, preprocessing, nodes/inference, seed/epoch, dependency hashes, and dataset SHA-256.

Historical full-C gate: five-seed mean validation F1 >=.80 and each mean recall >=.65, plus null/position review. Test requires a separate explicit launch after freezing all five checkpoints. Report all five mean/std rather than selecting the best test seed or ensemble. Reused validation is exploratory; no default/automatic final-test phase exists in this runner.

Files: run_c170_latent_angle_experiment.py and launch_c170_latent_angle_cuda.py. Write new outputs/c170_latent_angle_<timestamp> directories. Preserve historical outputs and design; do not modify raw/processed data or call legacy final-test. The [usage guide](../usage/c170-latent-angle-cuda-launcher.md) describes commands.
