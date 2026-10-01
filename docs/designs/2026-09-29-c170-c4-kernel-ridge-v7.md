# C170 C4 group-averaged kernel ridge study (v7)

## Goal and scope

This train/validation-only study asks whether a full discrete C4 group-averaged RBF kernel retains action information across held-out recording positions under fixed, explicit signal descriptions. The kernel is the group average

`K_G(x,y) = (1/4) * sum_{g in C4} exp(-||phi(x)-g phi(y)||^2 / (2 sigma^2))`.

The average is mandatory; replacing it with a minimum orbit distance or maximum single-orientation kernel changes the method and is forbidden. Normalize the group kernel by `sqrt(K_G(x,x) K_G(y,y))`. A plain normalized RBF kernel over the same descriptors is the matched non-invariant control. Claims are limited to the discrete C4 channel-permutation group, not continuous physical rotation invariance.

## Fixed data boundary and preprocessing

Use the existing C170 dataset, channel order, distance split, exclusions, and train/validation counts: train has 879 windows at positions 0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160 mm; validation has 320 windows at 5, 40, 90, 130 mm; 70 mm is excluded. The registered test portion may only be decoded by the integrity-checking loader. It must not be filtered, transformed, summarized, predicted, or scored.

Keep the raw input dtype while filtering. Compare three shared-across-sensor preprocessing choices: existing third-order zero-phase 0.5–40 Hz bandpass; per-window/per-channel demean only; third-order zero-phase 40 Hz low-pass followed by per-window/per-channel demean. All four sensors use the same operation.

## Per-sensor descriptions and views

For each window and sensor, derive four feature families from the chosen preprocessed signal:

1. `a`: demeaned waveform divided by that sensor's full-window RMS, then `resample_poly(up=16, down=125)` to 64 samples.
2. `b`: eight contiguous block RMS values divided by full-window RMS.
3. `c`: log relative Hann-periodogram powers in `[0.5,4)`, `[4,8)`, `[8,15)`, `[15,40)`, and `[40,125]` Hz.
4. `d`: log full-window RMS.

The three fixed views are `a+b`, `b+c+d`, and `a+b+c+d`. Use a fixed epsilon for zero RMS and spectral denominators and keep all resulting values finite. For each training fold, fit coordinate-wise mean and population SD using only the fold's training windows and all four sensors together. Apply those shared statistics to every sensor; never fit separate scales for fixed channels. Standardize each coordinate, scale each feature family by `1/sqrt(family_dimension)`, then scale all included families by `1/sqrt(number_of_families)`.

## Training, cross-validation, and selection

Run leave-one-position-out CV over the 11 training positions. For each fold, compute the base bandwidth from ordinary Euclidean descriptor distances among distinct fold-training pairs only: `sigma_base = sqrt(median(squared_distance_i_j))`. Evaluate sigma factors `0.5, 1, 2` and ridge alphas `0.01, 0.1, 1, 10`, for 108 candidates per arm. Score each candidate by the equally weighted mean of the 11 held-out-position macro-F1 values. Scores within `1e-8` tie; prefer larger alpha, then the fixed candidate enumeration order (sigma factor ascending, alpha ascending).

For each preprocessing/view pair, fit a true-label C4 group-kernel arm and five C4 group-kernel null arms (seeds 17, 29, 43, 61, 89). For each null seed, shuffle whole recording-block labels once on the full 879-window training set within each training position using the established block-shuffle rule; hold those pseudolabels fixed across all 11 CV folds. Weights always use original true action labels, position, and block. The plain RBF control is true-label only. This gives 9 feature configurations × (1 group true + 5 group null + 1 plain RBF true) = 63 searched arms, each with 108 candidates.

The weighted ridge solve is `A = sqrt(W) K sqrt(W) + alpha I`, `B = solve(A, sqrt(W)(Y-0.25))`, `C = sqrt(W)B`, and `scores = 0.25 + K_query C`. Recompute fold weights from only fold-training rows using the existing position/block/direction rule, then normalize them to mean one. For shuffled-label arms, `Y` uses fixed pseudolabels while weights still use original labels. Use CUDA `float64` Cholesky/solve, batch alphas and four label right-hand sides where practical, check finite outputs and residuals. A failed numerical check fails the arm; it is not silently replaced with a different solver.

After CV chooses one configuration per arm, fit once on all 879 training windows, then score the 320 validation windows once. External validation uses true action labels for every arm, including shuffled-label controls. Validation scores do not influence selection. LOPO scores are tuning criteria, not unbiased performance estimates. The deterministic algorithm is run once per configuration; do not manufacture repeated-seed standard errors for real-label arms.

## Artifacts and safeguards

Save a canonical JSON configuration and hashes, per-arm CV table, selected configurations, validation metrics/predictions, feature-fold statistics, training descriptors, and coefficients. Numeric arrays go to NPZ and must reload with `allow_pickle=False`; metadata stays in JSON. Snapshot frozen code, design/plan, environment, and dataset hash before a formal matrix. All writes use new output roots and refuse overwrite. All progress monitoring reads append-only stdout, never a CSV being atomically replaced.

## Required verification

Verify symmetric positive-semidefinite group kernels on deterministic fixtures; invariance of group-kernel train/query scores under all four cyclic channel rolls; that the ordinary RBF control changes on a nontrivial roll; fold statistics are fit without the held-out position; Cholesky residuals are small and finite; numeric NPZ reload preserves arrays; and a selected trained model's C4 score error is measured on train/validation only. Smoke tests use actual train/validation data and a reduced candidate subset. No test predictions or metrics are computed.
