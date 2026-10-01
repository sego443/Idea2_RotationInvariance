# C170 C4 kernel ridge v7 implementation plan

## Files

- `src/c170_c4_kernel_ridge_v7.py`: three preprocessors, four per-sensor feature families/views, fold-only shared normalization, C4 group-averaged and ordinary RBF kernels, weighted KRR, numeric safe persistence helpers.
- `run_c170_c4_kernel_ridge_v7.py`: fixed split, whole-block null labels, exact 11-position CV and deterministic tie rule, train-only selected fit, one validation evaluation, smoke/formal CLI, manifests and provenance.
- `tests/test_c170_c4_kernel_ridge_v7.py`: focused mathematical and leakage checks on synthetic fixtures.
- `docs/results/2026-09-29-c170-c4-kernel-ridge-v7-smoke.md`: real train/validation smoke and measured run-size estimate, written before any full formal matrix.
- `outputs/c170_c4_kernel_ridge_v7_formal_20260929/`: one fresh source snapshot and later formal artifacts; do not create the formal snapshot/run until parent reviews the smoke estimate.

## Tasks

1. Add focused tests for feature dimensions/finite zero handling, family and sensor-shared statistics, group-kernel symmetry/PSD and C4 invariance, ordinary-RBF behavior, exact weighted solve/residual, and safe NPZ reload.
2. Implement feature transforms and split-fit normalization; run the feature and leakage tests.
3. Implement group-averaged normalized RBF and matched plain RBF, batched weighted ridge solves, and selection tie handling; run the kernel/solve tests.
4. Implement the train/validation-only runner: three preprocessors × three views; true and five fixed per-position block-shuffled labels for group kernel; true labels for the plain control; 11-fold position holdout; 108 candidates/arm; selected train refit and one external validation score.
5. Run focused tests and `py_compile`; run the reduced actual-data CUDA smoke only. Validate finite kernels/solves, C4 score errors, hashes, and safe persistence. Write a brief measured smoke/estimated-work report and return its path plus runtime estimate to the parent. Stop before the 63-arm formal matrix until the parent reviews that estimate.

## Fixed work size

The full grid has 63 arms and 108 candidates per arm. For each arm, each LOPO fold computes three sigma kernels and four batched alpha solves, for 11 folds. Therefore the expensive selection stage is 2,079 fold-kernel preparations and 2,079 batched four-alpha solves, plus 63 selected full-train fits and validation evaluations. The smoke should report wall time by feature extraction, CV, and final fit/validation to support a realistic estimate.

## Stop conditions

Never touch the original dataset. Do not calculate test metrics. Do not use external validation in candidate selection. Fail closed on non-finite features/kernels/solves, large Cholesky residual, changed source/data hashes, or malformed safe NPZ. Do not start the full matrix until the parent explicitly reviews the smoke/runtime report.
