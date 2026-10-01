# C170 Harmonic Phase v4 Implementation and Run Plan

This plan follows the approved design in docs/designs/2026-09-29-c170-harmonic-phase-v4.md. Keep all v2 and v3 code and output directories unchanged.

## Files

- Add src/c170_harmonic_phase_model_v4.py for fixed feature extraction, masks, and the 5,932-parameter classifier.
- Add run_c170_harmonic_phase_experiment_v4.py as a separate single-run validation-only runner. Reuse the v3 shared-scale preprocessing and safe NPZ checkpoint routines, and v2 batching, weighted CE, validation metrics, checkpoint selection, and prediction serialization.
- Keep launch_c170_harmonic_phase_v4.py owned by the execution agent; update its runner path, source hashes, and v4 config checks after the runner contract is frozen.
- Add focused model and runner tests. Do not edit v2/v3 tests or code except to reuse their unchanged helpers.

## Implementation sequence

1. Test the 12-channel feature formula on hand-checked signals, including each lag's zero prefix and no circular wrap.
2. Test C4 channel-roll invariance and common complex-phase invariance for harmonic_gram and harmonic_power, while confirming harmonic_phase retains phase dependence.
3. Implement the fixed feature masks and shared network. Verify all variants have identical trainable parameter shapes, 5,932 parameters, logits of shape B×4, and optional normalized diagnostics of shape B×192.
4. Wire the v4 runner to the fixed v3 data preparation. Recompute the training-only scale and fail closed unless it is exactly 11123.037109375; record it in the manifest.
5. Reuse the v2 879-sample no-replacement sampler and weighted-CE objective. Keep seeds, initialization rule, optimizer, epoch limit, patience, validation-only checkpoint selection, per-epoch logging, and test boundary fixed.
6. Save safe numeric weights to best_weights.npz; verify a strict allow_pickle=False reload round trip without calling torch.load.
7. Build a new v4 source snapshot and independent output root. Verify training-source and dataset hashes before any formal run.
8. Run one-epoch CUDA smoke for harmonic_gram and harmonic_shuffle. Run numerical/model forward checks for harmonic_power and harmonic_phase. Review all four configs and exact sampler counts before freezing the runner.

## Formal matrix and acceptance

After the parent authorizes execution, run harmonic_gram, harmonic_power, harmonic_phase, and harmonic_shuffle for seeds 17, 29, 43, 61, and 89. Use the frozen Python 3.10, PyTorch 1.13.1+cu117 CUDA environment, two CPU threads per run, and the approved concurrency measurement. No formal run starts before all smoke and independent review checks pass.

Aggregate only runs whose completion marker, canonical config, source provenance, manifest digest, and all artifact digests verify. Report validation macro-F1, each class recall, all four validation positions, selected train-target metrics and train/validation gaps, per-seed paired differences, shuffle prediction distribution/diversity, and the frozen shared scale. Use population standard deviations. The candidate gate is five-seed mean validation macro-F1 ≥ 0.80 and mean recall ≥ 0.65 for every action. A failed gate leaves the registered test set unscored. The report must say that full NPZ decoding and finite-check may include registered-test waveforms, while test data are not preprocessed, scaled, predicted, or scored.

## Verification

Use the frozen Python executable for py_compile, focused tests, safe-NPZ reload, and CUDA checks. Required synthetic checks cover exact feature equations, finite values, zero-padded lag boundaries, C4 rolls, and arbitrary shared complex phases. Formal smoke output must have 20 optimizer updates for each one-epoch run and the exact four-variant/seed config. Do not install packages, change runtime, tune hyperparameters, add feature channels, or use the test split.
