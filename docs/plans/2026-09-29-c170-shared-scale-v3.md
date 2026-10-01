# C170 Shared-Scale C4 Equivariance v3 Implementation Plan

> **For agentic workers:** The parent has explicitly assigned implementation to this task. This plan records the fixed protocol and execution sequence; do not alter v2 or widen the experiment.

**Goal:** Add an independent validation-only CUDA runner for the approved shared-scale v3 experiment and verify its preprocessing symmetry and artifact contract.

**Architecture:** A small NumPy preprocessing module will reuse the existing filter implementation and estimate one training-only scalar after per-trial/channel temporal demeaning. The new runner will use the existing ring classifier and v2 frozen sampler, weighting, and validation helpers, while limiting its variants to CE-only true-label and block-shuffled-label runs. It will preserve the metadata checkpoint and add numeric-only NPZ state weights, loaded with pickle disabled and exact key/shape/dtype checks, as the only selected-model reload source.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch/CUDA, existing C170 helpers and ring representation.

**Spec:** `docs/designs/2026-09-29-c170-shared-scale-c4-equivariance.md`

## Global Constraints

- Keep v2 and all earlier experiment artifacts unchanged.
- Keep the 879-sample no-replacement sampler, 20 updates per epoch, training weights, optimizer, seeds, early stopping, and validation-only checkpoint selection fixed.
- Estimate the single scalar from train only; apply it unchanged to train and validation.
- Use CE only, with no SupCon; only `shared_scale_ce` and `shared_scale_shuffle` are allowed.
- Preserve the registered test positions without test preprocessing, model input, prediction, or scoring.
- Do not add or mutate source data; store only numeric state arrays and fail on unsafe contents or state mismatch.

## Review Focus

- A large validation-only outlier must not affect the fitted shared scale; test that the scale depends only on centered train samples.
- The shared scale must be one scalar across all four channels and must be bounded below by float32 epsilon for a zero train signal.
- Filtering and demeaning must preserve cyclic channel-roll correspondence before and after normalization.
- The numeric state dict must round-trip with `allow_pickle=False`, and runner reload must never call `torch.load`.
- A successful run must mark the exact 10-run variant/seed configuration and must record that the test split was not preprocessed or scored.
- Final reporting must include every validation position and class, with per-seed pairing against v2 `ring_ce`; no position may be dropped because its result is weak.
- Interpret the C4 check as discrete raw-channel cyclic symmetry, not continuous SO(2) or physical rotation invariance and not a novel algorithm claim.

---

### Task 1: Shared-scale preprocessing helper

**Files:**
- Create: `src/c170_shared_scale_preprocessing_v3.py`
- Test: `tests/test_c170_shared_scale_preprocessing_v3.py`

**Interfaces:**
- Consumes: `run_c170_experiment.filter_windows(windows, sample_rate=250)` for the unchanged order-3 bandpass.
- Produces: `calibrate_windows_shared_scale_v3(samples, scale=None) -> (normalized_float32, scalar_float)` and `prepare_model_inputs_shared_scale_v3(windows, splits) -> (processed_splits, scalar_float)`.

- [ ] **Step 1: Write failing tests** for hand-computed global median scale, train-only scale fitting, shared scale application to validation, zero-signal epsilon floor, and cyclic-roll correspondence after real filtering/demeaning.
- [ ] **Step 2: Run the focused tests** and confirm they fail because the v3 helper is absent.
- [ ] **Step 3: Implement the helper** using only the frozen filter, per-window temporal mean subtraction, and one global train median absolute value.
- [ ] **Step 4: Run the focused tests** and confirm all expected values and rotations match.

### Task 2: Independent v3 runner and safe selected-weight artifact

**Files:**
- Create: `run_c170_ring_cuda_experiment_v3.py`
- Test: `tests/test_c170_ring_cuda_experiment_v3.py`

**Interfaces:**
- Consumes: Task 1 preprocessing helper; existing `RingRepresentationClassifier`; frozen v2 sampler, weights, CE objective, evaluation, metrics, and prediction helpers.
- Produces: CLI `--output-dir PATH --variant {shared_scale_ce,shared_scale_shuffle} --seed {17,29,43,61,89} --max-epochs 160 --patience 25 --cpu-threads 2 --device {cuda,cpu} --dataset PATH --confirm-physical-order`.
- Produces per-run `run_manifest.json`, `RUN_COMPLETE.json`, `best_checkpoint.pt`, `best_weights.npz`, `learning_curve.csv`, and `validation_predictions.csv`; the completion marker contains hashes of the manifest and all four artifacts.

- [ ] **Step 1: Write failing contract tests** for the 10 permitted variant/seed combinations, CE-only model mapping, exact frozen config, and numeric-only NPZ selected-state save/load with pickle disabled.
- [ ] **Step 2: Run the focused tests** and confirm they fail because the v3 runner contract is absent.
- [ ] **Step 3: Implement the standalone runner** with the frozen 879-sample sampler and position weights, v3 shared-scale preprocessing, true/shuffled train targets, CE-only training, early stopping, train/validation metrics, and explicit test-access metadata.
- [ ] **Step 4: Save and reload selected weights** with numeric `np.savez` / `np.load(allow_pickle=False)` and exact state validation; keep `best_checkpoint.pt` as metadata-rich archival output, but never load it for evaluation.
- [ ] **Step 5: Run focused tests and `py_compile`**; then run a deterministic synthetic CUDA check from raw windows through filter/demean/shared-scale and all four channel rolls, asserting preprocessing and ring logits/projections agree within the documented tolerances.
- [ ] **Step 6: Verify the validation report contract** covers positions 5/40/90/130 mm, all four recalls, paired per-seed comparisons with v2 `ring_ce`, the validation gate, and `test_metrics_computed=false`.

### Task 3: Record protocol and handoff contract

**Files:**
- Create: `docs/designs/2026-09-29-c170-shared-scale-c4-equivariance.md`
- Create: `docs/plans/2026-09-29-c170-shared-scale-v3.md`

- [ ] **Step 1: Verify the documents** state the exact train-only scalar equation, unchanged split/training protocol, two variants and five seeds, safe tensor loading rule, and sealed-test boundary.
- [ ] **Step 2: Send the launcher owner** the frozen runner path, CLI, output/artifact names, manifest/completion fields, and source hashes after tests and CUDA smoke pass.
