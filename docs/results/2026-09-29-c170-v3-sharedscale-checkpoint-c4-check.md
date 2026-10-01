# C170 v3 shared-scale checkpoint C4 synthetic check

Date: 2026-09-29. This is a label-free synthetic-input structure check, not a real-signal performance result. No dataset or test waveform was opened, and no training or evaluation on real data was run.

## Procedure

- Waited for `outputs/c170_ring_v3_formal_cuda_20260929/shared_scale_ce__seed-17/RUN_COMPLETE.json`; it marked the run complete and `test_metrics_computed=false`.
- Read only the run identity and fitted scalar from metadata: variant `shared_scale_ce`, seed 17, shared scale `11123.037109375`.
- Loaded only `best_weights.npz` with the runner's `_load_numpy_state_dict` helper (`np.load(..., allow_pickle=False)`, exact keys/shapes/dtypes, strict `load_state_dict`). Did not open `best_checkpoint.pt`.
- Used one synthetic float32 batch, shape `(2,500,4)`, RNG seed 20260929. Constructed raw-like channel differences with gains `[0.25, 0.8, 2.0, 6.0]` and offsets `[3.0, -5.0, 0.5, 9.0]`, scaled by the run scalar. For each cyclic shift 0–3, applied the existing 250 Hz `filter_windows`, then per-window/channel demeaning and the fixed run scalar via `calibrate_windows_shared_scale_v3(..., scale=11123.037109375)`. No scale was fitted from the synthetic batch.
- Ran inference on CUDA `cuda:0`, PyTorch intra-op threads 2, with the loaded seed17 model in eval mode.

## Result

Across shifts 0–3, preprocessed tensors matched the corresponding roll of the unshifted tensor exactly (maximum absolute preprocessing error `0`). Relative to shift 0, maximum absolute logits error was `2.38418579e-7`; maximum absolute projection error was `9.31322575e-8`.

This confirms the discrete C4 correspondence for this completed shared-scale checkpoint on the specified synthetic batch and frozen scalar. It does not establish continuous physical rotation invariance or real-data generalization.
