# C170 ring v2 runner protocol review

Date: 2026-09-28
Scope: read-only review of `run_c170_ring_cuda_experiment_v2.py`, `launch_c170_ring_cuda_v2.py`, their contract tests, and the ring model/preprocessing helpers. No source was edited; no test suite, training run, or test scoring was launched.

## Verdict

No protocol blocker was found for the formal 20-run matrix under the parent-approved sealed-test boundary. This is a source review, not a claim that the formal runs have executed or that CUDA runtime behavior has been re-verified.

The boundary is **not zero waveform reads**: the legacy loader materializes the compressed NPZ arrays and checks finiteness over the complete `X_raw`. The parent explicitly accepted full NPZ decoding and this all-array finite-value integrity check. After that check, model preprocessing receives only train and validation indices; test waveforms are not preprocessed, used to fit a scale, provided to the model, predicted, scored, or used for checkpoint selection.

## Protocol checks

| Requirement | Source evidence |
|---|---|
| Four arms × five frozen seeds | Launcher formal suite builds `ring_main`, `ring_ce`, `slot_supcon`, `ring_shuffle` crossed with seeds 17, 29, 43, 61, 89. |
| Epoch and early-stop limits | Formal specs use 160 maximum epochs and patience 25. Runner defaults agree; launcher passes these values explicitly. |
| Ring main and CE comparison | `ring_main` selects `main` with SupCon; `ring_ce` selects `classification_only` without SupCon. In the current classifier, both use the same ring readout; only `slot_readout` takes the separate fixed-slot path. |
| Slot and shuffle controls | `slot_supcon` selects `slot_readout` with SupCon. `ring_shuffle` selects the main ring path and shuffles direction labels by complete training recording blocks, with no validation-label modification. |
| Sampler | Uses original position × direction groups, without replacement: 19 batches of 44 and one of 43, covering all 879 train windows once per epoch. |
| Objective | CE has label smoothing 0. Each batch uses `20 * sum(sample_weight * per_sample_CE)`; SupCon variants add fixed `.1 * scalar_batch_mean_SupCon`, temperature `.1`. Position weights are derived from the fixed train positions and normalized; the helper averages positive log-probability per anchor and then across anchors. |
| CUDA | CLI accepts CUDA/CPU, defaults to CUDA; launcher passes CUDA. Runner checks availability, moves the model, input/target/weight tensors and batch metadata to `cuda:0`, and uses CUDA synchronization for timing. |
| Selection and test use | Each run selects its own checkpoint by validation macro-F1, breaking ties by validation CE. Launcher aggregates throughput, not a cross-seed “best” result. Runner records `test_metrics_computed=false`; its prediction export is validation-only. |
| Split handling | Fixed train/validation/test positions and expected split counts are checked from metadata. `prepare_model_inputs` indexes only train and validation rows, fits sensor scales on train, and applies those scales to validation. |

## Test and evidence limits

The reviewed unit tests cover the variant map, no-replacement sampler, objective scaling, block-shuffle invariants, formal matrix, launcher arguments, and artifact/config hashes. They were not executed for this review. The launcher’s artifact validator checks the manifest’s `predicted_or_scored=false` assertion; it does not independently prove the data-access boundary. The boundary above is verified by the current runner/helper source paths, with the explicitly approved all-array finite check distinguished from preprocessing, fitting, prediction, scoring, or model selection on test.

Sources reviewed: `run_c170_ring_cuda_experiment_v2.py`, `launch_c170_ring_cuda_v2.py`, `src/c170_ring_representation.py`, `run_c170_experiment.py`, `tests/test_c170_ring_cuda_experiment_v2.py`, and `tests/test_launch_c170_ring_cuda_v2.py`.
