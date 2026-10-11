# Script Catalog

Run commands from the project root. Each entry point resolves project resources from its own location; relative `--dataset` and `--output-dir` arguments still follow the working directory. Import runners by their qualified module names, for example `from scripts.runners import run_c170_experiment`.

```bash
python scripts/runners/run_c170_experiment.py --help
python scripts/data/prepare_c170_session.py --help
python scripts/analysis/analyze_c170_ring_cuda_v2_results.py --help
```

Batch launchers can start training and write artifacts. Read the linked protocol before running them; importing or displaying a catalog does not authorize a new experiment. Some historical launchers require Windows and a configured CUDA environment.

## Responsibilities

| Directory | Purpose |
|---|---|
| `runners/` | Individual experiment methods and wrappers |
| `launchers/` | Fixed grids, batch scheduling, and recovery |
| `analysis/` | Saved result audits, summaries, frozen evaluation, figures, compute benchmarks |
| `data/` | Segmentation and processed dataset creation |
| `validation/` | Synthetic numerical checks |

## Method families

| Family | Versions / entry points |
|---|---|
| Early feasibility | `run_round1_experiment`, `run_lstm_graph_experiment`, `run_conditional_experiment` |
| Baselines and ensembles | `run_c170_experiment`, `run_c170_ring_experiment`, `run_c170_validation_ensemble` |
| Latent angle and ring | latent-angle, ring v2, shared-scale v3 |
| Harmonic and temporal phase | v4, Nyquist v5, temporal-phase v6, extended seeds and crop controls |
| Kernel and template | kernel ridge v7-v10, DTW templates, phase-average kernel v15 |
| Complex representations | complex-orbit v11, domain v12, adapter v13, normalization v14 |
| Later exploratory methods | multiscale v16, regularization v17, out-of-fold fusion v18, temporal Gram v19 |
| Frozen F2 | matched-null adapter/launcher and one-shot test evaluator |

Versioned files are distinct experiment methods, controls, or dependencies. Recovery files preserve distinct recovery protocols. They are retained even when their historical results were unsuccessful.

## Entry points

### Runners

- [run_c170_c4_dtw_template_experiment.py](runners/run_c170_c4_dtw_template_experiment.py)
- [run_c170_c4_kernel_ridge_v10_robust.py](runners/run_c170_c4_kernel_ridge_v10_robust.py)
- [run_c170_c4_kernel_ridge_v7.py](runners/run_c170_c4_kernel_ridge_v7.py)
- [run_c170_c4_kernel_ridge_v8.py](runners/run_c170_c4_kernel_ridge_v8.py)
- [run_c170_c4_kernel_ridge_v9.py](runners/run_c170_c4_kernel_ridge_v9.py)
- [run_c170_complex_orbit_domain_v12.py](runners/run_c170_complex_orbit_domain_v12.py)
- [run_c170_complex_orbit_experiment_v11.py](runners/run_c170_complex_orbit_experiment_v11.py)
- [run_c170_experiment.py](runners/run_c170_experiment.py)
- [run_c170_f2_matched_null_adapter.py](runners/run_c170_f2_matched_null_adapter.py)
- [run_c170_harmonic_nyquist_experiment_v5.py](runners/run_c170_harmonic_nyquist_experiment_v5.py)
- [run_c170_harmonic_phase_experiment_v4.py](runners/run_c170_harmonic_phase_experiment_v4.py)
- [run_c170_harmonic_phase_experiment_v4_augcrop.py](runners/run_c170_harmonic_phase_experiment_v4_augcrop.py)
- [run_c170_harmonic_phase_experiment_v4_seed_ext.py](runners/run_c170_harmonic_phase_experiment_v4_seed_ext.py)
- [run_c170_latent_angle_experiment.py](runners/run_c170_latent_angle_experiment.py)
- [run_c170_multiscale_nyquist_experiment_v16.py](runners/run_c170_multiscale_nyquist_experiment_v16.py)
- [run_c170_oof_fusion_v18.py](runners/run_c170_oof_fusion_v18.py)
- [run_c170_phase_average_kernel_v15.py](runners/run_c170_phase_average_kernel_v15.py)
- [run_c170_regularization_v17.py](runners/run_c170_regularization_v17.py)
- [run_c170_ring_cuda_experiment_v2.py](runners/run_c170_ring_cuda_experiment_v2.py)
- [run_c170_ring_cuda_experiment_v3.py](runners/run_c170_ring_cuda_experiment_v3.py)
- [run_c170_ring_experiment.py](runners/run_c170_ring_experiment.py)
- [run_c170_sensor_adapter_v13.py](runners/run_c170_sensor_adapter_v13.py)
- [run_c170_sensor_normalization_v14.py](runners/run_c170_sensor_normalization_v14.py)
- [run_c170_temporal_gram_v19.py](runners/run_c170_temporal_gram_v19.py)
- [run_c170_temporal_phase_invariant_experiment_v6.py](runners/run_c170_temporal_phase_invariant_experiment_v6.py)
- [run_c170_temporal_phase_invariant_experiment_v6_augcrop.py](runners/run_c170_temporal_phase_invariant_experiment_v6_augcrop.py)
- [run_c170_temporal_phase_invariant_experiment_v6_power_shuffle.py](runners/run_c170_temporal_phase_invariant_experiment_v6_power_shuffle.py)
- [run_c170_v7_oof_expert_v18.py](runners/run_c170_v7_oof_expert_v18.py)
- [run_c170_validation_ensemble.py](runners/run_c170_validation_ensemble.py)
- [run_conditional_experiment.py](runners/run_conditional_experiment.py)
- [run_lstm_graph_experiment.py](runners/run_lstm_graph_experiment.py)
- [run_round1_experiment.py](runners/run_round1_experiment.py)

### Launchers

- [launch_c170_complex_orbit_domain_v12.py](launchers/launch_c170_complex_orbit_domain_v12.py)
- [launch_c170_complex_orbit_domain_v12_recovery.py](launchers/launch_c170_complex_orbit_domain_v12_recovery.py)
- [launch_c170_complex_orbit_v11.py](launchers/launch_c170_complex_orbit_v11.py)
- [launch_c170_frozen_f2_matched_nulls.py](launchers/launch_c170_frozen_f2_matched_nulls.py)
- [launch_c170_harmonic_nyquist_v5.py](launchers/launch_c170_harmonic_nyquist_v5.py)
- [launch_c170_harmonic_phase_v4.py](launchers/launch_c170_harmonic_phase_v4.py)
- [launch_c170_latent_angle_cuda.py](launchers/launch_c170_latent_angle_cuda.py)
- [launch_c170_multiscale_nyquist_v16.py](launchers/launch_c170_multiscale_nyquist_v16.py)
- [launch_c170_oof_fusion_v18.py](launchers/launch_c170_oof_fusion_v18.py)
- [launch_c170_regularization_v17.py](launchers/launch_c170_regularization_v17.py)
- [launch_c170_regularization_v17_recovery.py](launchers/launch_c170_regularization_v17_recovery.py)
- [launch_c170_regularization_v17_recovery2.py](launchers/launch_c170_regularization_v17_recovery2.py)
- [launch_c170_ring_cuda_v2.py](launchers/launch_c170_ring_cuda_v2.py)
- [launch_c170_sensor_adapter_v13.py](launchers/launch_c170_sensor_adapter_v13.py)
- [launch_c170_sensor_normalization_v14.py](launchers/launch_c170_sensor_normalization_v14.py)
- [launch_c170_shared_scale_v3.py](launchers/launch_c170_shared_scale_v3.py)
- [launch_c170_temporal_gram_v19.py](launchers/launch_c170_temporal_gram_v19.py)
- [launch_c170_temporal_phase_invariant_v6.py](launchers/launch_c170_temporal_phase_invariant_v6.py)
- [launch_c170_v4v6_augcrop.py](launchers/launch_c170_v4v6_augcrop.py)
- [launch_c170_v4v6_power_extensions.py](launchers/launch_c170_v4v6_power_extensions.py)

### Analysis

- [analyze_c170_harmonic_nyquist_v5_results.py](analysis/analyze_c170_harmonic_nyquist_v5_results.py)
- [analyze_c170_harmonic_phase_v4_results.py](analysis/analyze_c170_harmonic_phase_v4_results.py)
- [analyze_c170_latent_angle_results.py](analysis/analyze_c170_latent_angle_results.py)
- [analyze_c170_ring_cuda_v2_results.py](analysis/analyze_c170_ring_cuda_v2_results.py)
- [analyze_c170_shared_scale_v3_results.py](analysis/analyze_c170_shared_scale_v3_results.py)
- [analyze_c170_temporal_phase_invariant_v6_results.py](analysis/analyze_c170_temporal_phase_invariant_v6_results.py)
- [analyze_c170_v4_v6_power_validation_fusion.py](analysis/analyze_c170_v4_v6_power_validation_fusion.py)
- [benchmark_c170_ring_compute.py](analysis/benchmark_c170_ring_compute.py)
- [evaluate_c170_frozen_f2_nulls_and_test.py](analysis/evaluate_c170_frozen_f2_nulls_and_test.py)
- [plot_c170_overlays.py](analysis/plot_c170_overlays.py)
- [summarize_c170_temporal_gram_v19.py](analysis/summarize_c170_temporal_gram_v19.py)

### Data

- [prepare_c170_session.py](data/prepare_c170_session.py)

### Validation

- [validate_c170_harmonic_nyquist_v5_synthetic_equivariance.py](validation/validate_c170_harmonic_nyquist_v5_synthetic_equivariance.py)
- [validate_c170_harmonic_v4_synthetic_equivariance.py](validation/validate_c170_harmonic_v4_synthetic_equivariance.py)

## Historical commands

Older dated documents and frozen manifests use root-level filenames. Use the [migration ledger](../docs/evidence/reorganization_manifest_20261011.json) to resolve their current paths. Do not change saved historical hashes to current source hashes. New source snapshots reflect the reorganized code; the v3 analyzer accepts both original and reorganized snapshot paths.
