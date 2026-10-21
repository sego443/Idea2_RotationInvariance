# Portable experiment evidence

The table below contains historical artifacts copied on 2026-10-01, not fresh training results. [artifact_index.json](artifact_index.json) records their original output paths, SHA-256 and sizes. [english_conversion_manifest.json](english_conversion_manifest.json) records text migration; original run hashes remain historical. [release_verification.json](release_verification.json) records that release's checks.

Fresh October 12 v22 evidence lives separately in [v22_ablation_20261012](v22_ablation_20261012/summary.json): 81 new completed runs, [preflight](v22_ablation_20261012/preflight.json), [score table](v22_ablation_20261012/score_table.md), [verified run records](v22_ablation_20261012/verified_run_evidence.json) and [verification](v22_ablation_20261012/verification.json). See the [dated report](../results/2026-10-12-v22-window-normalization-ablation.md) for interpretation and test-access boundaries.

Some copied JSON files refer to additional checkpoints/run directories not included here; obtain the complete local artifact bundle for replay. Machine paths are provenance.

| Artifact | Original local source |
|---|---|
| [c170_complex_regularization_v17_validation_summary.json](c170_complex_regularization_v17_recovery2_20260930/c170_complex_regularization_v17_validation_summary.json) | `outputs/c170_complex_regularization_v17_recovery2_20260930/c170_complex_regularization_v17_validation_summary.json` |
| [config.json](c170_distance_holdout_20260927/config.json) | `outputs/c170_distance_holdout_20260927/config.json` |
| [validation_metrics.json](c170_distance_holdout_20260927/validation_metrics.json) | `outputs/c170_distance_holdout_20260927/validation_metrics.json` |
| [config.json](c170_distance_holdout_candidates_20260927/config.json) | `outputs/c170_distance_holdout_candidates_20260927/config.json` |
| [validation_metrics.json](c170_distance_holdout_candidates_20260927/validation_metrics.json) | `outputs/c170_distance_holdout_candidates_20260927/validation_metrics.json` |
| [validation_ensemble_metrics.json](c170_distance_holdout_ensemble_verified_20260927/validation_ensemble_metrics.json) | `outputs/c170_distance_holdout_ensemble_verified_20260927/validation_ensemble_metrics.json` |
| [NULL_RUNS_COMPLETE.json](c170_f2_matched_nulls_20260929/NULL_RUNS_COMPLETE.json) | `outputs/c170_f2_matched_nulls_20260929/NULL_RUNS_COMPLETE.json` |
| [driver_state.json](c170_f2_matched_nulls_20260929/driver_state.json) | `outputs/c170_f2_matched_nulls_20260929/driver_state.json` |
| [experiment_manifest.json](c170_f2_matched_nulls_20260929/experiment_manifest.json) | `outputs/c170_f2_matched_nulls_20260929/experiment_manifest.json` |
| [frozen_f2_and_null_test_predictions.csv](c170_f2_matched_nulls_20260929/frozen_f2_and_null_test_predictions.csv) | `outputs/c170_f2_matched_nulls_20260929/frozen_f2_and_null_test_predictions.csv` |
| [frozen_f2_final_acceptance.json](c170_f2_matched_nulls_20260929/frozen_f2_final_acceptance.json) | `outputs/c170_f2_matched_nulls_20260929/frozen_f2_final_acceptance.json` |
| [frozen_f2_final_acceptance.md](c170_f2_matched_nulls_20260929/frozen_f2_final_acceptance.md) | `outputs/c170_f2_matched_nulls_20260929/frozen_f2_final_acceptance.md` |
| [frozen_f2_final_test_metrics.json](c170_f2_matched_nulls_20260929/frozen_f2_final_test_metrics.json) | `outputs/c170_f2_matched_nulls_20260929/frozen_f2_final_test_metrics.json` |
| [frozen_f2_rule_revision.json](c170_f2_matched_nulls_20260929/frozen_f2_rule_revision.json) | `outputs/c170_f2_matched_nulls_20260929/frozen_f2_rule_revision.json` |
| [frozen_f2_validation_replay.json](c170_f2_matched_nulls_20260929/frozen_f2_validation_replay.json) | `outputs/c170_f2_matched_nulls_20260929/frozen_f2_validation_replay.json` |
| [matched_null_mapping.json](c170_f2_matched_nulls_20260929/matched_null_mapping.json) | `outputs/c170_f2_matched_nulls_20260929/matched_null_mapping.json` |
| [matched_null_validation_metrics.json](c170_f2_matched_nulls_20260929/matched_null_validation_metrics.json) | `outputs/c170_f2_matched_nulls_20260929/matched_null_validation_metrics.json` |
| [matched_null_validation_metrics_frozen_f2.json](c170_f2_matched_nulls_20260929/matched_null_validation_metrics_frozen_f2.json) | `outputs/c170_f2_matched_nulls_20260929/matched_null_validation_metrics_frozen_f2.json` |
| [validation_summary.md](c170_harmonic_nyquist_v5_validation_summary_20260929/validation_summary.md) | `outputs/c170_harmonic_nyquist_v5_validation_summary_20260929/validation_summary.md` |
| [validation_summary.md](c170_harmonic_phase_v4_validation_summary_20260929/validation_summary.md) | `outputs/c170_harmonic_phase_v4_validation_summary_20260929/validation_summary.md` |
| [validation_summary.md](c170_latent_angle_formal_cuda_20260928/validation_analysis_20260928/validation_summary.md) | `outputs/c170_latent_angle_formal_cuda_20260928/validation_analysis_20260928/validation_summary.md` |
| [sensor_normalization_v16_validation_summary.json](c170_multiscale_nyquist_v16_formal_20260930/sensor_normalization_v16_validation_summary.json) | `outputs/c170_multiscale_nyquist_v16_formal_20260930/sensor_normalization_v16_validation_summary.json` |
| [v18_fusion_summary.json](c170_oof_fusion_v18_formal_20260930/v18_fusion_summary.json) | `outputs/c170_oof_fusion_v18_formal_20260930/v18_fusion_summary.json` |
| [validation_summary.md](c170_ring_v2_validation_summary_recovery1_20260928/validation_summary.md) | `outputs/c170_ring_v2_validation_summary_recovery1_20260928/validation_summary.md` |
| [validation_summary.md](c170_ring_v3_validation_summary_20260929_retry1/validation_summary.md) | `outputs/c170_ring_v3_validation_summary_20260929_retry1/validation_summary.md` |
| [sensor_adapter_v13_validation_summary.json](c170_sensor_adapter_v13_formal_20260930/sensor_adapter_v13_validation_summary.json) | `outputs/c170_sensor_adapter_v13_formal_20260930/sensor_adapter_v13_validation_summary.json` |
| [sensor_normalization_v14_validation_summary.json](c170_sensor_normalization_v14_formal_20260930/sensor_normalization_v14_validation_summary.json) | `outputs/c170_sensor_normalization_v14_formal_20260930/sensor_normalization_v14_validation_summary.json` |
| [v19_oldval_summary.json](c170_temporal_gram_v19_formal_20260930/v19_oldval_summary.json) | `outputs/c170_temporal_gram_v19_formal_20260930/v19_oldval_summary.json` |
| [validation_summary.md](c170_temporal_phase_invariant_v6_validation_summary_20260929/validation_summary.md) | `outputs/c170_temporal_phase_invariant_v6_validation_summary_20260929/validation_summary.md` |
| [core_summary.md](c170_trainval_spectral_diagnostic_20260929/core_summary.md) | `outputs/c170_trainval_spectral_diagnostic_20260929/core_summary.md` |
| [summary.md](c170_trainval_spectral_diagnostic_20260929/summary.md) | `outputs/c170_trainval_spectral_diagnostic_20260929/summary.md` |
| [fusion_validation_summary.md](c170_v4_v6_power_fusion_validation_20260929/fusion_validation_summary.md) | `outputs/c170_v4_v6_power_fusion_validation_20260929/fusion_validation_summary.md` |
| [summary.md](c170_v7_v4_v6_fixed_vote_validation_20260929/summary.md) | `outputs/c170_v7_v4_v6_fixed_vote_validation_20260929/summary.md` |
| [summary.md](c170_v7_v4_v6_soft_probability_validation_20260929/summary.md) | `outputs/c170_v7_v4_v6_soft_probability_validation_20260929/summary.md` |
| [metrics.json](conditional_directions_2026-09-26/metrics.json) | `outputs/conditional_directions_2026-09-26/metrics.json` |
| [report.md](conditional_directions_2026-09-26/report.md) | `outputs/conditional_directions_2026-09-26/report.md` |
| [comparison_report.md](lstm_graph_round1_2026-09-25/comparison_report.md) | `outputs/lstm_graph_round1_2026-09-25/comparison_report.md` |
| [metrics.json](lstm_graph_round1_2026-09-25/metrics.json) | `outputs/lstm_graph_round1_2026-09-25/metrics.json` |
| [experiment_report.md](round1_rotation_poc_2026-08-27/experiment_report.md) | `outputs/round1_rotation_poc_2026-08-27/experiment_report.md` |
| [metrics.json](round1_rotation_poc_2026-08-27/metrics.json) | `outputs/round1_rotation_poc_2026-08-27/metrics.json` |
