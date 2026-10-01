# Direction-classification training closeout, 2026-09-30

Historical closeout translated/consolidated on2026-10-01; no training rerun.

## Goal, data and training

The goal was0.90 direction-classification macro-F1 while investigating four-channel piezoelectric rotation invariance. Frozen counts879 train/320 old-validation/320 registered-test; exclude70 mm and B019 copy, preserve S0–S3 and raw data. Common late-stage input uses0.5–40 Hz and a train-fitted shared scalar. Baseline/most neural batches inherit v11 AdamW lr/weight_decay .001, dropout.2, max160/patience25. v18 LOPO folds use fixed120-epoch end checkpoints. Classes/fusion rules were fixed before each batch.

## Main results

| Batch | Aggregation and reporting basis | Old-validation macro-F1 | Registered test | Evidence |
|---|---|---:|---|---|
| Frozen F2 | Hierarchical three-family hard vote | 0.826872 | 0.674897 | [JSON](evidence/c170_f2_matched_nulls_20260929/frozen_f2_final_acceptance.json) |
| v13, 18 runs | Best listed fixed fusion: shared FIR | 0.826961 | Not evaluated | [JSON](evidence/c170_sensor_adapter_v13_formal_20260930/sensor_adapter_v13_validation_summary.json) |
| v14, 24 runs | Best listed fixed fusion: bounded gain | 0.820132 | Not evaluated | [JSON](evidence/c170_sensor_normalization_v14_formal_20260930/sensor_normalization_v14_validation_summary.json) |
| v16, 18 runs | Best listed fixed fusion: Nyquist | 0.826959 | Not evaluated | [JSON](evidence/c170_multiscale_nyquist_v16_formal_20260930/sensor_normalization_v16_validation_summary.json) |
| v17, 36 runs | Best listed fixed fusion: wd .1 / drop .2 | 0.817095 | Not evaluated | [JSON](evidence/c170_complex_regularization_v17_recovery2_20260930/c170_complex_regularization_v17_validation_summary.json) |
| v18, 66 neural + 11 KRR folds | Four-expert equal soft mean | 0.715902 | Not evaluated | [JSON](evidence/c170_oof_fusion_v18_formal_20260930/v18_fusion_summary.json) |
| v19, 36 runs | Best listed fixed fusion: hybrid L64 | 0.776975 | Not evaluated | [JSON](evidence/c170_temporal_gram_v19_formal_20260930/v19_oldval_summary.json) |

F2 old-validation F1 .826872 versus registered test .674897. Recalls UP/DOWN/LEFT/RIGHT .675/.475/.7875/.775 fail the .80 F1/.65 recall gate. Shared FIR's improvement over old validation is about.000090, not a substantive result. v13–v19 repeatedly use the same old validation and are exploratory; their scores are not test scores.

## Conclusion and execution record

The0.90 target was not reached. Sensor gain/normalization/Nyquist/temporal Gram did not reliably outperform F2. Mathematical C4/ideal phase invariance does not establish physical wearing robustness. No new acquisition was arranged. The test was used for frozen F2 acceptance only; later batches recorded no test preprocessing/prediction/scoring.

Python3.10/PyTorch1.13.1+CUDA11.7/RTX2080 Ti; commonly <=6 workers×one CPU thread, RAM admission used+2 GiB<=24 GiB. Historical GPU utilization reached98%; host RAM samples about21–25 GiB. Windows session/process interruptions, cuda/cuda:0 differences and one corrupt v18 NPZ were recorded; retain the corrupt original and rerun only that fold. Summaries use valid artifacts. Closeout recorded zero matching workers/supervisors and v18/v19/v20 tasks; v18/v19 finished, v20 remained design-only, without implementation/launch. Raw data and incomplete outputs were retained.

See [the detailed report](EXPERIMENT_RESULTS.md) for per-class/position F2 results, all listed ablation tables, null controls, protocol and limitations.
