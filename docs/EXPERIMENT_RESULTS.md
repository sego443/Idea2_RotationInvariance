# Experimental results and evidence report

Historical report date: 2026-10-01; fresh-run addendum: 2026-10-12 (section 8). Sections 1–7 consolidate acquisition/training from 2026-08-27 to 2026-09-30; training was not rerun and no new test-waveform scoring was performed during that documentation release. The copied historical JSON/CSV summaries have checksums in [the evidence index](evidence/artifact_index.json).

## Overall finding

The project has not met its rotation-generalization gate or the later 0.90 macro-F1 target. W01's validation-selected model failed on unseen nominal angles. Frozen C170 F2 passed its reused validation threshold but failed the registered test. Later sensor calibration, normalization, multiscale/Nyquist, regularization, OOF fusion, and temporal Gram experiments did not establish a reliable improvement or independently verified physical rotation invariance.

## 1. Chronology and evaluation status

| Period | Executed work | Evidence status |
|---|---|---|
| Aug27 | W01 raw/Orbit/harmonic held-angle proof of concept and rest/null controls | Single participant/session/wear; test opened; gate failed |
| Sep25–26 | Independent-LSTM graph fusion and conditional angle/movement experiments | W01 test reused; exploratory |
| Sep26 | C170 preparation: 80 recordings/1,600 windows and metadata | 1,599 original windows plus one marked copy |
| Sep27 | Raw/Orbit baseline, harmonic/probability-Orbit candidates and ensembles | Fixed position holdout; validation gates failed |
| Sep28 | Latent-angle and temporal ring CUDA experiments, protocol/quality diagnostics | Train/validation studies; mathematical and data-access boundaries documented |
| Sep29 | Shared-scale, phase/Nyquist/temporal, KRR/template/complex families and frozen F2/null evaluation | Reused validation development; F2 registered test opened once |
| Sep30 | v13/v14/v16/v17/v18/v19 extensions and closeout | Old-validation exploration; no later test scoring recorded |
| Oct01 | English conversion, documentation, evidence packaging and independent Git setup | Source/text/data-integrity verification; no new acquisition/training |

v15 phase-kernel and other intermediate outputs exist locally, but this report does not invent a final score for a batch without extracting its saved reporting basis. v20 remained a proposal, without implementation or launch in the September30 closeout. W02 and irregular-angle challenges were not collected in this record.

## 2. W01 and exploratory re-evaluations

W01 uses240 movements at12 nominal angles, four directions, five repetitions, seeds17/29/43/61/89. Train/validation/test counts120/60/60. See [dated full report](results/2026-08-27-round1-rotation-invariance-feasibility.md) and [saved metrics](evidence/round1_rotation_poc_2026-08-27/metrics.json).

| Model | Train F1 | Validation F1 | Held-angle test F1 | Test accuracy | Minimum mean class recall |
|---|---:|---:|---:|---:|---:|
| Raw CNN | .985±.030 | .437±.062 | .495±.106 | .523 | .240 |
| Cyclic-orbit CNN | .940±.071 | .441±.039 | .670±.026 | .683 | .427 |
| Harmonic CNN | .993±.015 | .531±.033 | .478±.040 | .493 | .213 |

Validation selects harmonic; orbit's higher test score is post hoc. Null test F1 .229±.071/accuracy .247; rest F1 .182 raw/.238 harmonic. Start/end rest RMS rose about19–51 fold. Thirty-one of48 movement files were too short under the collection protocol; complete wear metadata is missing and acquisition order confounds time.

The [LSTM/graph re-evaluation](evidence/lstm_graph_round1_2026-09-25/comparison_report.md) selected hidden16/layer1/two-head ring graph across12 configurations. Test F1 .339±.034 versus mean-fusion .392±.098, 125-point matched CNN .497±.062 and original orbit .670±.026. Graph UP recall .160; null/rest F1 .198/.171. No improvement or gate pass.

The [conditional experiment](evidence/conditional_directions_2026-09-26/report.md) reports CNN independent angle MAE58.8°±11.1°, independent movement F1 .379±.076, joint MAE70.4°±15.9°/F1 .456±.098. Graph counterparts: angle77.3°±8.5°, movement .306±.030, joint79.5°±7.8°/.267±.083. The no-waveform condition-only angle baseline is90°. Adding90° to supplied angles leaves95.7%–99.3% of movement predictions unchanged; these scores do not demonstrate effective angle-conditioned recognition. All are exploratory after prior W01 test exposure.

## 3. Early C170 baseline and quality findings

The fixed split has879 train,320 old-validation and320 registered-test windows, excluding70 mm and copied B019. Baseline mean validation F1: raw .6039±.0368, orbit .6718±.0425; shuffled raw .2382±.0661, orbit .2137±.0834. Harmonic candidate .663354±.039916; probability-Orbit .688593±.051461, best seed17 .77501. Five-seed probability averaging .670852; fifteen-member averaging .664871. All failed the original validation gate; see the [dated acceptance record](results/2026-09-27-c170-exploratory-validation-acceptance.md).

The [train/validation quality diagnostic](results/2026-09-28-c170-train-validation-quality-diagnostic.md) found no nearly flat processed sensor-windows under RMS/peak-to-peak<=1e-6, one train/two validation crest>10 candidates, and no identical whole processed windows after copy exclusion. These dimensionless filtered statistics do not establish raw ADC saturation or hardware causality. Map, position range and acquisition time are confounded: validation5/40 mm uses0,2,4,6;90/130 uses0,1,2,3. The fixed4,500,000 boundary audit records27 affected windows (24/1/2 train/validation/test), without authorizing score-driven exclusions.

## 4. Frozen F2 registered test

F2 uses hierarchical hard votes from complex_gram5, learned_power10 and v7 group_true, with member/family ties to lowest class ID. The rule revision corrected an earlier prose description and reproduced all320 validation rows. All15 neural member validation outputs agree with their saved CSVs. The true-label validation macro-F1 is0.826871507; matched-null validation replicate scores range0.065923–0.205869, below the recorded .5 null gate.

The final test contains320 windows at15/25/110/150 mm,80 per class and position. No fitting, scale fitting, checkpoint selection, threshold changes or fusion changes were permitted at this test stage. Source: [final acceptance](evidence/c170_f2_matched_nulls_20260929/frozen_f2_final_acceptance.json) and [saved predictions](evidence/c170_f2_matched_nulls_20260929/frozen_f2_and_null_test_predictions.csv).

Accuracy **0.678125**; macro-F1 **0.674897**. Test-minus-validation F1 is -0.151974. The joint acceptance flag is **false**.

| Class | Recall | Recall >=.65 |
|---|---:|---|
| UP | 0.6750 | Yes |
| DOWN | 0.4750 | No |
| LEFT | 0.7875 | Yes |
| RIGHT | 0.7750 | Yes |

Confusion matrix: true classes are rows, predicted classes columns, order UP/DOWN/LEFT/RIGHT.

| True / predicted | UP | DOWN | LEFT | RIGHT |
|---|---:|---:|---:|---:|
| UP | 54 | 6 | 4 | 16 |
| DOWN | 36 | 38 | 0 | 6 |
| LEFT | 1 | 0 | 63 | 16 |
| RIGHT | 0 | 9 | 9 | 62 |

| Position mm | N | Accuracy | Macro-F1 | Predicted classes |
|---|---:|---:|---:|---:|
| 15 | 80 | 0.887500 | 0.886597 | 4 |
| 25 | 80 | 0.725000 | 0.732269 | 4 |
| 110 | 80 | 0.512500 | 0.469841 | 4 |
| 150 | 80 | 0.587500 | 0.518848 | 4 |

Matched-null test macro-F1 by replicate:

| Replicate | Macro-F1 |
|---|---:|
| null_rep_17 | 0.065844 |
| null_rep_29 | 0.118428 |
| null_rep_43 | 0.142697 |
| null_rep_61 | 0.169321 |
| null_rep_89 | 0.068759 |

All four classes are predicted, but DOWN recall .475 and overall F1 below.80 fail acceptance. Strong15 mm performance does not override poor110/150 mm transfer. These repetitions do not establish independent wear robustness. The test is now exposed; later alternatives require a new holdout.

## 5. Later exploratory batches

The table compares saved reporting bases, rather than pretending seed means and fusion scores are interchangeable. “Best listed” means a maximum in these old-validation experiments, not a selected independent test performance.

| Batch | Aggregation and reporting basis | Old-validation macro-F1 | Registered test | Evidence |
|---|---|---:|---|---|
| Frozen F2 | Hierarchical three-family hard vote | 0.826872 | 0.674897 | [JSON](evidence/c170_f2_matched_nulls_20260929/frozen_f2_final_acceptance.json) |
| v13, 18 runs | Best listed fixed fusion: shared FIR | 0.826961 | Not evaluated | [JSON](evidence/c170_sensor_adapter_v13_formal_20260930/sensor_adapter_v13_validation_summary.json) |
| v14, 24 runs | Best listed fixed fusion: bounded gain | 0.820132 | Not evaluated | [JSON](evidence/c170_sensor_normalization_v14_formal_20260930/sensor_normalization_v14_validation_summary.json) |
| v16, 18 runs | Best listed fixed fusion: Nyquist | 0.826959 | Not evaluated | [JSON](evidence/c170_multiscale_nyquist_v16_formal_20260930/sensor_normalization_v16_validation_summary.json) |
| v17, 36 runs | Best listed fixed fusion: wd .1 / drop .2 | 0.817095 | Not evaluated | [JSON](evidence/c170_complex_regularization_v17_recovery2_20260930/c170_complex_regularization_v17_validation_summary.json) |
| v18, 66 neural + 11 KRR folds | Four-expert equal soft mean | 0.715902 | Not evaluated | [JSON](evidence/c170_oof_fusion_v18_formal_20260930/v18_fusion_summary.json) |
| v19, 36 runs | Best listed fixed fusion: hybrid L64 | 0.776975 | Not evaluated | [JSON](evidence/c170_temporal_gram_v19_formal_20260930/v19_oldval_summary.json) |

### v13 sensor adapters

| Adapter | Seed-mean F1 | Seed population SD | Family-vote F1 | Fixed-fusion F1 | Shuffle seed17 F1 |
|---|---:|---:|---:|---:|---:|
| per_sensor_gain | 0.702516 | 0.028815 | 0.695097 | 0.798560 | 0.111772 |
| per_sensor_fir_gain | 0.713625 | 0.023112 | 0.734087 | 0.814201 | 0.107756 |
| shared_fir | 0.729157 | 0.012905 | 0.746481 | 0.826961 | 0.092924 |

Three candidates×(five true seeds+one seed17 null)=18 runs. Shared FIR exceeds F2 by only 0.000090 F1 on old validation; this is not a substantive verified improvement.

### v14 normalization

| Candidate | Seed-mean F1 | Family-vote F1 | Fixed-fusion F1 | Shuffle seed17 F1 |
|---|---:|---:|---:|---:|
| bounded_sensor_gain | 0.732737 | 0.736561 | 0.820132 | 0.098231 |
| sensor_median | 0.684043 | 0.683267 | 0.785327 | 0.126286 |
| sensor_rms | 0.580417 | 0.600546 | 0.756050 | 0.097771 |
| window_sensor_rms | 0.527054 | 0.544923 | 0.804487 | 0.211421 |

Four×six=24 runs. Window-wise RMS family performance is particularly weak despite a higher fusion score supplied by the other frozen families; do not attribute the fused score solely to that candidate.

### v16 multiscale/Nyquist

| Candidate | Seed-mean F1 | Family-vote F1 | Fixed-fusion F1 | Shuffle seed17 F1 |
|---|---:|---:|---:|---:|
| multiscale | 0.723936 | 0.738141 | 0.819521 | 0.130060 |
| multiscale_nyquist | 0.729656 | 0.755838 | 0.819856 | 0.110551 |
| nyquist | 0.715424 | 0.744301 | 0.826959 | 0.097367 |

Three×six=18 runs. The Nyquist fusion is numerically close to F2, without new independent evidence.

### v17 regularization

| Weight-decay/dropout configuration | Seed-mean F1 | Family-vote F1 | Fixed-fusion F1 | Shuffle seed17 F1 |
|---|---:|---:|---:|---:|
| wd-0p01__drop-0p2 | 0.711389 | 0.725149 | 0.811892 | 0.099770 |
| wd-0p01__drop-0p4 | 0.669497 | 0.694121 | 0.793970 | 0.103662 |
| wd-0p01__drop-0p6 | 0.685415 | 0.698426 | 0.797805 | 0.146732 |
| wd-0p1__drop-0p2 | 0.715808 | 0.733001 | 0.817095 | 0.099770 |
| wd-0p1__drop-0p4 | 0.677395 | 0.699606 | 0.801044 | 0.091515 |
| wd-0p1__drop-0p6 | 0.659125 | 0.677203 | 0.797229 | 0.145089 |

Six×six=36 runs across original/recovery roots, with valid completed/reused records summarized. No configuration improves the frozen baseline reliably.

### v18 training-position OOF fusion

| Expert / fusion | OOF-training F1 | Old-validation F1 |
|---|---:|---:|
| complex_gram | 0.625404 | 0.718238 |
| learned_power | 0.640358 | 0.705722 |
| window_sensor_rms_complex_gram | 0.452792 | 0.508139 |
| v7_group_krr | 0.634823 | 0.689462 |
| equal_soft_mean_3_experts | 0.672565 | 0.705156 |
| linear_stack | 0.795983 | 0.553303 |
| linear_gate | 0.701357 | 0.709640 |
| equal_soft_mean_4_experts | 0.691570 | 0.715902 |
| linear_stack_4_experts | 0.830338 | 0.589990 |
| linear_gate_4_experts | 0.752663 | 0.679888 |

Eleven training-position folds, seeds17/29 and three neural experts yield66 runs plus11 KRR folds. Fold models use fixed120-epoch endpoints; fusion uses200 steps, lr.01, L2 mean-weight-square.01. Protocol flags say old validation was not used for fusion training/selection and registered test was not touched. Four-expert equal averaging is the highest listed old-validation fusion (.715902), below F2; this retrospectively observed maximum is still exploratory. A corrupt fold NPZ was preserved and rerun individually.

### v19 temporal Gram

| Representation | Five-seed mean F1 | Family-vote F1 | Fixed-fusion F1 | Shuffle seed17 F1 |
|---|---:|---:|---:|---:|
| gz_L64 | 0.605851 | 0.640149 | 0.758007 | 0.185012 |
| gz_L128 | 0.608711 | 0.653549 | 0.748832 | 0.178881 |
| gzn_L64 | 0.642860 | 0.676228 | 0.774673 | 0.177139 |
| gzn_L128 | 0.654768 | 0.680715 | 0.762217 | 0.150448 |
| hybrid_L64 | 0.643073 | 0.675332 | 0.776975 | 0.108421 |
| hybrid_L128 | 0.639617 | 0.670551 | 0.771203 | 0.131097 |

Six representations×six=36 runs. Hybrid/L64 is the best listed replacement fusion but scores .776975, below F2. No later batch in this table has a registered-test result.

## 6. Runtime, failures, and limits

Historical environment: Python3.10/PyTorch1.13.1+cu117/RTX2080 Ti. Later scheduling commonly used <=6 workers and one CPU thread, with used-RAM+2 GiB <=24 GiB admission. Historical monitoring reached98% GPU utilization and sampled about21–25 GiB host RAM. Windows process/session interruptions, cuda versus cuda:0 display inconsistencies and the damaged v18 NPZ required explicit recovery. The September30 closeout recorded no remaining matching workers/supervisors for these tasks; this is a dated execution record, not a fresh process-monitoring claim.

Shared scalar normalization fixes an algebraic scale/permutation commutation gap; synthetic invariance checks establish only their stipulated tensor action. Unequal sensor/contact responses and physical repositioning remain unresolved. Neither normalized waveform distribution shifts nor attention weights identify a hardware or physiological cause.

Old validation has been used repeatedly across many candidates. Seed variability is not a confidence interval for new people/wears. Only one registered F2 test is reported; later extensions did not preprocess/predict/score it according to their saved protocol/closeout records. These records are not a new waveform-access audit.

No0.90 target was reached. The next confirmatory evidence needs a new independent, balanced-order, measured-angle rewear acquisition after inference freeze. See [design](EXPERIMENT_DESIGN.md), [procedure](EXPERIMENT_PROCEDURE.md), and [reproduction](REPRODUCIBILITY.md). No proposed W02/v20 outcome is fabricated.

## 7. Figures and documentation verification

See the [English figure gallery](FIGURES.md). Five existing overlays were regenerated from the same selected train/validation/excluded-position windows, with traces exactly matching the original calculation and no registered-test preprocessing. The original English W01 result figure is unchanged. [Release verification](evidence/release_verification.json) records current text, source, metadata, data-hash, saved-prediction arithmetic and Git checks; it is distinct from a training rerun.

## 8. Fresh v22 exploratory ablations (2026-10-12)

The [dated v22 report](results/2026-10-12-v22-window-normalization-ablation.md) records 81 completed CUDA runs: time-window and amplitude-normalization ablations, three real-label seeds per variant/protocol and 15 matched block-permuted controls. It explicitly scores the already exposed C170 held-out positions and reused W01 data after fixed 120-epoch training; these are exploratory results, not new confirmatory evidence. The original v22 group_sensor design uses an authorized transductive evaluation input pool.

W01 ZERO macro-F1 remains weak (fresh full/shared mean 0.233; best exploratory mean 0.337 for prefix_200/shared). Full/group_sensor CAL reaches a 0.944 mean, but its matched training-label null also scores about 0.944 with true calibration support. High same-recording calibration therefore does not establish correctly learned cross-session direction semantics or physical rotation invariance. See the dated report for per-class, per-position, channel-map and access-boundary detail; no W02 data was collected or evaluated.
