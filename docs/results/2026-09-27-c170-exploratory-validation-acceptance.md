# C170 exploratory position-holdout protocol and acceptance record

Historical record date: 2026-09-27. Baseline, second-round candidates, and validation ensembles completed without meeting the fixed gate. At this date final-test was not executed and W02 was not collected. The later F2 test on 2026-09-29 is reported in [EXPERIMENT_RESULTS.md](../EXPERIMENT_RESULTS.md); it does not change this dated record.

## Objective and data facts

Hold out complete distance_mm positions to assess transfer within one participant/session. Position holdout alone does not establish arbitrary/continuous-angle or rewearing invariance. Fix criteria before seeing validation/test and do not revise them after results. W01 was a separate failed experiment: selected test F1 .478±.040, post hoc best .670±.026 and minimum mean recall .427; do not pool its metrics with C170.

Source OpenBCISession_2026-09-26_20-05-53 has 20 recorded positions, 80 complete files, 1,600 two-second windows at 250 Hz/four EXG channels, plus 56 excluded short files. Forty-one recordings use columns 0,2,4,6 and 39 use 0,1,2,3; both orders correspond to S0,S1,S2,S3 per user confirmation. B057/B058 are at 70 mm, excluded as a complete group. B019 LEFT has 19 detected movements; its copied repetition 20 from repetition 10 is excluded from model input. Preserve original data and the NPZ.

| Split | Positions (mm) | Windows |
|---|---|---:|
| Train | 0,10,20,30,50,60,80,100,120,140,160 | 879 |
| Validation | 5,40,90,130 | 320 |
| Registered test | 15,25,110,150 | 320 |
| Excluded | 70 and B019 copy | 81 |

Total retained 1,519. Check actual configuration counts before interpreting results. The fixed 4,500,000 numeric-boundary audit identified 27 affected windows: train 24, validation 1, test 2. This is a quality record, not a hardware-saturation conclusion or permission to select exclusions from scores.

## Models, preprocessing, selection, and access boundary

run_c170_experiment.py applies 0.5–40 Hz bandpass, window/channel temporal demeaning, and training-fitted per-sensor scales. Compare raw/orbit and true/shuffled training labels over seeds 17/29/43/61/89. Shuffle complete direction recording blocks within each training position, preserving counts; validation/test labels stay true.

Select each seed checkpoint by validation F1, then validation loss, then seed. Select the principal family using true-label mean validation F1. A legacy final-test stage loads four selected checkpoints without retraining; test cannot select family/preprocessing/epochs. Preserve input, classes, split, hyperparameters, checkpoints and predictions.

The loader materializes the full NPZ and finite-checks all waveforms, including test, and checks whole-array block/position/test channel-map structure. Training, scaling, early stopping, and model selection use only train/validation. State that test metrics were not used; do not claim zero test-data reads. The historical record originally required adding dataset content SHA-256, command and code version to the final archive; later runners record hashes.

## Fixed gate and required reporting

Primary overall validation macro-F1 >=.80; each UP/DOWN/LEFT/RIGHT recall >=.65; true-label predictions cover all four classes; matched label-null accuracy should return toward four-class chance (~25%). Report true/null paired by model and seed, prediction counts, accuracy, precision/recall/F1 and confusion, overall and by position and position×channel-map. Do not replace overall acceptance with a favorable single-position result.

After failed validation, change models/training/preprocessing using train/validation only, retaining separate output directories and fixed splits/gates/null rules. Before final-test freeze family/checkpoints/seeds/preprocessing/scales/classes/gate/scorer. An exclusive final_test_started.json prevents replay within an output directory. If it fails after creating the marker, preserve partial artifacts and failure status; do not remove the marker to repeat scoring. Directory protection does not replace dataset identity checks. Changes after test exposure make further results exploratory and require a new independent holdout.

## Confounds and unexecuted follow-up

Position and direction order track acquisition time. Position holdout avoids shared positions across splits but cannot remove time/wear confounding. Stratify both maps while retaining their verified physical order. C170 contains movement windows, without a rest-window control. At this historical date raw-session access failed with Operation not permitted despite escalation; no new rest extraction or source-slice audit was completed. No causal explanation of failure follows from these limits.

Proposed W02: a separate rewear with the same participant/hardware for rewearing assessment; multiple participants require a separate design. Freeze 12 angles (0–330 degrees in 30-degree increments), four directions, five repetitions each, filename/quality/exclusion rules, and actual angle/reference measurement. Balance/randomize angles and randomize directions. Add rest/channel RMS every three angles and at start/end. Interrupt/log channel/contact changes and separately identify any rewear. Do not convert distance_mm into measured angle without calibration.

Freeze selected C170 checkpoints/scales/classes/seeds/scorer and SHA-256 before W02. Score W02 separately without training or candidate selection. If W02 drives changes, treat it as development data and collect W03. Neither this proposal nor W01 proves arbitrary-angle, cross-day, or cross-person capability.

## Actual validation results

All runs confirm 879/320/320 metadata counts. Baseline principal family: orbit.

| Family/condition | Mean validation macro-F1 ± seed SD |
|---|---:|
| Orbit, true | .6718 ± .0425 |
| Raw CNN, true | .6039 ± .0368 |
| Orbit, shuffled | .2137 ± .0834 |
| Raw CNN, shuffled | .2382 ± .0661 |
| Harmonic candidate, true | .663354 ± .039916 |
| Probability-orbit candidate, true | .688593 ± .051461 |
| Harmonic candidate, shuffled | .213785 ± .059045 |
| Probability-orbit candidate, shuffled | .199295 ± .086656 |

Orbit recalls UP/DOWN/LEFT/RIGHT: .8550/.6100/.6300/.6075. Orbit position F1 at 5/40/90/130 mm: .5941/.7103/.5227/.8045. Overall and three recall gates fail; 130 mm alone does not change acceptance.

Harmonic recalls .7850/.6525/.6475/.5750, position F1 .476104/.743105/.628076/.747853. Probability-orbit recalls .7975/.7200/.6975/.5500, position F1 .564089/.702544/.587166/.832229. Its best seed 17 scores .77501, still below .80. Both overall gates fail; probability-orbit RIGHT recall fails.

Equal softmax averaging of five probability-orbit seeds gives F1 .670852 and recalls .8875/.6625/.6375/.5125. Fifteen checkpoints across harmonic/probability-orbit/orbit give F1 .664871 and recalls .8750/.6250/.6500/.5250. Both ensembles fail. No final-test was triggered by these experiments. Training scores at validation-selected checkpoints are not capacity upper bounds.

## Artifacts and historical state

Baseline outputs/c170_distance_holdout_20260927 contains config.json, validation_metrics.json, validation_predictions.csv, and 20 checkpoints. Legacy selected checkpoints: true orbit seed17, true raw seed89, shuffled orbit seed61, shuffled raw seed29. Candidate outputs/c170_distance_holdout_candidates_20260927 contains its config/metrics/predictions and selected true orbit_prob seed17. Ensemble outputs/c170_distance_holdout_ensemble_verified_20260927 contains config, ensemble metrics/predictions and five-/fifteen-member checkpoints.

These dated directories have no final_test_metrics.json/final_test_predictions.csv from this phase; W02 remained uncollected. See [data README](../../data/processed_c170_20260926/README.md), [manifest](../../data/processed_c170_20260926/manifest.json), [builder](../../prepare_c170_session.py), [runner](../../run_c170_experiment.py), [tests](../../tests/test_run_c170_experiment.py), and [W01 report](2026-08-27-round1-rotation-invariance-feasibility.md).
