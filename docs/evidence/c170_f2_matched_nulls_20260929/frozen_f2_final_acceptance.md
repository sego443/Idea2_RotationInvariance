# Frozen F2 final acceptance

Created UTC: 2026-09-29T15:02:57+00:00

Fusion rule: Per member, argmax raw neural logits or raw v7 KRR scores. Within complex_gram5 and learned_power10, choose the most frequent member argmax, ties to lowest class ID. The two neural family winners and v7 group_true prediction each cast one vote; choose the most frequent family vote, ties to lowest class ID.

Rule correction: the historical F2 CSV was reproduced exactly on 320/320 rows, including validation macro-F1 **0.826871507239**. The earlier evaluator description that directly averaged family vote-frequency vectors is preserved at `E:\MyCode\codex\Idea2\Idea2_RotationInvariance\outputs\c170_f2_matched_nulls_20260929\matched_null_validation_metrics.json`; it was incorrect for the saved F2 predictions. Frozen candidates/checkpoints were not changed. Rule freeze SHA256: `61d2ab79191a829edbcb5435d812e85cd072204e755b61b815b2f0b8bd8cebe9`.

Validation replay: all 15 neural member predictions matched their saved validation CSVs. The five matched-null validation macro-F1 values were rep 17=0.065923, rep 29=0.142741, rep 43=0.126106, rep 61=0.205869, rep 89=0.127433; maximum **0.205869**, below the pre-set 0.5 gate.

Test was evaluated once after that gate, with no fitting, normalization fitting, checkpoint selection, threshold, or fusion changes. It used the train-fitted neural shared scalar and the saved v7 feature scaler. The final F2 has accuracy **0.6781**, macro-F1 **0.674897**, and recalls UP 0.6750, DOWN 0.4750, LEFT 0.7875, RIGHT 0.7750. It **does not pass** the pre-set macro-F1 0.80 and per-class recall 0.65 thresholds.

| Position (mm) | Accuracy | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.8875 | 0.886597 | 0.9500 | 0.6500 | 0.9500 | 1.0000 |
| 25 | 0.7250 | 0.732269 | 0.7500 | 0.7000 | 0.9000 | 0.5500 |
| 110 | 0.5125 | 0.469841 | 1.0000 | 0.2000 | 0.3000 | 0.5500 |
| 150 | 0.5875 | 0.518848 | 0.0000 | 0.3500 | 1.0000 | 1.0000 |

Confusion matrix (true rows, predicted columns; order UP, DOWN, LEFT, RIGHT):

```text
54 6 4 16
36 38 0 6
1 0 63 16
0 9 9 62
```

Matched-null test macro-F1: rep 17=0.065844, rep 29=0.118428, rep 43=0.142697, rep 61=0.169321, rep 89=0.068759.

Artifacts are in `E:\MyCode\codex\Idea2\Idea2_RotationInvariance\outputs\c170_f2_matched_nulls_20260929`: final metrics JSON, per-row test predictions CSV, validation replay, corrected null validation outputs, and the rule-revision record. The earlier erroneous null fusion file remains untouched. The test result is final for this frozen candidate and was not used to change it.


## Independent data inventory

No unused independent real multi-position dataset was found in the checked project metadata. The Rotation session `data/OpenBCISession_2026-08-27_14-37-15` contains direction-labeled blocks at 0°–330° in 30° increments, but the existing feasibility report records it as a previously used single-person, single-session, single-wear study. `data/processed_c170_20260926` is the C170 bundle used here, including the evaluated test split, so it cannot become a new holdout. The three HWverify sessions contain direction or letter filenames but no wearing-angle/position metadata. The round-one collection plan has empty `actual_filename`/`accepted` fields. This inventory used filenames and existing documentation only; it did not read or process signal data.
