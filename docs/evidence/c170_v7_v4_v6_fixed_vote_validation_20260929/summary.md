# Fixed v7 + v4 + v6 validation vote diagnostic

Exploratory, validation-only cross-method fusion from existing predictions. No inference was rerun, and no raw dataset or registered-test predictions/metrics were read. This unweighted validation fusion was not a preregistered model-selection step and cannot serve as formal target-gate evidence.

The three families are: v7 `bandpass_0p5_40 / bcd / group_true` as one one-hot prediction; v4 `harmonic_gram` as the equal five-seed one-hot vote; and v6 `learned_power` as the equal five-seed one-hot vote. Those three family vectors receive equal weight. All ties resolve to the first class in the fixed order UP, DOWN, LEFT, RIGHT.

Validation sample identity was checked across all v4/v6 seed files and against the v7 ordered row labels: 320 rows, positions 5/40/90/130 mm. V7 row indices are 0–319 and match the canonical true-label sequence.

## Validation metrics

| Family / prediction | Accuracy | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Prediction counts (UP, DOWN, LEFT, RIGHT) |
|---|---:|---:|---:|---:|---:|---:|---|
| v7_group_true_single | 0.6781 | 0.6864 | 0.6500 | 0.7750 | 0.5500 | 0.7375 | (56, 72, 88, 104) |
| v4_harmonic_gram_5seed_vote | 0.7094 | 0.7077 | 0.7625 | 0.7625 | 0.5250 | 0.7875 | (72, 84, 70, 94) |
| v6_learned_power_5seed_vote | 0.7250 | 0.7203 | 0.8250 | 0.8125 | 0.7500 | 0.5125 | (80, 96, 84, 60) |
| equal-family fusion | 0.7875 | 0.7893 | 0.8375 | 0.9000 | 0.6250 | 0.7875 | (70, 81, 75, 94) |

Fusion per-position macro-F1:

| Position (mm) | n | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 5 | 80 | 0.6000 | 0.6245 |
| 40 | 80 | 0.9250 | 0.9246 |
| 90 | 80 | 0.7500 | 0.7289 |
| 130 | 80 | 0.8750 | 0.8710 |

## Error overlap and oracle reference

| Family pair | Left errors | Right errors | Both wrong | Either wrong | Jaccard |
|---|---:|---:|---:|---:|---:|
| v7_group_true_single / v4_harmonic_gram_5seed_vote | 103 | 93 | 43 | 153 | 0.2810 |
| v7_group_true_single / v6_learned_power_5seed_vote | 103 | 88 | 35 | 156 | 0.2244 |
| v4_harmonic_gram_5seed_vote / v6_learned_power_5seed_vote | 93 | 88 | 46 | 135 | 0.3407 |

At least one of the three family-majority predictions was correct on 296/320 samples (0.9250). This is an oracle upper bound that consults each sample’s true label and is not deployable performance.

These results are exploratory because the cross-family fusion rule is evaluated on the same validation set used in the prior method development. There is no matching, untouched fusion set, so this result does not establish the project’s formal target as met. The held-out test set remains unopened.

Machine-readable metrics, per-class precision/recall/F1, position summaries, sample IDs, and input SHA-256 values are in `summary.json`. Reproduction: run `analyze_fixed_vote.py` with the project’s Python 3.10 environment.
