# Fixed v7 + v4 + v6 soft-probability validation fusion

Exploratory validation-only fusion from saved model predictions plus one reconstructed v7 validation score matrix. No weights, temperatures, seeds or thresholds were tuned. This does not count as a test-suite run or evidence that the formal project target is met; the same validation split has informed prior method development, and no matched three-family shuffle-fusion null is included.

## Fixed fusion rule

For v4 `harmonic_gram` and v6 `learned_power`, the five saved per-seed softmax vectors are averaged per sample. For v7, the selected `bandpass_0p5_40 / bcd / group_true` score is reconstructed from its tensor-only safe NPZ with the original C4 group-RBF equation; each class score is clipped to [0,1] and each row is normalized, with uniform fallback only if the clipped row sum is zero. The three family vectors are then averaged with equal one-third weight. `argmax` ties prefer UP, then DOWN, LEFT, RIGHT.

The v7 full raw NPZ loader decodes the compressed file and performs its pre-existing finite-value scan. The fixed split is then applied; only train and validation indices enter filtering, feature extraction or scaler fitting. No registered-test rows were processed or scored. No test metrics were read.

Row identity was explicitly reconstructed from v7's fixed validation split metadata and exported to `validation_row_identity.csv`; all 320 row IDs, blocks, positions, directions and channel maps match v4/v6 predictions. The reconstructed unmodified v7 score argmax matches all stored predictions and its recorded macro-F1 before clipping/normalization.

## Validation metrics

| Family / fusion | Accuracy | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Predicted counts (UP, DOWN, LEFT, RIGHT) |
|---|---:|---:|---:|---:|---:|---:|---|
| v7_group_true_single | 0.6781 | 0.6864 | 0.6500 | 0.7750 | 0.5500 | 0.7375 | (56, 72, 88, 104) |
| v4_harmonic_gram_5seed_mean_probability | 0.6844 | 0.6882 | 0.7750 | 0.6625 | 0.6250 | 0.6750 | (75, 69, 94, 82) |
| v6_learned_power_5seed_mean_probability | 0.7406 | 0.7370 | 0.8125 | 0.8375 | 0.7750 | 0.5375 | (75, 95, 89, 61) |
| equal-family soft fusion | 0.7625 | 0.7655 | 0.8375 | 0.8250 | 0.6875 | 0.7000 | (72, 79, 88, 81) |

Equal-family fusion by validation position:

| Position (mm) | n | Accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 5 | 80 | 0.5750 | 0.5947 |
| 40 | 80 | 0.8750 | 0.8735 |
| 90 | 80 | 0.7625 | 0.7567 |
| 130 | 80 | 0.8375 | 0.8349 |

Pairwise overlap of family-majority errors:

| Family pair | Both wrong | Either wrong | Jaccard |
|---|---:|---:|---:|
| v7_group_true_single / v4_harmonic_gram_5seed_mean_probability | 40 | 164 | 0.2439 |
| v7_group_true_single / v6_learned_power_5seed_mean_probability | 35 | 151 | 0.2318 |
| v4_harmonic_gram_5seed_mean_probability / v6_learned_power_5seed_mean_probability | 53 | 131 | 0.4046 |

Oracle upper bound: at least one family was correct on 296/320 samples (0.9250); this uses the true label per sample and is not deployable.

The historical gate reference is macro-F1 ≥0.80 and recall ≥0.65 for every class. This fusion's validation macro-F1 is 0.7655; its recalls are UP 0.8375, DOWN 0.8250, LEFT 0.6875, RIGHT 0.7000. Regardless of those point estimates, this exploratory validation fusion is not formal gate evidence without a matched fusion null and a later authorized sealed-test evaluation.

Full per-class/per-position metrics, fusion definitions, safe-NPZ provenance, source/data hashes, and runtime are in `summary.json`. The row-level score, family hard-vote/soft-probability, and ensemble vectors are in `family_probability_vectors.csv` for the single pre-authorized follow-up blend.
