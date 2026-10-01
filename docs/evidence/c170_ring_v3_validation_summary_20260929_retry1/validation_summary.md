# C170 Shared-Scale v3 Validation Summary

Verified v3 runs: 10/10. Test metrics computed: false.
Train-only shared scale recorded by all runs: 11123.03711.

## Validation metrics

Mean ± population SD across five seeds.

| Variant | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Selected train-target F1 | Train-target weighted CE | Unique predicted classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_scale_ce | 0.6245 ± 0.0202 | 0.8375 ± 0.0848 | 0.4725 ± 0.0588 | 0.6225 ± 0.1076 | 0.5850 ± 0.0619 | 0.9803 ± 0.0192 | 0.0587 ± 0.0523 | 4.00 ± 0.00 |
| shared_scale_shuffle | 0.1544 ± 0.0142 | 0.0725 ± 0.0629 | 0.2300 ± 0.0816 | 0.1875 ± 0.0806 | 0.1450 ± 0.0882 | 0.9882 ± 0.0220 | 0.0478 ± 0.0699 | 4.00 ± 0.00 |

## Validation gate

Candidate: shared_scale_ce. Thresholds: mean macro-F1 ≥ 0.80 and every class mean recall ≥ 0.65.

- UP mean recall 0.8375: PASS.
- DOWN mean recall 0.4725: FAIL.
- LEFT mean recall 0.6225: FAIL.
- RIGHT mean recall 0.5850: FAIL.
- Macro-F1 mean 0.6245: FAIL.
- Overall gate: FAIL.

## Validation macro-F1 by position

| Position (mm) | shared_scale_ce | shared_scale_shuffle |
|---:|---:|---:|
| 5 | 0.4853 ± 0.0758 | 0.1872 ± 0.0614 |
| 40 | 0.6451 ± 0.0321 | 0.1248 ± 0.0813 |
| 90 | 0.5899 ± 0.0584 | 0.1192 ± 0.0563 |
| 130 | 0.7411 ± 0.0747 | 0.1208 ± 0.0324 |

## Train/validation gaps

Train-target scores for the shuffle arm use permuted training labels.

- shared_scale_ce: F1 gap +0.3558 ± 0.0286; weighted-CE gap -1.0792 ± 0.3816.
- shared_scale_shuffle: F1 gap +0.8338 ± 0.0196; weighted-CE gap -4.2293 ± 0.3720.

## Paired seed differences

Difference is candidate minus comparator validation macro-F1.

- shared_scale_ce_minus_v2_ring_ce: +0.0071 ± 0.0202 (population SD).
- shared_scale_ce_minus_shared_scale_shuffle: +0.4701 ± 0.0258 (population SD).
- shared_scale_shuffle_minus_v2_ring_ce: -0.4630 ± 0.0353 (population SD).

## Shuffle prediction distribution

- predicted UP: 75.8 ± 20.2 / 320.
- predicted DOWN: 104.8 ± 31.1 / 320.
- predicted LEFT: 83.6 ± 5.5 / 320.
- predicted RIGHT: 55.8 ± 19.3 / 320.

The complete compressed NPZ is decoded and the loader finite-checks X_raw, including registered-test waveforms. Only train and validation indices enter preprocessing and model computations; test is not preprocessed, used to estimate the shared scale, predicted, or scored. Shuffle-arm train metrics use permuted training labels. This experiment tests the discrete C4 channel-roll property of preprocessing and does not establish continuous SO(2) or real physical rotation invariance. A failed validation gate does not authorize test evaluation.
