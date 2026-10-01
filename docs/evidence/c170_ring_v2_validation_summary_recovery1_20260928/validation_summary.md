# C170 Ring CUDA v2 Validation Summary

Verified runs: 20/20. Test metrics computed: **false**.

## Variant results

| Variant | Macro-F1 mean ± SD | UP recall | DOWN recall | LEFT recall | RIGHT recall | selected train-target F1 | train weighted CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| ring_main | 0.6180 ± 0.0206 | 0.8375 | 0.5500 | 0.6375 | 0.4800 | 0.9594 ± 0.0364 | 0.1356 ± 0.1147 |
| ring_ce | 0.6173 ± 0.0361 | 0.7875 | 0.4900 | 0.7225 | 0.4900 | 0.9630 ± 0.0382 | 0.1173 ± 0.1005 |
| slot_supcon | 0.5795 ± 0.0256 | 0.7500 | 0.3750 | 0.6075 | 0.6375 | 0.9959 ± 0.0066 | 0.0251 ± 0.0274 |
| ring_shuffle | 0.1788 ± 0.0222 | 0.0650 | 0.2125 | 0.2675 | 0.1975 | 0.9920 ± 0.0139 | 0.0344 ± 0.0335 |

## Gate

ring_main macro-F1 gate: FAIL (mean 0.6180 vs 0.80).
- UP mean recall 0.8375 vs 0.65: PASS.
- DOWN mean recall 0.5500 vs 0.65: FAIL.
- LEFT mean recall 0.6375 vs 0.65: FAIL.
- RIGHT mean recall 0.4800 vs 0.65: FAIL.

Overall ring_main gate: **FAIL**.

## Validation macro-F1 by position

Mean ± population SD across five seeds.

| Position (mm) | ring_main | ring_ce | slot_supcon | ring_shuffle |
|---:|---:|---:|---:|---:|
| 5 | 0.4843 ± 0.0792 | 0.4116 ± 0.0929 | 0.6998 ± 0.1046 | 0.2108 ± 0.0760 |
| 40 | 0.5906 ± 0.0579 | 0.6356 ± 0.0833 | 0.5569 ± 0.0929 | 0.1758 ± 0.1128 |
| 90 | 0.6354 ± 0.0946 | 0.6503 ± 0.1047 | 0.4707 ± 0.0874 | 0.1387 ± 0.0665 |
| 130 | 0.7321 ± 0.0743 | 0.7209 ± 0.0723 | 0.4707 ± 0.0304 | 0.1225 ± 0.0378 |

## Paired seed differences

Difference is ring_main minus comparator validation macro-F1.

- ring_main_minus_ring_ce: mean +0.0006, population SD 0.0255.
- ring_main_minus_slot_supcon: mean +0.0385, population SD 0.0436.
- ring_main_minus_ring_shuffle: mean +0.4392, population SD 0.0326.

Population standard deviations (ddof=0) are used throughout. Train/validation metric gaps are in the per-seed CSV and JSON.

## Shuffle validation prediction distribution

- predicted UP: 77.0 ± 17.7 / 320 across seeds.
- predicted DOWN: 83.4 ± 23.2 / 320 across seeds.
- predicted LEFT: 98.2 ± 23.1 / 320 across seeds.
- predicted RIGHT: 61.4 ± 23.2 / 320 across seeds.

The loader decodes the complete NPZ and finite-checks X_raw, including registered-test waveforms. Only train/validation indices enter preprocessing and model computations; test contributes split metadata only. Position and channel-map strata are descriptive; this single-session split does not identify causal effects. The C4 ring tensor is discrete and does not establish continuous physical rotation invariance. The registered test split is not predicted or scored; a failed validation gate does not authorize a test run.
