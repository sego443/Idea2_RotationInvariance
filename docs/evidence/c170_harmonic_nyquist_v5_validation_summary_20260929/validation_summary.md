# C170 Harmonic-Nyquist v5 Validation Summary

Verified runs: 20/20. Test metrics computed: false.
Mean ± population SD across five seeds; all reported model scores are validation-only.

| Variant | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Validation CE | Selected train-target F1 | Train-target F1 gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nyquist_coupled | 0.6075 ± 0.0562 | 0.6950 ± 0.1722 | 0.4950 ± 0.1091 | 0.5350 ± 0.0515 | 0.7225 ± 0.0860 | 1.2700 ± 0.3938 | 0.9543 ± 0.0470 | 0.3469 ± 0.0592 |
| nyquist_uncoupled | 0.6506 ± 0.0643 | 0.6750 ± 0.0607 | 0.5850 ± 0.1465 | 0.5800 ± 0.1461 | 0.7675 ± 0.1120 | 1.0024 ± 0.1705 | 0.8734 ± 0.0822 | 0.2227 ± 0.0914 |
| gram_control | 0.6567 ± 0.0243 | 0.7250 ± 0.1247 | 0.5800 ± 0.0835 | 0.6325 ± 0.1027 | 0.6875 ± 0.1294 | 1.0762 ± 0.1633 | 0.9376 ± 0.1028 | 0.2809 ± 0.1068 |
| nyquist_shuffle | 0.2084 ± 0.0321 | 0.1550 ± 0.1172 | 0.2750 ± 0.0806 | 0.2500 ± 0.0627 | 0.1650 ± 0.1590 | 4.3117 ± 1.4721 | 0.9094 ± 0.1756 | 0.7010 ± 0.1858 |

## Validation Macro-F1 by Position

| Position (mm) | Coupled | Uncoupled | Gram control | Shuffle |
|---:|---:|---:|---:|---:|
| 5 | 0.5272 ± 0.1403 | 0.4820 ± 0.0654 | 0.5057 ± 0.1065 | 0.1605 ± 0.1094 |
| 40 | 0.8355 ± 0.0519 | 0.8106 ± 0.0593 | 0.7147 ± 0.0878 | 0.1699 ± 0.0596 |
| 90 | 0.3871 ± 0.1077 | 0.5281 ± 0.1571 | 0.5794 ± 0.0918 | 0.2623 ± 0.1943 |
| 130 | 0.6606 ± 0.0510 | 0.7253 ± 0.0368 | 0.7227 ± 0.0887 | 0.1663 ± 0.0926 |

## Paired Seed Differences

Differences are first arm minus comparator; mean ± population SD.

- nyquist_coupled_minus_nyquist_uncoupled: -0.0432 ± 0.0915; 17:+0.0830, 29:-0.1458, 43:+0.0028, 61:-0.0041, 89:-0.1517.
- nyquist_coupled_minus_gram_control: -0.0492 ± 0.0504; 17:-0.0275, 29:-0.1070, 43:+0.0320, 61:-0.0465, 89:-0.0971.
- nyquist_coupled_minus_nyquist_shuffle: +0.3991 ± 0.0860; 17:+0.4987, 29:+0.3252, 43:+0.5092, 61:+0.3417, 89:+0.3206.
- nyquist_coupled_minus_v4_harmonic_gram: -0.0495 ± 0.0690; 17:+0.0254, 29:-0.0834, 43:+0.0392, 61:-0.0952, 89:-0.1337.
- gram_control_minus_v4_harmonic_gram: -0.0003 ± 0.0377; 17:+0.0529, 29:+0.0236, 43:+0.0072, 61:-0.0487, 89:-0.0366.

All recomputed scores use only the 320-row validation prediction CSVs; selected-checkpoint training F1 is read from each run manifest. Per-seed differences are paired by the fixed seed. The complete compressed NPZ may be decoded and integrity-checked by the training runner, but test waveforms were not preprocessed, used to fit statistics, supplied to a model, predicted, or scored. The coupled Nyquist feature tests a discrete C4-restricted phase tradeoff, not continuous physical SO(2) wrist invariance. The shuffle-arm training score is against permuted training targets; it is not a clean-label train score.
