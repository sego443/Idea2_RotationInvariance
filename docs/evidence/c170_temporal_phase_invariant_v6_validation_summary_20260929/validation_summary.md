# C170 Temporal Phase-Invariant v6 Validation Summary

Verified runs: 20/20. Test metrics computed: false.
All means and standard deviations are across five seeds; scores are validation-only.

| Arm | Total/trainable params | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Selected train-target F1 | Train−validation F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| learned_gram | 8763/8763 | 0.6531 ± 0.0257 | 0.6025 ± 0.0772 | 0.7050 ± 0.0683 | 0.5475 ± 0.1125 | 0.7475 ± 0.0756 | 0.8929 ± 0.0951 | 0.2398 ± 0.1020 |
| fixed_gram | 8763/8308 | 0.6559 ± 0.0292 | 0.6550 ± 0.0816 | 0.7750 ± 0.1072 | 0.4500 ± 0.1671 | 0.7700 ± 0.0861 | 0.8610 ± 0.1491 | 0.2051 ± 0.1351 |
| learned_power | 8763/8763 | 0.6602 ± 0.0479 | 0.7700 ± 0.0843 | 0.7100 ± 0.1528 | 0.6600 ± 0.0718 | 0.5100 ± 0.0382 | 0.9667 ± 0.0200 | 0.3065 ± 0.0457 |
| learned_shuffle | 8763/8763 | 0.2256 ± 0.0454 | 0.1525 ± 0.0686 | 0.2850 ± 0.1174 | 0.3900 ± 0.0940 | 0.0850 ± 0.0722 | 0.8483 ± 0.2225 | 0.6227 ± 0.2231 |

## Validation Macro-F1 by Position

| Position (mm) | learned_gram | fixed_gram | learned_power | learned_shuffle |
|---:|---:|---:|---:|---:|
| 5 | 0.4183 ± 0.1040 | 0.4425 ± 0.0664 | 0.6139 ± 0.0639 | 0.1781 ± 0.0922 |
| 40 | 0.7732 ± 0.1093 | 0.8046 ± 0.0625 | 0.6440 ± 0.0841 | 0.2216 ± 0.0995 |
| 90 | 0.6039 ± 0.1577 | 0.5645 ± 0.0152 | 0.6302 ± 0.0978 | 0.2300 ± 0.1256 |
| 130 | 0.7480 ± 0.0773 | 0.7777 ± 0.0655 | 0.7230 ± 0.0823 | 0.2136 ± 0.0735 |

## Paired Seed Differences

Differences are first arm minus comparator.

- learned_gram_minus_fixed_gram: -0.0028 ± 0.0139; 17:+0.0152, 29:-0.0102, 43:+0.0105, 61:-0.0069, 89:-0.0228.
- learned_gram_minus_learned_power: -0.0072 ± 0.0441; 17:-0.0735, 29:-0.0312, 43:-0.0138, 61:+0.0398, 89:+0.0429.
- learned_gram_minus_learned_shuffle: +0.4275 ± 0.0566; 17:+0.4713, 29:+0.3899, 43:+0.5052, 61:+0.3460, 89:+0.4250.
- learned_gram_minus_v4_harmonic_gram: -0.0039 ± 0.0225; 17:+0.0303, 29:-0.0206, 43:+0.0134, 61:-0.0304, 89:-0.0124.
- fixed_gram_minus_v4_harmonic_gram: -0.0011 ± 0.0142; 17:+0.0152, 29:-0.0104, 43:+0.0029, 61:-0.0236, 89:+0.0104.
- learned_gram_minus_v5_gram_control: -0.0036 ± 0.0259; 17:-0.0225, 29:-0.0442, 43:+0.0062, 61:+0.0183, 89:+0.0242.
- fixed_gram_minus_v5_gram_control: -0.0008 ± 0.0330; 17:-0.0377, 29:-0.0340, 43:-0.0044, 61:+0.0252, 89:+0.0470.

Metrics are recomputed only from validation prediction CSVs and selected-checkpoint train metrics in run manifests. Aggregation verifies hashes and reads the numeric selected-weight NPZ with allow_pickle=False; it does not open the dataset NPZ. The test split was not preprocessed, used to estimate statistics, passed to a model, predicted, or scored. The phase invariance is exact for C4 channel rolls and an ideal common first-harmonic phase; it does not establish continuous physical SO(2) wrist invariance. The shuffle-arm train-target score uses permuted training labels.
