# C170 Harmonic-Phase v4 Validation Summary

Verified v4 runs: 20/20. Test metrics computed: false.
Mean ± population SD across five seeds; all scores below are validation-only.

| Variant | Macro-F1 | UP recall | DOWN recall | LEFT recall | RIGHT recall | Validation CE | Selected train-target F1 | Train-target F1 gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| harmonic_gram | 0.6570 ± 0.0240 | 0.6900 ± 0.0871 | 0.6775 ± 0.1372 | 0.5525 ± 0.1029 | 0.7100 ± 0.0668 | 1.0072 ± 0.1406 | 0.8635 ± 0.1334 | +0.2065 ± 0.1341 |
| harmonic_power | 0.5570 ± 0.0293 | 0.6825 ± 0.0400 | 0.5375 ± 0.1140 | 0.6125 ± 0.1061 | 0.3900 ± 0.0804 | 1.5537 ± 0.2640 | 0.9454 ± 0.0766 | +0.3883 ± 0.0497 |
| harmonic_phase | 0.6502 ± 0.0120 | 0.7650 ± 0.0604 | 0.6125 ± 0.0454 | 0.4625 ± 0.0403 | 0.7625 ± 0.0818 | 0.9909 ± 0.0941 | 0.9264 ± 0.0800 | +0.2762 ± 0.0746 |
| harmonic_shuffle | 0.2010 ± 0.0654 | 0.1650 ± 0.1592 | 0.2150 ± 0.1379 | 0.2525 ± 0.1379 | 0.2050 ± 0.1125 | 3.2022 ± 1.1090 | 0.9124 ± 0.0756 | +0.7114 ± 0.1248 |

## Validation Macro-F1 by Position

| Position (mm) | harmonic_gram | harmonic_power | harmonic_phase | harmonic_shuffle |
|---:|---:|---:|---:|---:|
| 5 | 0.5263 ± 0.0575 | 0.5309 ± 0.0302 | 0.7080 ± 0.0927 | 0.2046 ± 0.0738 |
| 40 | 0.7201 ± 0.0372 | 0.5989 ± 0.1086 | 0.8089 ± 0.0673 | 0.1785 ± 0.1038 |
| 90 | 0.5945 ± 0.1160 | 0.4930 ± 0.0792 | 0.4248 ± 0.0821 | 0.1961 ± 0.1526 |
| 130 | 0.7400 ± 0.0616 | 0.4917 ± 0.0726 | 0.5663 ± 0.0960 | 0.1690 ± 0.0891 |

## Paired Seed Differences

Difference is harmonic_gram minus comparator validation macro-F1.

- harmonic_gram_minus_harmonic_power: +0.1000 ± 0.0472; 17:+0.0600, 29:+0.0599, 43:+0.0649, 61:+0.1519, 89:+0.1633.
- harmonic_gram_minus_harmonic_phase: +0.0068 ± 0.0255; 17:+0.0007, 29:-0.0279, 43:+0.0095, 61:+0.0004, 89:+0.0511.
- harmonic_gram_minus_harmonic_shuffle: +0.4560 ± 0.0466; 17:+0.4478, 29:+0.5284, 43:+0.4870, 61:+0.4138, 89:+0.4030.
- harmonic_gram_minus_v3_shared_scale_ce: +0.0325 ± 0.0112; 17:+0.0167, 29:+0.0392, 43:+0.0213, 61:+0.0442, 89:+0.0412.

The harmonic_shuffle train-target F1 is measured against its permuted training labels. The full NPZ may be decoded and finite-checked; the registered test split was not preprocessed, used to estimate statistics, predicted, or scored.

Metrics are recomputed only from the 320-row validation prediction CSVs and selected-checkpoint training metrics in the manifests. The complete compressed NPZ may have been decoded and finite-checked by the runner, but test waveforms were not preprocessed, used to fit statistics, supplied to a model, predicted, or scored. The harmonic_gram/power features are invariant to a common complex phase in the discrete first-harmonic representation; this does not establish continuous physical SO(2) invariance. The harmonic_shuffle training score is against permuted training targets. A failed validation gate does not authorize test evaluation.
