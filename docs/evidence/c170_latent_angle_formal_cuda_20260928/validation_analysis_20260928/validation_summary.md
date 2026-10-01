# C170 latent-angle validation results

This report uses validation predictions only. It does not read test predictions, scores, or labels.

Integrity: 25/25 formal runs verified; all marker, manifest, config, artifact and source hashes passed.

## Validation results

| Variant | Macro-F1 mean ± SD | UP recall | DOWN recall | LEFT recall | RIGHT recall | Unique predicted classes |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 0.609 ± 0.039 | 0.900 ± 0.071 | 0.400 ± 0.064 | 0.515 ± 0.025 | 0.667 ± 0.074 | 4.000 ± 0.000 |
| no_angle | 0.596 ± 0.029 | 0.948 ± 0.018 | 0.340 ± 0.046 | 0.450 ± 0.026 | 0.725 ± 0.041 | 4.000 ± 0.000 |
| full | 0.611 ± 0.051 | 0.930 ± 0.019 | 0.378 ± 0.089 | 0.450 ± 0.066 | 0.758 ± 0.036 | 4.000 ± 0.000 |
| no_rms | 0.640 ± 0.031 | 0.938 ± 0.011 | 0.443 ± 0.077 | 0.517 ± 0.017 | 0.700 ± 0.059 | 4.000 ± 0.000 |
| shuffle | 0.172 ± 0.044 | 0.048 ± 0.040 | 0.273 ± 0.070 | 0.240 ± 0.081 | 0.148 ± 0.067 | 4.000 ± 0.000 |

## Validation gate

Full mean macro-F1 gate: FAIL (0.611 against 0.800).
All four mean class-recall gates: FAIL (each must be at least 0.650).
Final-test evaluation was not started by this analysis.

## Paired seed differences

| Comparison | Mean full-minus-control macro-F1 | SD | Seed differences |
|---|---:|---:|---|
| full minus mlp | +0.002 | 0.054 | 17: +0.052, 29: -0.052, 43: -0.006, 61: -0.058, 89: +0.075 |
| full minus no angle | +0.016 | 0.038 | 17: +0.067, 29: -0.032, 43: -0.023, 61: +0.018, 89: +0.048 |
| full minus no rms | -0.029 | 0.049 | 17: +0.060, 29: -0.058, 43: -0.075, 61: -0.058, 89: -0.013 |
| full minus shuffle | +0.439 | 0.071 | 17: +0.552, 29: +0.445, 43: +0.379, 61: +0.350, 89: +0.468 |

## By-position validation macro-F1

| Variant | Position (mm) | Mean ± SD |
|---|---:|---:|
| mlp | 5 | 0.711 ± 0.159 |
| mlp | 40 | 0.684 ± 0.045 |
| mlp | 90 | 0.460 ± 0.123 |
| mlp | 130 | 0.489 ± 0.102 |
| no_angle | 5 | 0.802 ± 0.061 |
| no_angle | 40 | 0.610 ± 0.026 |
| no_angle | 90 | 0.384 ± 0.062 |
| no_angle | 130 | 0.425 ± 0.040 |
| full | 5 | 0.729 ± 0.106 |
| full | 40 | 0.712 ± 0.040 |
| full | 90 | 0.441 ± 0.040 |
| full | 130 | 0.441 ± 0.054 |
| no_rms | 5 | 0.772 ± 0.041 |
| no_rms | 40 | 0.619 ± 0.076 |
| no_rms | 90 | 0.496 ± 0.029 |
| no_rms | 130 | 0.546 ± 0.040 |
| shuffle | 5 | 0.091 ± 0.030 |
| shuffle | 40 | 0.152 ± 0.101 |
| shuffle | 90 | 0.202 ± 0.132 |
| shuffle | 130 | 0.117 ± 0.063 |

## Selected epochs

| Variant | Seed 17 | Seed 29 | Seed 43 | Seed 61 | Seed 89 |
|---|---:|---:|---:|---:|---:|
| mlp | 1 | 38 | 1 | 13 | 10 |
| no_angle | 20 | 8 | 21 | 14 | 16 |
| full | 72 | 27 | 2 | 20 | 18 |
| no_rms | 29 | 84 | 39 | 44 | 28 |
| shuffle | 1 | 15 | 6 | 7 | 64 |

## 64-versus-128 angle-grid stability

This validation-only diagnostic did not participate in checkpoint or model selection.
Mean argmax agreement: 1.000000; mean maximum action-probability difference: 4.17e-07; mean macro-F1 difference (128−64): +0.0000.

The full JSON and curve CSV include per-seed confusion matrices, per-class recalls, prediction diversity, all validation curves, position×channel-map strata, and the shuffle-control distribution.
