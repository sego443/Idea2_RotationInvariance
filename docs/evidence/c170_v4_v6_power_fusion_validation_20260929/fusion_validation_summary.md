# Fixed v4/v6 Validation Probability Fusion Diagnostic

Exploratory validation-only late fusion: fixed 0.5/0.5 probability average, then argmax; no weight or temperature search.
The fixed seeds 17/29/43/61/89 are paired branch-by-branch; no best seed was selected. All 320 rows per seed passed identity, probability, marker, config, and artifact-hash checks. Test metrics were not computed.

| Metric | Mean ± population SD |
|---|---:|
| v4 harmonic_gram macro-F1 | 0.6570 ± 0.0240 |
| v6 learned_power macro-F1 | 0.6602 ± 0.0479 |
| Fused macro-F1 | 0.6789 ± 0.0335 |
| Fused − v4 paired macro-F1 | +0.0219 ± 0.0445 |
| Fused − v6 paired macro-F1 | +0.0187 ± 0.0224 |

| Fused recall | Mean ± population SD |
|---|---:|
| UP | 0.7750 ± 0.0542 |
| DOWN | 0.6975 ± 0.0917 |
| LEFT | 0.6450 ± 0.0367 |
| RIGHT | 0.5950 ± 0.0491 |

| Position (mm) | Fused macro-F1 |
|---:|---:|
| 5 | 0.6045 ± 0.0501 |
| 40 | 0.7054 ± 0.0680 |
| 90 | 0.6359 ± 0.0354 |
| 130 | 0.7513 ± 0.0821 |

## Error Complementarity

Mean fraction of the 320 paired validation examples per seed; categories are mutually exclusive.

| Category | Mean ± population SD |
|---|---:|
| both_wrong | 0.1694 ± 0.0193 |
| exactly_one_wrong | 0.3412 ± 0.0196 |
| both_correct | 0.4894 ± 0.0260 |

Exploratory late fusion on the fixed validation split; the five fixed seeds 17/29/43/61/89 are paired branch-by-branch, with no best-seed selection, fitted fusion weights, or temperature. All ten branch runs were verified by config/marker/artifact hashes, and each same-seed pair sample_index, distance, true label, and channel map aligned exactly. Each branch probability vector and its 0.5/0.5 average was checked for finite normalized probabilities. Both branches have exact C4 and ideal common first-harmonic U1 invariance, which arithmetic averaging preserves; this is not evidence of physical continuous rotation generalization, a novelty claim, or a formal performance gate. The descriptive fusion remains below the 0.80 macro-F1 / 0.65-per-class validation gate (RIGHT recall is 0.595); no additional training is warranted for this diagnostic. There is no matched learned_power_shuffle branch, so the matched fusion-null control is absent. Both-wrong/exactly-one-wrong/both-correct counts are descriptive error complementarity only; no oracle upper bound is computed. Artifact hashes were checked, but no checkpoint was deserialized; the dataset NPZ was not opened and test waveforms or metrics were not accessed.
