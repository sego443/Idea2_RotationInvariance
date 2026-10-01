# C170 train/validation waveform quality and confound diagnostic

Date: 2026-09-28. Read-only fixed train/validation diagnostic; no model training, source-data edits, or GPU use. Historical values were translated, not recomputed from waveforms in this documentation task.

## Access and preprocessing

Use data/processed_c170_20260926/dataset.npz and manifest.json. Existing split_by_distance/filter_windows/calibrate_windows/prepare_model_inputs helpers define train 0,10,20,30,50,60,80,100,120,140,160 and validation 5,40,90,130 mm. No test waveform indices were created or used.

NPZ stores X_raw as one compressed array, so selecting rows decodes the complete compressed entry; immediately retain only deduplicated train/validation rows. No test slice is assigned, finite-checked, filtered, summarized, printed, or used for fitting. Do not confuse container decoding with analysis. The diagnostic did not call the legacy loader's whole-array finite check.

Third-order Butterworth 0.5–40 Hz sosfiltfilt, per-window/channel demeaning, then training-fitted sensor scales `[10522.129,9305.551,11992.974,12856.614]`. Validation reuses them. All following RMS, median magnitude, peaks and crest factors describe processed dimensionless waveforms, not raw ADC values.

## Channel map, position, and sequence

Both EXG maps correspond to S0,S1,S2,S3, not distinct physical layouts.

| Split | 0,2,4,6 | 0,1,2,3 | Total |
|---|---:|---:|---:|
| Train | 479 | 400 | 879 |
| Validation | 160 | 160 | 320 |

Old-map train positions: 0,10,20,30,50,60; new-map: 80,100,120,140,160. Old-map validation: 5,40; new-map: 90,130. None of the 15 included positions contains both maps. The change occurs at B057→B058 at excluded 70 mm: UP uses old, DOWN onward uses new. Map, position range, and time are inseparable in these data.

Old-map train direction counts UP/DOWN/LEFT/RIGHT: 120/120/119/120; new-map 100 each. Validation has 40 per class per map. The 19-window cell is 20 mm LEFT/B019 after copy exclusion.

Received full-model validation F1 by 5/40/90/130 mm: .729/.712/.441/.441; no-RMS .772/.619/.496/.546. These correlate with metadata, but cannot identify map, RMS, or drift as causes.

## Training-scaled amplitude

Pooled RMS is sqrt(mean of squares across windows/time); median-|x| pools absolute samples. The all-train median per sensor is 1 by definition, not necessarily 1 within strata.

| Map | Sensor | Train RMS | Train median-|x| | Validation RMS | Validation median-|x| |
|---|---|---:|---:|---:|---:|
| 0,1,2,3 | S0 | 8.195 | .901 | 12.564 | 2.092 |
| 0,1,2,3 | S1 | 7.125 | .796 | 5.720 | 1.048 |
| 0,1,2,3 | S2 | 2.939 | .747 | 3.730 | .974 |
| 0,1,2,3 | S3 | 6.175 | .983 | 3.932 | .445 |
| 0,2,4,6 | S0 | 4.771 | 1.087 | 4.070 | .897 |
| 0,2,4,6 | S1 | 6.915 | 1.293 | 2.890 | .798 |
| 0,2,4,6 | S2 | 7.148 | 1.256 | 4.034 | 1.116 |
| 0,2,4,6 | S3 | 7.553 | 1.012 | 4.968 | 1.442 |

These describe distribution shifts after filtering/scaling, not hardware faults, physiological mechanisms, or causal explanations.

## Flatness, peaks, and duplication

Units: 44 train plus 16 validation position-direction groups; 20 windows each except B019's 19.

| Check | Train | Validation | Definition/interpretation |
|---|---:|---:|---|
| Almost flat sensor-windows | 0 | 0 | RMS or peak-to-peak <=1e-6, normalized units |
| Crest factor >10 | 1 | 2 | max-|x|/RMS; train 80 mm UP; validation 130 mm DOWN/S3; candidates, not saturation |
| Above train sensor-window peak Q99.9 | 4 | 0 | Training thresholds S0–S3: 153.30/88.76/119.40/75.03 |
| Adjacent identical sensor samples | 0 | 1 | Validation 40 mm RIGHT: 1/39,920=.002505%; filtered float values |
| Byte-identical processed windows | 0 | 0 | Within train/validation, after excluding the marked copy |

Maximum crest factor: train 10.07, validation 12.87. Global normalized absolute peaks: 164.55/96.78. No physical ADC rail threshold is recorded; do not call these saturation. Filtered repeated floats do not substitute for raw repeated-code statistics.

## Interpretation and recomputation method

The checks support absence of flat windows under these thresholds, a few high-crest candidates, and map/position distribution differences. They do not attribute failed classification to hardware, map, RMS, or time, and do not establish cross-person/day/rewear performance. Existing validation remains exploratory.

For recomputation: read metadata, exclude is_duplicate, form fixed train/validation indices, decode X_raw and immediately subset train+validation, concatenate those rows, form local indices, and call prepare_model_inputs. This historical diagnostic saved no extra script or waveform copy.
