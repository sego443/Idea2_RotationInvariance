# Round-1 Rotation-Invariance Feasibility Result

## Evidence boundary

Single participant, single session, single donning, four pre-segmented directional strokes. Training angles were 0/60/120/180/240/300 degrees; validation angles were 30/150/270; test angles were 90/210/330 and were not used for model selection.

## Data audit

- 48 accepted movement files produced 240 strokes; class counts: [60, 60, 60, 60].
- Rejected recordings retained: 22; unlabelled accepted-looking extras: ['OpenBCI-RAW-2026-08-27_15-55-27.txt'].
- Files outside the planned 20-35 s duration: 31/48. All 48 nevertheless yielded five regular action centers; this is retained as a protocol deviation.
- Maximum action-interval CV: 0.088; maximum identical-sample run: 3 rows.
- Only EXG channels 0-3 were used. Raw files were not modified.

## Model results

Metrics below are mean over five frozen random seeds.

| Experiment/model/split | Macro F1 | Accuracy | Mean minimum recall |
|---|---:|---:|---:|
| primary_stroke/harmonic/test | 0.478 +/- 0.040 | 0.493 | 0.213 |
| primary_stroke/harmonic/train | 0.993 +/- 0.015 | 0.993 | 0.980 |
| primary_stroke/harmonic/val | 0.531 +/- 0.033 | 0.543 | 0.333 |
| primary_stroke/orbit/test | 0.670 +/- 0.026 | 0.683 | 0.427 |
| primary_stroke/orbit/train | 0.940 +/- 0.071 | 0.940 | 0.887 |
| primary_stroke/orbit/val | 0.441 +/- 0.039 | 0.453 | 0.267 |
| primary_stroke/raw/test | 0.495 +/- 0.106 | 0.523 | 0.240 |
| primary_stroke/raw/train | 0.985 +/- 0.030 | 0.985 | 0.967 |
| primary_stroke/raw/val | 0.437 +/- 0.062 | 0.437 | 0.293 |
| rest_only_control/harmonic/test | 0.238 +/- 0.076 | 0.267 | 0.093 |
| rest_only_control/raw/test | 0.182 +/- 0.042 | 0.197 | 0.053 |
| shuffled_label_control/harmonic/test | 0.229 +/- 0.071 | 0.247 | 0.067 |

Validation-selected model family: `harmonic`.

## Decision rules

Round-1 support requires test macro F1 >= 0.80, every class recall >= 0.65, a matched shuffled-label control near 0.25, and no strong classification from pre-action rest windows. This round cannot establish continuous-angle robustness because the frozen irregular-angle challenge set has not yet been collected.
