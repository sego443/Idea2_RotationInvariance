# W01 conditional bidirectional experiment: movement to wearing angle, angle to movement

The model receives four-channel movement waveforms and a condition: predict nominal angle given movement, or movement given angle.
Train both directions independently and also train two conditional heads with a shared encoder. Angle estimation uses sin/cos regression and circular absolute error.
Fixed train/validation/test angles: 0/60/120/180/240/300, 30/150/270, and 90/210/330 degrees.
The test angles were evaluated in earlier experiments; every test metric in this report is an exploratory re-evaluation.

| Backbone | Training mode | Validation angle MAE | Test angle MAE | Validation movement macro-F1 | Test movement macro-F1 |
|---|---|---:|---:|---:|---:|
| cnn | angle | 58.6° | 58.8° ± 11.1° | — | — |
| cnn | gesture | — | — | 0.431 | 0.379 ± 0.076 |
| cnn | joint | 66.2° | 70.4° ± 15.9° | 0.452 | 0.456 ± 0.098 |
| graph | angle | 65.4° | 77.3° ± 8.5° | — | — |
| graph | gesture | — | — | 0.444 | 0.306 ± 0.030 |
| graph | joint | 69.7° | 79.5° ± 7.8° | 0.424 | 0.267 ± 0.083 |

Two-seed control with paired training labels shuffled: movement and angle remain paired, while their correspondence to waveforms is broken:

| Backbone | Training mode | Control angle MAE | Control movement macro-F1 | Predicted 30-degree angle bins on test | Predicted movement classes on test |
|---|---|---:|---:|---:|---:|
| cnn | angle | 93.9° | — | 11.4 | — |
| cnn | gesture | — | 0.219 | — | 4.0 |
| cnn | joint | 92.1° | 0.190 | 11.2 | 4.0 |
| graph | angle | 80.5° | — | 10.8 | — |
| graph | gesture | — | 0.144 | — | 4.0 |
| graph | joint | 81.5° | 0.296 | 10.8 | 4.0 |

Without waveform input, a training-angle median candidate conditioned on the known movement gives test angle MAE: 90.0°.
Angles are nominal recording settings in 30-degree increments, rather than continuous measured angles.
The five repetitions at an angle share a recording; sequential angle acquisition may confound session drift and contact state.
Configuration selection and early stopping use validation angles only; configurations, per-seed metrics, predictions, and checkpoints are saved in this directory.

## Known-condition sensitivity (post hoc diagnostic)

Keep test waveforms and true targets fixed; change the supplied movement to the next class or add 90 degrees to the supplied angle. Replay the saved five-seed checkpoints. This diagnostic does not select models.

| Backbone/mode | Correct-movement angle MAE | Wrong-movement angle MAE | Correct-angle movement F1 | Movement F1 after +90 degrees | Unchanged movement predictions |
|---|---:|---:|---:|---:|---:|
| CNN/independent | 58.8° | 62.5° | 0.379 | 0.374 | 96.7% |
| CNN/joint | 70.4° | 74.8° | 0.456 | 0.445 | 97.3% |
| Graph/independent | 77.3° | 80.7° | 0.306 | 0.307 | 99.3% |
| Graph/joint | 79.5° | 81.4° | 0.267 | 0.251 | 95.7% |

Although the movement head receives angle input, most predictions do not change with angle; scores above the shuffled control do not show that known angle improves recognition. The angle head is slightly sensitive to movement conditions, but one session, nominal angles, and acquisition-time confounding limit interpretation.

## Reproduction

From the standalone project root, run `python run_conditional_experiment.py --output-dir outputs/conditional_new_run` using a nonexistent directory. Defaults: tuning max90/patience15, formal max220/patience35. metrics.json records seeds, selected configuration, dataset SHA-256 and all metrics. Existing outputs cannot be overwritten.
