# Round-one rotation-invariance feasibility results

Date: 2026-08-27. Source: data/OpenBCISession_2026-08-27_14-37-15. One participant/session/wear; four presegmented stroke directions and held-out nominal angles. Historical report translated on 2026-10-01; no training rerun.

## Conclusion

Small models fit the data well but failed the predefined rotation-generalization gate. Mean training accuracy was 94.0%–99.3%. The validation-selected harmonic model had test macro-F1 0.478 ± 0.040. The orbit model's post hoc test macro-F1 was 0.670 ± 0.026, below 0.80; its minimum mean class recall was 0.427, below 0.65. Shuffled-label macro-F1 was 0.229 ± 0.071, accuracy 0.247. Pre-movement rest macro-F1 was 0.182 (raw) and 0.238 (harmonic), without strong simple rest-file leakage.

The data supports some cross-angle transferable movement information and moderate recognition at some angles. It does not establish reliable arbitrary continuous-angle recognition. This failed result did not justify irregular-angle challenges or downstream text-entry expansion.

## Data and quality audit

Raw recordings were preserved. Select one non-rejected take for each B001–B048; preserve/exclude 22 .rejected takes and unlabeled OpenBCI-RAW-2026-08-27_15-55-27.txt. Metadata confirms 250 Hz and eight EXG channels; retain 0–3. Locate five approximately periodic peaks per file and extract two-second movements and one-second preceding-rest controls. Apply 0.5–40 Hz filtering, remove window temporal means, and fit channel scales on training angles only.

Frozen train/validation/test angles are 0/60/120/180/240/300, 30/150/270, and 90/210/330 degrees. Validation selects family/early stopping; test does not select architecture, preprocessing, or epochs.

- 48 accepted movement files yield 240 movements, 60 per class. Every file has five centers; maximum interval coefficient of variation 0.088.
- Longest identical-value run is three rows; no sustained flatline or obvious digital-rail saturation observed.
- 31/48 accepted movement files are shorter than the 20-second protocol minimum. Five strokes remain complete, so analysis retains them and discloses this deviation.
- Completed plan, zero-degree photo, wrist side, and adjustment log are absent; full wear metadata cannot be independently verified.
- Start-rest filtered RMS `[407,416,188,184]`; end-rest `[7577,17750,9534,8420]`, about 19–51 times higher. Movement contamination, contact/cable changes, and hardware drift cannot be separated.
- Sequential angles confound time. Near-chance rest reduces simple leakage concern but does not eliminate changing movement responses.

## Five-seed results

Seeds 17/29/43/61/89; ± denotes seed variability, not a cross-person confidence interval.

| Model | Train macro-F1 | Validation macro-F1 | Test macro-F1 | Test accuracy | Minimum mean test class recall |
|---|---:|---:|---:|---:|---:|
| Raw CNN | 0.985 ± 0.030 | 0.437 ± 0.062 | 0.495 ± 0.106 | 0.523 | 0.240 |
| Four-shift cyclic-orbit CNN | 0.940 ± 0.071 | 0.441 ± 0.039 | 0.670 ± 0.026 | 0.683 | 0.427 |
| Ring Fourier-magnitude CNN | 0.993 ± 0.015 | 0.531 ± 0.033 | 0.478 ± 0.040 | 0.493 | 0.213 |

Validation selects harmonic. The higher orbit test score is post hoc and cannot replace the selected model. Orbit accuracy by validation angle: 30° 0.38, 150° 0.36, 270° 0.62; by test angle: 90° 0.75, 210° 0.47, 330° 0.83. Each is 30° from a training angle, yet performance varies widely.

| Criterion | Observation | Decision |
|---|---|---|
| Unseen-angle macro-F1 >= 0.80 | Selected 0.478; post hoc best 0.670 | Failed |
| Every recall >= 0.65 | Best post hoc minimum mean 0.427 | Failed |
| Shuffled accuracy about 25% | 0.247 | Met |
| Rest cannot reliably predict action | F1 0.182–0.238 | Met |
| Irregular challenge F1 >= 0.75 | Not collected | Not evaluated |

## Proposed follow-up

Defer irregular 1/55/123/203/317 degree challenges. Collect W02 with randomized/Latin-square-balanced angle order and randomized directions; add strict rest/RMS checks every three angles and stop if drift changes by orders of magnitude. Retain five repetitions per combination and independent wear identity. Freeze current preprocessing/three models before external W02 evaluation. If orbit remains useful only at a few angles, investigate more sensors, coarse-angle/calibration input, or a narrower supported wearing range. These are recommendations, not executed results.

## Reproduction

From the standalone root, with separately provided raw data:

```bash
python run_round1_experiment.py --session-dir data/OpenBCISession_2026-08-27_14-37-15 --output-dir outputs/round1_new_run
```

Use a nonexistent output directory. Historical metrics, predictions, segments, quality audit, and checkpoints reside in the local outputs/round1_rotation_poc_2026-08-27 bundle. See [portable results](../EXPERIMENT_RESULTS.md).
