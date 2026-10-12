# C170 calibration pilot (exploratory)

Date: 2026-10-12. Script: [diagnose_c170_calibration_pilot.py](../../scripts/analysis/diagnose_c170_calibration_pilot.py). Output: `outputs/c170_calibration_pilot_20261012/calibration_pilot.json` (local). Dataset SHA-256 `d57a7fef61ae4930b3a91fd917ce44ef481a7a84e58b261deb44a3040569e349`, matching the F2 manifests.

## Scope and access boundary

- 15 train and old-validation positions, 1,199 windows. Registered-test positions 15/25/110/150 mm, all of 70 mm and the B019 copy are removed right after loading and never processed.
- Leave-one-position-out (LOPO). Features: 0.5–12 Hz bandpass, 25 Hz decimation, per-window global gain normalization plus log gain (201 dimensions). Model: L2 logistic regression (C=0.05).
- Purpose: compare **protocols** to inform the W02 design. These linear scores are not candidate classifiers and are not comparable with F2. All results are exploratory; the old validation positions have been reused many times.
- Separately, a descriptive per-position channel RMS table, including test positions, was viewed interactively on 2026-10-11. No predictions were made on those positions.

## Results

LOPO mean macro-F1 across 15 held-out positions:

| Question | Arm | Mean F1 | Min F1 |
|---|---|---:|---:|
| Representation | Raw signed channels | 0.515 | 0.23 |
| | Per-wear unsupervised channel gain | 0.501 | 0.18 |
| | Hottest-channel canonicalization (C4-invariant) | 0.378 | 0.11 |
| | Per-wear gain + canonicalization | 0.399 | 0.13 |
| Few-shot calibration (score reps 6–20) | 0 strokes/direction | 0.525 | 0.16 |
| | 1 stroke/direction | 0.653 | 0.33 |
| | 2 strokes/direction | 0.713 | 0.38 |
| | 5 strokes/direction | 0.842 | 0.67 |
| Joint direction × 30° bin, C4 roll augmentation | Marginal over bin | 0.440–0.455 | |
| | Wear-level unlabeled orientation inference | 0.348–0.393 | |
| | Oracle true bin | 0.315–0.376 | |

Matched nulls for calibration (direction blocks permuted within every position, five seeds): mean F1 0.19–0.34 for every shot count.

Wear-level orientation estimates were no better than chance (mean error 3.0–3.1 bins of 30°; chance ≈ 3). Both roll directions were tested.

Training only on the circularly nearest positions beats an equal number of random positions: K=2: 0.491 vs 0.277; K=4: 0.511 vs 0.369; K=6: 0.545 vs 0.417.

## Interpretation and limits

1. **Absolute channel identity carries direction information.** Discarding it through C4 canonicalization costs about 0.14 F1.
2. **Under this protocol, a cyclic channel shift does not stand in for physical rotation.** Roll augmentation hurt even with the true bin given. Possible reasons: sensor spacing or response is unequal, anatomy differs around the wrist, or the position label is not an angle. Orientation-latent models are therefore not the primary W02 approach. The sensor geometry measurement in the W02 protocol is needed before revisiting them.
3. **Few labeled strokes from the new wear help strongly, and nulls stay low.** In C170, however, calibration and scored strokes come from the same recording block, so recording-specific drift may inflate this gain. W02 separates the CAL recording from the TRIAL recordings to measure the real effect.
4. Nearest-position training helps, but in C170 position neighbors are also time neighbors, so angle and drift cannot be separated. W02 randomizes angle order.
5. These results come from one participant and one session, with a linear model.
