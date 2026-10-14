# W02 exploratory development: frozen baseline and covariance recipes

Written 2026-10-15. **Everything in this record is exploratory.** All numbers come from the nine W02 development wears (W01–W06, W08, W09, W11), which were used repeatedly to choose features, classifiers and settings. The sealed wears W07, W10 and W12 ([seal record](2026-10-15-w02-seal-and-exploratory-switch.md)) were not opened. Every scripted entry point refuses a dataset that contains a sealed wear, and the window re-cut filters sealed plan rows before any raw file is read.

Common facts:
- **Data.** Development dataset `outputs/w02_dev_20261015/dataset.npz` (SHA-256 `f5ec8530…a764`). It holds 1,080 strokes, of which 648 are scored body strokes (12 per recording, 3 per class). Segmentation: [grid record](2026-10-15-w02-grid-segmentation.md).
- **Class order:** UP, DOWN, LEFT, RIGHT (0–3).
- **Participant and hardware:** one participant (P01) on one hardware setup, all within two consecutive days.
- **Unit of analysis:** the wear.
- **Nulls.** For calibrated recipes, the null permutes the support labels within each support set. For trained models, it permutes the training labels by recording block.

## 1. Frozen v21 models on the development wears (exploratory baseline)

These are the frozen S1/S2 manifests, scored once each with the unchanged frozen scorer on the development dataset. Outputs: `outputs/v21_S1_c170_devbaseline_20261015`, `outputs/v21_S2_devbaseline_20261015`.

| | S1 (9 wears, 648 body strokes) | S2 (W08, W09, W11; 216 body strokes) |
|---|---|---|
| E1 (B2, own prefix) | 0.829 [0.798, 0.864]; recall U/D/L/R 0.86/0.88/0.86/0.71 | 0.883 [0.752, 0.986]; recall 0.98/0.96/0.96/0.65 |
| E1 matched nulls (5) | 0.717–0.757 | 0.842–0.911 |
| E1b (B1, other prefixes) | 0.681 | 0.736 |
| E2 (label-free) / E3 (zero) | 0.401 / 0.408 | 0.434 / 0.445 |
| E4 (training-free nearest centroid) | 0.830 | 0.871 |
| Gate (exploratory reading) | passed | failed (RIGHT recall) |

The matched nulls come close to the real models, and E4 equals E1. The in-recording benefit therefore comes from template matching against the labeled prefix, not from direction knowledge learned in training.

## 2. Covariance recipes (`src/w02_cov_fewshot.py`)

**Features.** Band-pass 0.5–12 Hz, then the 28×28 covariance of the four channels stacked at lags 0–30 samples in steps of 5. Each covariance is mapped to the tangent space at a reference covariance. **Scoring:** `scripts/analysis/evaluate_w02_cov_recipes.py --mode dev`, leaving one wear out for the recipes that need training wears. Output: `outputs/w02_explore_20261015/recipes_dev_v1` (results SHA-256 `e74c22e9…8544`).

| Recipe | Labels from the test wear | Macro-F1 [wear bootstrap 95%] | Recall U/D/L/R | Per-wear F1 range | Null mean (5 seeds) |
|---|---|---:|---|---|---:|
| `rec_nc` | own 8-stroke prefix (2/class) | **0.963** [0.925, 0.988] | 0.98/0.93/0.99/0.95 | 0.83 (W02) – 1.00 | 0.253 |
| `rec1_borrowed` | first prefix stroke of each class (4 strokes) | 0.933 [0.896, 0.966] | 0.90/0.90/0.99/0.94 | 0.82 (W02) – 1.00 | 0.240 |
| `wear_logreg` | all 6 prefixes of the wear (48; includes later recordings) | 0.980 [0.964, 0.995] | 0.98/0.98/0.99/0.96 | 0.94 – 1.00 | 0.251 |
| `zero_logreg` | none (angle and unlabeled strokes of the wear only) | 0.546 [0.465, 0.636] | 0.38/0.64/0.64/0.54 | 0.37 – 0.82 | — |

Prediction counts are balanced for every recipe (148–173 per class).

**Recipe details:**
- `rec_nc` needs no training data.
- `rec1_borrowed` learns a within-class covariance on the training wears (PCA 30, then Ledoit-Wolf). It refines the class means with the recording's unlabeled body strokes.
- `zero_logreg` cyclically shifts the channels by round(−θ/90). It re-centers each wear at the mean of its unlabeled strokes, which is transductive over the wear.

## 3. What was tried

All settings were tuned on the development wears. Logs are in `outputs/w02_explore_20261015/`.

- **Features and classifiers** (`explore_w02_fewshot.py`, `cov_tuning_v2.log`):
  - Raw-waveform features (the E4 family, 9 variants) reach at most 0.91 with the recording's own prefix.
  - Lagged covariance in tangent space reaches 0.94–0.96. Dense lags (0–30 in steps of 5) score best; most variants fall within ±0.02 of each other.
  - LDA fails with 8 support strokes (0.18–0.56) and works with 48.
  - Using other recordings' prefixes (as E1b does) reaches 0.95 with covariance features. So E1b's 0.68 came from the B1 model, not from cross-recording drift.
- **Calibration size:** 1 stroke per class gives 0.873 with nearest centroid and 0.933 with the borrowed metric. 2 per class give 0.963.
- **Transfer without calibration** (`explore_w02_zerocal.py`, leaving one wear out):

  | Method | Macro-F1 |
  |---|---:|
  | Global tangent space | 0.35 |
  | Per-wear re-centering (unlabeled wear strokes) | 0.46 |
  | Re-centering on the recording's unlabeled prefix | 0.44 |
  | Re-centering on REST windows | 0.34 (hurts) |
  | Angle derotation + re-centering, harmonic or cyclic, either sign | 0.53–0.57 |
  | Adding all of C170 to training, re-centered per position | 0.48–0.52 (no help) |

- **Rotation check** (`pairwise_shift_diag.log`): for each ordered pair of development wears, a single-wear classifier was applied under all four cyclic shifts. The best shift equalled the tape-predicted shift in 28/72 pairs with one sign convention and 33/72 with the other; chance is about 18. Pairs worn at the same tape angle transferred at only 0.30–0.71 (W02/W08, W03/W09, W05/W11). The measured angle explains part of the wear-to-wear change. Contact, tightness and angle error account for much of the rest.
- **Label-free clustering** (`explore_w02_labelfree.py`, `cluster_naming_v1.log`):
  - Per-wear k-means on tangent features, with clusters named by Hungarian assignment from the zero model, gives 0.68–0.78.
  - Results are bimodal per wear: 0.93–1.00 when naming succeeds, 0.0 or 0.5 when it swaps clusters.
  - Cluster purity under oracle naming is 0.928; W05 (0.61) and W06 (0.76) are impure.
  - Naming with 4 labeled strokes per wear reaches 0.919.
  - Supervised 2-shot beats the purity ceiling.
- **Ensembles and errors:**
  - Summing nearest-centroid distances over 5 feature variants: 0.958, no gain.
  - Only 2 of 648 body strokes are wrong under at least 4 of 5 methods, so prompt-following errors are not the limit.
- **Windows** (`explore_w02_windows.py`, re-cut from raw, development wears only):

  | Window | 2-shot F1 | Wear-level F1 |
  |---|---:|---:|
  | 2.0 s centred on the slot energy maximum (current) | 0.963 | 0.980 |
  | 2.5 s centred | 0.954 | 0.977 |
  | 1.5 s centred | 0.946 | 0.978 |
  | Full 3 s slot | 0.957 | 0.977 |
  | Full slot shifted 0.5 s later | 0.951 | 0.974 |

## 4. Interpretation and limits

- With 2 labeled strokes per direction from the same recording, a training-free covariance classifier separates the four directions on these development wears (0.963). With 1 stroke per direction plus a metric borrowed from other wears, it reaches 0.933.
- **These are exploratory.** The settings were chosen on these wears. They cover one participant, two days and re-wears at six band angles. They concern presegmented strokes only: no continuous detection, handwriting or open vocabulary.
- Recognition with no labels from the new wear remains poor (≤0.57). Label-free clustering is unreliable. The rotation-invariance hypothesis is only partly supported by the angle check (§3).
- The calibrated recipes do not use the band angle at all. Their performance therefore does not show rotation invariance; it shows that per-wear calibration works.

## 5. Proposed sealed test (not run)

`scripts/analysis/evaluate_w02_cov_recipes.py --mode sealed --output-dir <new dir>` would score all four recipes once on W07/W10/W12. It uses all development wears as training wears and writes `EVALUATION_STARTED.json` exclusively before reading the sealed data.

Proposed primary endpoint: `rec_nc`, with pass criteria macro-F1 ≥ 0.85 and every class recall ≥ 0.75 on the 216 sealed body strokes. The other three recipes and the matched nulls would be reported as secondary. The recipe source and scorer must be committed before the run. The thresholds and the decision to end development belong to the operator.
