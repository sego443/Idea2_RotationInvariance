# C170 train/validation spectral and amplitude diagnostic

This is a descriptive signal check, not a classifier fit. It uses only the fixed training and validation windows after loading the approved processed NPZ and immediately selecting those splits. `load_dataset` decodes the complete compressed NPZ and checks all `X_raw` values for finiteness; test rows were not indexed for filtering, statistics, or predictions. Original data files were not modified.

Sampling rate is 250 Hz (`X_raw` = 500 samples × 4 channels; 2 s), so Nyquist is 125 Hz. The dataset README says the stored `X_raw` windows are unfiltered EXG values; 0.5–12 Hz and a 75-sample smoothed envelope were used only for action localization. The inspected raw header records sample rate and OpenBCI GUI Cyton board but no hardware filter settings. Thus any hardware/front-end filtering is undocumented here, and frequencies above 125 Hz are not represented in these samples.

Welch-free Hann periodograms were computed after subtracting each window/channel mean, at 0.5 Hz resolution. Band fractions use summed 0.5–125 Hz power as the denominator; power is averaged over S0–S3. Existing preprocessing was also applied only to train/validation with the 0.5–40 Hz order-3 Butterworth SOS `sosfiltfilt`; the shared amplitude scale was fitted on train only and reused on validation.

Counts: train 879; validation 320; train+validation shape after duplicate exclusion 1199×500×4. The registered-test split was not processed or summarized. Train-derived shared scale: 11123.037109375.

## Overall split summaries

Median [Q1, Q3] across windows; RMS is in raw EXG units before filtering, and filtered normalized RMS is dimensionless.

| Split | Raw RMS | 0.5–40 power fraction | 0.5–4 | 4–8 | 8–15 | 15–40 | 40–125 | Filtered 40–125 fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6.28e+04 [3.7e+04, 1.16e+05] | 99.9 [99.7, 100] | 94 [89.7, 96.4] | 4.55 [2.73, 7.49] | 0.94 [0.402, 1.99] | 0.158 [0.0599, 0.383] | 0.0612 [0.0146, 0.3] | 0.00383 [0.00102, 0.0147] |
| validation | 6.82e+04 [3.8e+04, 1.04e+05] | 100 [99.8, 100] | 94.1 [91.1, 96.6] | 4.7 [2.44, 6.66] | 0.707 [0.383, 1.24] | 0.0931 [0.047, 0.205] | 0.0314 [0.00991, 0.221] | 0.00179 [0.000969, 0.0135] |

## Validation positions and directions

Values are median [Q1, Q3] over windows in the single recording block for each distance × direction cell. These cell-level ranges are descriptive; the 20 within-block windows are not counted as 20 independent recording replicates.

| Position mm | Direction | n | Raw RMS | 0.5–40 fraction | 0.5–4 | 4–8 | 8–15 | 15–40 | 40–125 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | UP | 20 | 8.92e+04 [6.78e+04, 1.08e+05] | 100 [100, 100] | 95 [92.1, 96.3] | 3.75 [2.85, 6.66] | 0.966 [0.569, 1.27] | 0.237 [0.199, 0.314] | 0.0204 [0.0149, 0.0245] |
| 5 | DOWN | 20 | 6.77e+04 [5.57e+04, 8.23e+04] | 99.9 [99.9, 100] | 88.2 [86.8, 90.1] | 8.95 [7.89, 10.8] | 1.82 [1.56, 2.2] | 0.366 [0.303, 0.525] | 0.0622 [0.0364, 0.079] |
| 5 | LEFT | 20 | 3.46e+04 [3.08e+04, 3.79e+04] | 100 [100, 100] | 97.9 [97, 98.2] | 1.8 [1.56, 2.61] | 0.184 [0.131, 0.291] | 0.034 [0.0185, 0.0434] | 0.0325 [0.028, 0.0378] |
| 5 | RIGHT | 20 | 3.8e+04 [3.47e+04, 4.21e+04] | 100 [100, 100] | 97.5 [95.8, 98.3] | 1.91 [1.37, 3.32] | 0.373 [0.203, 0.502] | 0.0222 [0.0136, 0.0701] | 0.0266 [0.0212, 0.0319] |
| 40 | UP | 20 | 1.05e+05 [1.03e+05, 1.12e+05] | 100 [100, 100] | 94.8 [94.4, 95.2] | 4.33 [3.89, 4.73] | 0.774 [0.614, 0.85] | 0.103 [0.0885, 0.117] | 0.0136 [0.00824, 0.023] |
| 40 | DOWN | 20 | 8.38e+04 [6.82e+04, 1.14e+05] | 99.3 [99.1, 99.5] | 89.6 [88.3, 90.8] | 7.49 [6.25, 8.52] | 1.61 [1.29, 2.36] | 0.519 [0.218, 0.616] | 0.731 [0.526, 0.918] |
| 40 | LEFT | 20 | 5.1e+04 [4.41e+04, 5.48e+04] | 98.1 [97.5, 98.4] | 95.1 [92.9, 95.8] | 2.31 [1.82, 3.86] | 0.201 [0.153, 0.518] | 0.0318 [0.016, 0.117] | 1.87 [1.57, 2.51] |
| 40 | RIGHT | 20 | 4.73e+04 [3.86e+04, 5.51e+04] | 96.1 [94.6, 97.3] | 88.7 [85.6, 91.2] | 5.99 [5.14, 7.12] | 0.574 [0.453, 0.731] | 0.098 [0.0529, 0.136] | 3.87 [2.75, 5.4] |
| 90 | UP | 20 | 1.02e+05 [9.86e+04, 1.13e+05] | 100 [100, 100] | 93.5 [89.8, 95.5] | 5.41 [3.51, 8.25] | 1.18 [0.845, 1.74] | 0.113 [0.0562, 0.176] | 0.0311 [0.0257, 0.0367] |
| 90 | DOWN | 20 | 1.34e+05 [1.3e+05, 1.4e+05] | 100 [100, 100] | 93.1 [92.6, 93.7] | 5.91 [5.47, 6.36] | 0.867 [0.711, 0.945] | 0.0768 [0.0634, 0.0989] | 0.0327 [0.0299, 0.0346] |
| 90 | LEFT | 20 | 8.56e+04 [8.28e+04, 9.98e+04] | 99.9 [99.9, 99.9] | 98.1 [97.5, 98.4] | 1.46 [1.21, 1.96] | 0.289 [0.253, 0.406] | 0.0535 [0.0444, 0.0644] | 0.0893 [0.0773, 0.109] |
| 90 | RIGHT | 20 | 3.62e+04 [3.24e+04, 4.05e+04] | 99.3 [99.1, 99.4] | 92.9 [91.7, 93.5] | 5.8 [4.98, 7] | 0.479 [0.351, 0.709] | 0.0841 [0.0552, 0.142] | 0.696 [0.571, 0.868] |
| 130 | UP | 20 | 2.21e+05 [2.12e+05, 2.39e+05] | 100 [100, 100] | 98.1 [97.6, 98.4] | 1.44 [1.1, 1.84] | 0.436 [0.297, 0.595] | 0.0513 [0.0344, 0.0793] | 0.00168 [0.00138, 0.00307] |
| 130 | DOWN | 20 | 8.81e+04 [7.82e+04, 9.3e+04] | 100 [100, 100] | 91.8 [89.6, 92.7] | 6.27 [5.27, 7.97] | 1.61 [1.44, 2.26] | 0.331 [0.252, 0.529] | 0.00919 [0.00764, 0.0105] |
| 130 | LEFT | 20 | 1.73e+04 [1.5e+04, 1.92e+04] | 100 [100, 100] | 95.4 [94, 96.2] | 3.88 [3.22, 4.99] | 0.724 [0.636, 0.898] | 0.122 [0.0872, 0.144] | 0.00135 [0.000817, 0.00217] |
| 130 | RIGHT | 20 | 2.42e+04 [1.92e+04, 2.81e+04] | 100 [100, 100] | 91.6 [90.2, 94.1] | 7.37 [5.11, 8.65] | 0.763 [0.675, 1.31] | 0.0584 [0.0338, 0.0785] | 0.00119 [0.000798, 0.00163] |

## Position and class checks

Exploratory tests use one median per recording block: Friedman tests compare directions with distance as the matched block, and Spearman tests look for monotone distance trends within each direction. Holm adjustment is applied within each listed split family. Within every position, directions were recorded in the fixed order UP, DOWN, LEFT, RIGHT, so apparent direction effects may include order or recording drift. There is only one recording block per position × direction, so causal separation is unsupported; validation has only four positions. Treat these p-values as exploratory, not confirmatory.

**train: direction effect (Friedman, Holm-adjusted)**

| Feature | p raw | p Holm |
|---|---:|---:|
| raw_rms | 1.973e-05 | 0.0001184 |
| raw_log10_pass_power_0p5_40 | 1.44e-05 | 0.0001008 |
| raw_share_pct_0.5-4 | 0.1381 | 0.2513 |
| raw_share_pct_4-8 | 0.1257 | 0.2513 |
| raw_share_pct_8-15 | 0.07075 | 0.2122 |
| raw_share_pct_15-40 | 0.01968 | 0.0984 |
| raw_share_pct_40-125 | 0.03234 | 0.1293 |

**validation: direction effect (Friedman, Holm-adjusted)**

| Feature | p raw | p Holm |
|---|---:|---:|
| raw_rms | 0.01944 | 0.136 |
| raw_log10_pass_power_0p5_40 | 0.01944 | 0.136 |
| raw_share_pct_0.5-4 | 0.04399 | 0.176 |
| raw_share_pct_4-8 | 0.03356 | 0.1678 |
| raw_share_pct_8-15 | 0.05033 | 0.176 |
| raw_share_pct_15-40 | 0.2123 | 0.4246 |
| raw_share_pct_40-125 | 0.3476 | 0.4246 |

Per-distance/channel-map metadata is included in `summary.json`; 90/130 mm are on the later channel-map regime, confounded with position and recording order. This diagnostic cannot attribute differences causally to distance or channel map.

## Reading the 0.5–40 filter result

The table and JSON compare unfiltered raw band shares with shares after the existing 0.5–40 Hz filter. Removed 40–125 Hz energy cannot be recovered by the present time-domain model because it never receives those frequencies. A raw high-band fraction alone is not evidence that the band is movement-related or class-useful; class/position separability and acquisition noise remain to be established on independent blocks. The 250 Hz sampling rate imposes a 125 Hz Nyquist limit; this analysis cannot say anything about higher frequencies.

See `summary.json` for per-train/validation class summaries, all train and validation position × direction cells, absolute band powers, filtered/raw retention ratios, channel maps, and block-level p-values.

## Actual 0.5–40 Hz filter amplitude retention

The raw 0.5–40 Hz spectral fraction above is not the Butterworth filter's retention. This table uses paired windows and reports RMS after the existing filter but before shared-scale normalization, divided by raw per-window demeaned RMS. Median [Q1, Q3], in original signal units for RMS and dimensionless for the ratio:

| Split/group | Raw demeaned RMS | Filtered RMS before scale | Filtered/raw RMS |
|---|---:|---:|---:|
| Train | 62,820 [37,035, 116,319] | 44,478 [25,570, 85,115] | 0.718 [0.624, 0.789] |
| Validation | 68,186 [37,980, 104,013] | 41,623 [23,107, 69,969] | 0.639 [0.564, 0.710] |
| Validation 5 mm | 48,566 [36,951, 79,182] | 29,748 [21,741, 48,751] | 0.626 [0.574, 0.671] |
| Validation 40 mm | 67,090 [49,370, 103,200] | 42,763 [28,579, 64,535] | 0.633 [0.550, 0.690] |
| Validation 90 mm | 99,353 [63,905, 121,401] | 59,756 [39,440, 90,337] | 0.705 [0.641, 0.737] |
| Validation 130 mm | 54,000 [19,242, 135,583] | 30,170 [11,136, 80,698] | 0.604 [0.526, 0.681] |

The train-only shared scale was reproduced exactly at `11123.037109375` by preserving the source NPZ dtype through filtering. This correction changes none of the raw PSD shares above.
