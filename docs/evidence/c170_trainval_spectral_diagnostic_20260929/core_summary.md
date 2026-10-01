# Train/validation signal spectrum: core results

- **Scope:** processed dataset `data/processed_c170_20260926/dataset.npz`; after the approved loader decoded the full archive and finite-checked `X_raw`, calculations indexed only fixed train (879 windows) and validation (320 windows). Test waveforms were not filtered, normalized, summarized, or predicted. Raw files were not changed.
- **Rate and source:** 250 Hz, 500 samples (2 s), Nyquist 125 Hz. The dataset README says `X_raw` retains unfiltered EXG windows; the 0.5–12 Hz filter plus 75-sample envelope was used for action localization. The inspected OpenBCI raw header gives 250 Hz/Cyton but no hardware-filter settings. Frequencies above 125 Hz are not available in these samples.
- **PSD definition:** per-window/per-channel temporal mean removed; Hann periodogram, `nfft=500`, 0.5 Hz bins, average band power over S0–S3. Band fractions divide by total 0.5–125 Hz power. RMS and RMS-ratio entries are median [Q1,Q3] across windows; PSD-share entries are median percentages.
- **Amplitude definition:** raw RMS is `sqrt(mean((x - mean_time(x))**2))` over each window’s 500×4 values. Filtered RMS is the same statistic after the existing order-3 Butterworth SOS `sosfiltfilt` 0.5–40 Hz filter and a second per-window/channel de-mean, before shared-scale normalization. The ratio is paired per-window filtered RMS / raw RMS. Shared scale was fit on train only and exactly reproduced as 11123.037109375; it is not applied to the RMS retention figures.

| Group | Raw RMS | Filtered RMS, pre-scale | Filtered/raw RMS | Raw 0.5–40 PSD share | Raw shares 0.5–4 / 4–8 / 8–15 / 15–40 / 40–125 Hz |
|---|---:|---:|---:|---:|---|
| Train | 62,820 [37,035, 116,319] | 44,478 [25,570, 85,115] | 0.718 [0.624, 0.789] | 99.939% | 93.969 / 4.553 / 0.940 / 0.158 / 0.061% |
| Validation | 68,186 [37,980, 104,013] | 41,623 [23,107, 69,969] | 0.639 [0.564, 0.710] | 99.969% | 94.071 / 4.695 / 0.707 / 0.093 / 0.031% |

| Validation position | Raw RMS | Filtered/raw RMS | Raw 0.5–40 PSD share | Raw 40–125 Hz share |
|---:|---:|---:|---:|---:|
| 5 mm | 48,566 [36,951, 79,182] | 0.626 [0.574, 0.671] | 99.971% | 0.029% |
| 40 mm | 67,090 [49,370, 103,200] | 0.633 [0.550, 0.690] | 98.976% | 1.024% |
| 90 mm | 99,353 [63,905, 121,401] | 0.705 [0.641, 0.737] | 99.938% | 0.062% |
| 130 mm | 54,000 [19,242, 135,583] | 0.604 [0.526, 0.681] | 99.998% | 0.0018% |

**Interpretation:** The raw signal’s energy is overwhelmingly below 40 Hz, especially 0.5–4 Hz. The raw 0.5–40 PSD share is *not* the Butterworth filter’s retained amplitude: the paired filtered/raw RMS ratio is 0.718 train and 0.639 validation. The 40–125 Hz share is small in these windows, but this alone does not establish whether that band contains class information. At block level, training shows a direction effect in raw RMS and 0.5–40 power after Holm correction (q≈1.18e-4 and 1.01e-4); band-share effects are not significant after correction. Validation direction tests are not significant after correction (smallest q≈0.136). Within every position, the recording order was UP, DOWN, LEFT, RIGHT; apparent direction effects may include order or recording drift. Each position×direction cell has one recording block, so causal separation is unsupported; cell quantiles in `summary.md`/`summary.json` are descriptive. Position/channel-map/time effects are confounded, especially at 90/130 mm.

Full cell summaries, filtered/raw band-power ratios, channel-map metadata and exploratory block-level test details are in [summary.json](summary.json); the expanded tables are in [summary.md](summary.md).
