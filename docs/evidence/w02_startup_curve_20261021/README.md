# Fresh W02 startup curve evidence

Executed 2026-10-21 on the nine development wears only. No sealed W02 waveform or C170 data was accessed. See the [dated report](../../results/2026-10-21-w02-startup-curve.md) for the reordered-startup assumption, metrics, controls and limitations.

- `results.json`: exact runner output, all 98 real rows and 19 summaries, with per-class/wear/angle/recording/position metrics and five null results per row.
- `summary.csv`, `row_metrics.csv`: independently checked readable aggregate/real metrics.
- `null_metrics.csv`: all 490 matched-null rows, including independently recomputed macro-F1, per-class recall and prediction counts.
- `preflight.json`, `run.json`, `exit.json`: actual inputs, environment, hashes, source revision, defaults and process status.
- `verification.json`, `tests_retry.log`: prediction replay and four focused passing tests; initial test-launch failure is documented and retained locally.
- `startup_curve.png`: scientific plot of the checked metrics. Lines average repeated subset/draw analyses, with no confidence-interval claim.
- `manifest.json`: exact hashes of included evidence and excluded local prediction/log/verification artifacts.

Full per-stroke predictions and sample metadata remain under ignored `outputs/w02_explore_20261015/`. They require the separate local data bundle; this compact bundle does not claim to permit full waveform replay. Raw accepted recordings were not modified. Re-execution requires a fresh output directory and log. Model PCA randomness was not seeded in the executed source.
