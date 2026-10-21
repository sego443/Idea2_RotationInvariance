# Fresh W02 stream-family evidence

Executed on 2026-10-21 from commit 4e6e10a: eight completed development jobs, 1,563 real rows, 7,815 original control aggregates, a diagnostic and 75 separately labeled supplementary recording-block controls. See the [English report](../../results/2026-10-21-w02-stream-families.md) for findings, startup chronology and boundaries.

- `row_metrics.csv`: all real rows, including per-class/wear/angle/position/recording/minute metrics, null means/ranges and actual action distributions. `actual_macro_f1_with_abstention` counts no-action separately. Generic D macro-F1/counts reproduce the runner's artificial wrong-class substitutions and should not be interpreted as actual commands.
- `null_aggregates.csv`: all original controls and available prediction counts. Most original null predictions were not saved, so this does not imply complete null trajectory replay.
- `usage_true_counts.csv`: four true-class denominators for every family F usage row; combine with the recalls/counts in `row_metrics.csv` when comparing skewed scenarios.
- `selected_rows.json`, `family_summaries.json`: selected original detailed results and checked C/D summaries plus F mean metrics.
- `drift_diagnostics.json`: exact E output. Transfer recording keys 1-5 denote T2-T6. These are label-assisted/batch diagnostics, not causal startup results.
- `D_cost_replay.csv`, `D_block_nulls.json`: all D real interaction-cost replays and 75 extra post-sweep recording-block teacher controls, respectively. Original D controls used per-stroke random teachers.
- `preflight.json`, `run_records.json`, `child_exits.json`, `launcher_exit.json`, `launcher.log`: source/data/plan hashes, settings and actual process outcomes.
- `verification.json`, `supplement_verification.json`, `primary_verification.json`, `tests.log`: verification scopes and fresh checks. Ten selected primary null trajectories were replayed; no claim covers all original null trajectories.
- Two PNG figures show matched adaptation ablations and artificial usage scenarios. Means over repeated draws do not provide independent-replication confidence intervals.
- `manifest.json`: hashes of published evidence, report and all locally retained run/verification artifacts.

Full original result JSON, per-stroke predictions, logs, package list and helper scripts remain under ignored outputs. Raw accepted recordings and sealed waveforms were not modified or opened. This compact evidence is not a complete standalone waveform replay bundle. Use a fresh root and logs for another run.
