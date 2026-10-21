# Fresh W02 family G evidence

Executed 2026-10-21 from commit 10ff6cd, G job only. See the [English report](../../results/2026-10-21-w02-stream-G-robustness.md) for the negative/tradeoff findings and oracle-intervention assumptions. No sealed W02/C170 waveform was accessed.

- `row_metrics.csv`: all 1,156 real rows, true/predicted class counts, recalls, wear/angle/recording/position/minute strata and available label-action totals.
- `original_null_metrics.csv`: all 5,500 original part-1 nulls replayed from features, including additional F1/recall/count metrics. Replayed per-stroke arrays remain local.
- `summaries_and_checks.json`, `selected_rows.json`: all 448 summaries and raw original check flags, full intervention rows and selected detailed comparisons.
- `action_costs.csv`, `check_assignments.json`: newly reconstructed action/redo costs for all 56 intervention trajectories and all 18 original Hungarian assignments.
- `check_block_controls.json`, `feedback_block_controls.json`: additional post-sweep training-block check cases (90) and recording-block teacher controls (210), kept distinct from original G outputs. No matched check+redo stream null was added.
- `preflight.json`, `run.json`, `child_exit.json`, `launcher_exit.json`, `launcher.log`: actual command, source/data/plan hashes, settings and process outcomes.
- `verification.json`, `tests.log`: verification scopes and four focused passing tests. Replayed stream functions are shared with the supplied implementation; mask/metric checks are separate.
- Two scientific PNG figures compare usage and intervention tradeoffs. Redo scores exclude extra calibration actions and use a different query set; draws and synthetic scenarios reuse data.
- `manifest.json`: published and retained-local hashes. Full results, signals, prediction/log artifacts and helpers stay in ignored outputs. This is not a standalone waveform replay bundle.

The high detection rate does not make the check useful when it rejects two thirds of correct calibration sets. Residual updates do not dominate the existing EMA, and feedback gains assume immediate correct user labels. All findings remain exploratory.
