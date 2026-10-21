# W02 k-shot naming evidence, 2026-10-21

Fresh exploratory execution of code commit `2039d19086562d997f4552b29ffff4176b8590bb` on the nine W02 development wears. The experiment source was unchanged. [Report](../../results/2026-10-21-w02-kshot-naming.md).

| Artifact | Purpose |
|---|---|
| [results.json](results.json) | Original full metrics: 64 real rows, five null scores per row, oracle diagnostic and 20 budget summaries |
| [row_metrics.csv](row_metrics.csv) | Derived flat table of class recalls, prediction counts, confidence intervals and each null score |
| [run.json](run.json) | Original start record, dataset/source hashes and seeds |
| [preflight.json](preflight.json) | Input, cropping, channels, filters, labels, counts, environment, source/data provenance and overwrite checks recorded before processing |
| [exit.json](exit.json) | Exit code 0 and end timestamp |
| [stdout.log](stdout.log) | Original experiment output |
| [packages.txt](packages.txt) | Available packages in the project-local environment with inherited system packages |
| [verification.json](verification.json) | Artifact consistency checks, focused tests and initial NumPy DLL test failure |
| [tests_retry.log](tests_retry.log) | Focused test output with one OpenBLAS thread |
| [manifest.json](manifest.json) | SHA-256 and byte counts for the original copied and derived metrics artifacts |

Original local directory: `outputs/w02_explore_20261015/kshot_naming_v1` with adjacent log/preflight/exit/package records. Raw signals and the private data repository are excluded from this evidence bundle. The sealed W07/W10/W12 dataset was not supplied or opened. No historical hash has been rewritten.

The artifact manifest covers the run files and the derived CSV/verification record; this navigation README is outside its scope. The current experiment does not save per-stroke predictions, null diversity or null per-position metrics. Those cannot be reconstructed from the result JSON alone. All real-row per-wear and per-position metrics are retained in `results.json`; one channel-map stratum (`0,1,2,3`) covers all development trial recordings.
