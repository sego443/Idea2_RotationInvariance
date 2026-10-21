# W02 naming robustness evidence, 2026-10-21

Fresh exploratory run of `scripts.analysis.explore_w02_naming_robustness` from code commit `c74b3dfc4d014c5130684ad49dc524cb52000c83`, with unchanged numerical source. [English report](../../results/2026-10-21-w02-naming-robustness.md).

| Artifact | Purpose |
|---|---|
| [results.json](results.json) | Original 1,600 real rows, five matched nulls per row and 1,088 summary rows |
| [predictions.npz](predictions.npz) | Original 9,600 integer prediction arrays and `y`; no waveform signals |
| [sample_metadata.csv](sample_metadata.csv) | Derived index-to-recording/wear/angle/prefix/label mapping, single acquisition-map stratum |
| [row_metrics.csv](row_metrics.csv) | Derived real metrics, recalls/counts, collisions/chosen-k counts and all null accuracy/F1/prediction counts |
| [summary.csv](summary.csv) | Original summary rows in CSV format; random-drop seeds averaged |
| [confusion_matrices.json](confusion_matrices.json) | Derived real confusion matrices, true rows/predicted columns in UP/DOWN/LEFT/RIGHT order |
| [run.json](run.json) | Original start record, conditions, source/data hashes and seeds |
| [preflight.json](preflight.json) | Input/cropping/channels/filters/labels/outputs/overwrite checks, environment and provenance before batch processing |
| [exit.json](exit.json) | Exit code 0 and finish timestamp |
| [stdout.log](stdout.log) | Original complete console output |
| [packages.txt](packages.txt) | Package snapshot of the local environment, including inherited system packages |
| [tests.log](tests.log) | Focused existing covariance-recipe tests: 2 passed |
| [verification.json](verification.json) | Complete saved-prediction metric/mask replay and baseline comparisons |
| [manifest.json](manifest.json) | Artifact byte counts and SHA-256 hashes; navigation README excluded |

Original local run: `outputs/w02_explore_20261015/naming_robustness_v1`, with adjacent preflight/log/exit/package/verification records. All original run bytes were preserved. The dataset's raw signal arrays, raw recordings, private data checkout, virtual environment and caches are not included in this bundle. Sealed W07/W10/W12 were neither supplied nor opened.

Each NPZ vector has 1,080 elements. Prediction arrays use -1 for unscored strokes and 0–3 for UP/DOWN/LEFT/RIGHT. Keys are `method|condition|prefix|k_mode|subset`, with `|nullN` appended for a null seed; `y` is the true class vector. `sample_metadata.csv` uses the same index order. All real/null scored masks exclude prefix strokes. The source NPZ contains no per-recording cluster assignments, so support-collision counts and chosen-k histograms can only be checked for ranges/totals during prediction replay.

The 16 real `full/with/fixed4` rows reproduce corresponding previous k-shot metrics. Null support permutations have a different RNG schedule and are newly computed. Macro-F1 is averaged over true classes present in the scored subset; absent-class recall is null. No confidence intervals or confirmatory test claims are attached.
