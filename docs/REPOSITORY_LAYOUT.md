# Repository Guide

Updated: 2026-10-11. This organization change moves entry points and removes disposable files. It does not rerun or replace research experiments.

## Where to work

| Location | Contents and rules |
|---|---|
| [Project README](../README.md) | Research status, setup, and a validation-only example |
| [scripts/](../scripts/README.md) | Entry points grouped into runners, launchers, analysis, data preparation, and synthetic validation |
| `src/` | Reusable model, signal-processing, representation, and loss modules; version numbers identify distinct research methods |
| `tests/` | Existing unit and experiment-protocol tests |
| [docs/README.md](README.md) | Living experiment, reproduction, release, and figure guides |
| `docs/data_collection/` | Acquisition plans and prompts |
| `docs/designs/` | Dated model designs |
| `docs/plans/` | All execution plans, including the formerly separate temporal-ring plan |
| `docs/results/` | Dated diagnostics and reviews |
| `docs/usage/` | Command-specific instructions |
| `docs/figures/` | Report figures |
| [docs/evidence/](evidence/README.md) | Portable saved results and provenance ledgers |
| `data/OpenBCISession_2026-08-27_14-37-15/` | Original recordings, preserved including rejected takes |
| `data/processed_c170_20260926/` | Dataset and its segmentation/source metadata |
| `outputs/` | Local experiment bundles, logs, checkpoints, failed runs, and frozen source snapshots |

## Finding local artifacts

Stable paths in `data/` and `outputs/` are retained because recorded manifests, launchers, analyzers, and dated reports refer to them. Their long names identify method, purpose, and date. Use these filename groups to navigate:

| Name pattern under `outputs/` | Role |
|---|---|
| `*formal*`, `*distance_holdout*`, `*round1*`, `conditional_directions*` | Training/evaluation bundles; completeness must be checked from their markers |
| `*recovery*`, `*retry*` | Recovery/retry evidence; separate attempts must not be merged or discarded |
| `*smoke*`, `*probe*`, `*speed_probe*` | Historical setup/performance investigations |
| `*summary*`, `*diagnostic*`, `*fusion*`, `*selection*` | Saved analyses and selection artifacts |
| `*.log`, `*.stdout.log`, `*.stderr.log` | Execution logs associated with the corresponding attempt |
| `source_snapshot/` within a run | Frozen executed source and its original hashes |

Start from [the evidence index](evidence/README.md) and [experimental results](EXPERIMENT_RESULTS.md) when looking for a reported outcome. Names alone do not prove successful execution. Scratch drivers already stored under `outputs/` remain historical artifacts, not maintained entry points.

## Cleanup decisions

Removed files are Python bytecode caches, pytest caches, and `.DS_Store` metadata. Maintained source and documentation contained no byte-identical duplicates. Similar versioned scripts implement distinct experiments and are referenced by later methods.

Repeated source files in frozen snapshots and portable evidence copies serve reproducibility and distribution purposes. They remain intact. Raw recordings, processed datasets, checkpoints, logs, incomplete/failed runs, exclusive test-start markers, and original evidence manifests remain intact as well.

The [organization ledger](evidence/reorganization_manifest_20261011.json) records old/new paths, source hashes, and every removed disposable file. It supplements the earlier English-conversion and release ledgers without changing their historical hashes. Dated protocols retain their historical commands; resolve root-level filenames through this ledger before reuse. Living guides use current paths.

## Keeping the repository organized

Put new entry points in the matching `scripts/` directory and import them with qualified names such as `scripts.runners.run_c170_experiment`. Put reusable implementations in `src/`. Keep execution plans in `docs/plans/`, reports in `docs/results/`, and generated runs in a fresh `outputs/` directory. Avoid root-level scripts, copied working variants, and checked-in caches. Preserve a frozen snapshot for each executed protocol rather than treating it as disposable duplication.
