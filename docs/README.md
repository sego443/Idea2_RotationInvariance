# Documentation index

Updated: 2026-10-01. This release translates existing project text and consolidates saved evidence; it does not rerun training or collect new data.

| Document | Purpose |
|---|---|
| [Repository layout](REPOSITORY_LAYOUT.md) | Directory roles, cleanup policy, migration, and local artifact navigation |
| [Script catalog](../scripts/README.md) | Commands grouped by responsibility and method family |
| [Experimental design](EXPERIMENT_DESIGN.md) | Questions, acquisition, representations, controls, split, gates, and proposed independent follow-up |
| [Experimental procedure](EXPERIMENT_PROCEDURE.md) | Data preparation, training, selection, run artifacts, recovery, and scoring boundaries |
| [Experimental results](EXPERIMENT_RESULTS.md) | Historical chronology, batch comparisons, frozen F2 test, nulls, failures, and limitations |
| [Reproducibility](REPRODUCIBILITY.md) | Environments, commands, data requirements, portability limits, and verification |
| [Release](RELEASE.md) | Independent Git scope, evidence/data packaging, provenance migration, and remote setup |
| [Figures](FIGURES.md) | English waveform overlays and the historical W01 result plot |
| [September 30 closeout](TRAINING_REPORT_20260930.md) | Dated final training summary |
| [Evidence index](evidence/README.md) | Portable summaries and their original local sources |

## Historical protocols and diagnostics

- [W01 collection](data_collection/round1_rotation_poc_task.md) and [feasibility results](results/2026-08-27-round1-rotation-invariance-feasibility.md)
- [C170 exploratory acceptance](results/2026-09-27-c170-exploratory-validation-acceptance.md)
- [Original temporal ring plan](plans/2026-09-27-c170-ring-representation.md)
- [Latent-angle design](designs/2026-09-28-latent-angle-temporal-classifier.md), [execution plan](plans/2026-09-28-c170-latent-angle-cuda-experiments.md), and [launcher guide](usage/c170-latent-angle-cuda-launcher.md)
- [Ring v2 plan](plans/2026-09-28-c170-ring-cross-position-cuda.md) and [protocol review](results/2026-09-28-c170-ring-v2-runner-protocol-review.md)
- [Train/validation quality](results/2026-09-28-c170-train-validation-quality-diagnostic.md), [scale commutation](results/2026-09-29-c170-c4-scale-commutation-check.md), and [later v3 checkpoint check](results/2026-09-29-c170-v3-sharedscale-checkpoint-c4-check.md)

Additional dated harmonic, Nyquist, temporal-phase, and kernel designs/plans are in designs/ and plans/. Their dates and original authorization statements describe history; an unchecked plan is not an executed experiment. The consolidated results and saved JSON evidence take precedence for current status.
