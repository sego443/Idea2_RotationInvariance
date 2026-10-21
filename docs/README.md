# Documentation index

Updated: 2026-10-21. The October 1 documentation release translated existing text and consolidated historical evidence. Dated reports below distinguish fresh exploratory runs from historical results and proposed evaluations.

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

## W02 next study and exploratory development

- [W02 rewear acquisition protocol](data_collection/w02_rewear_protocol.md)
- [v21 wear-calibrated classifier design and pre-registered endpoints](designs/2026-10-12-v21-wear-calibrated-classifier.md)
- [v21 development on existing data (exploratory)](results/2026-10-12-v21-development-on-existing-data.md)
- [C170 calibration pilot (exploratory)](results/2026-10-12-c170-calibration-pilot.md)
- [Fresh v22 time-window and normalization ablations (81 completed CUDA runs)](results/2026-10-12-v22-window-normalization-ablation.md)
- [W02 Part A and v21 Stage-1/Stage-2 freezes (2026-10-14; no W02 scores)](results/2026-10-14-w02-part-a-and-v21-freezes.md)
- [W02 grid segmentation and rejected-take deletion (2026-10-15; no W02 scores)](results/2026-10-15-w02-grid-segmentation.md)
- [W02 sealed holdout and switch to exploratory development (2026-10-15)](results/2026-10-15-w02-seal-and-exploratory-switch.md)
- [W02 exploratory development: frozen baseline and covariance recipes (2026-10-15; development wears only)](results/2026-10-15-w02-exploratory-development.md)
- [Fresh W02 startup curve (2026-10-21; 98 real rows and 490 matched null rows, development wears only)](results/2026-10-21-w02-startup-curve.md)
- [Fresh W02 stream families (2026-10-21; eight jobs, 1,563 real rows and 7,815 original control aggregates, development wears only)](results/2026-10-21-w02-stream-families.md)
- Freeze manifests: `evidence/v21_freeze/v21_freeze_S1_c170.json` (S1, C170 only) and `evidence/v21_freeze/v21_freeze_S2.json` (S2); `v21_freeze_S1.json` is superseded ([index](evidence/v21_freeze/README.md))

## Application module

- [Four-key agent-assisted PC control, MVP design (2026-10-17; proposed, not implemented)](designs/2026-10-17-four-key-agent-pc-control-mvp.md)

## Historical protocols and diagnostics

- [W01 collection](data_collection/round1_rotation_poc_task.md) and [feasibility results](results/2026-08-27-round1-rotation-invariance-feasibility.md)
- [C170 exploratory acceptance](results/2026-09-27-c170-exploratory-validation-acceptance.md)
- [Original temporal ring plan](plans/2026-09-27-c170-ring-representation.md)
- [Latent-angle design](designs/2026-09-28-latent-angle-temporal-classifier.md), [execution plan](plans/2026-09-28-c170-latent-angle-cuda-experiments.md), and [launcher guide](usage/c170-latent-angle-cuda-launcher.md)
- [Ring v2 plan](plans/2026-09-28-c170-ring-cross-position-cuda.md) and [protocol review](results/2026-09-28-c170-ring-v2-runner-protocol-review.md)
- [Train/validation quality](results/2026-09-28-c170-train-validation-quality-diagnostic.md), [scale commutation](results/2026-09-29-c170-c4-scale-commutation-check.md), and [later v3 checkpoint check](results/2026-09-29-c170-v3-sharedscale-checkpoint-c4-check.md)

Additional dated harmonic, Nyquist, temporal-phase, and kernel designs/plans are in designs/ and plans/. Their dates and original authorization statements describe history; an unchecked plan is not an executed experiment. The consolidated results and saved JSON evidence take precedence for current status.
