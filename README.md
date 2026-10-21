# Four-Channel Wristband Rotation Invariance

Research code for recognizing four presegmented movement directions from four circumferential piezoelectric wristband sensors while studying changes in wearing position. The project compares ordinary temporal classifiers, cyclic ring representations, harmonic/phase features, kernel methods, sensor adapters, and fixed or learned fusion.

**Research status:** the rotation-generalization acceptance criterion has not been met. The frozen C170 F2 ensemble scored **0.826872 macro-F1 on the repeatedly used validation set** and **0.674897 on its registered test**, failing the test gate of macro-F1 >= 0.80 and every class recall >= 0.65. Later experiments remain exploratory. Mathematical C4 invariance and ideal phase invariance do not establish robust physical rotation, rewearing, cross-day, or cross-person performance.

## Start here

- [Documentation index](docs/README.md)
- [Detailed experimental design](docs/EXPERIMENT_DESIGN.md)
- [Execution and artifact procedure](docs/EXPERIMENT_PROCEDURE.md)
- [Results, controls, and limitations](docs/EXPERIMENT_RESULTS.md)
- [Fresh v22 window and normalization ablations (exploratory)](docs/results/2026-10-12-v22-window-normalization-ablation.md)
- [Fresh W02 startup curve: 0-4 calibration actions and sequential recognition (exploratory)](docs/results/2026-10-21-w02-startup-curve.md)
- [Fresh W02 stream families: adaptation, feedback, drift and usage robustness (exploratory)](docs/results/2026-10-21-w02-stream-families.md)
- [Environment and reproduction](docs/REPRODUCIBILITY.md)
- [Standalone Git and remote release](docs/RELEASE.md)

## Repository layout

```text
scripts/runners/      Experiment runners, including versioned research methods
scripts/launchers/    Batch launchers and recovery workflows
scripts/analysis/     Saved-artifact analysis, evaluation, plotting, benchmarks
scripts/data/         Recording segmentation and dataset preparation
scripts/validation/   Synthetic numerical invariance/equivariance checks
src/                  Reusable model, representation, preprocessing, loss modules
tests/                Unit and experimental protocol tests
docs/                 Living guides and dated research records
data/                 Original recordings and processed dataset metadata
outputs/              Local run bundles and frozen snapshots, excluded from Git
```

See the [repository guide](docs/REPOSITORY_LAYOUT.md) for preservation rules and the [script catalog](scripts/README.md) for entry points. Execution plans now share `docs/plans/`. The [migration ledger](docs/evidence/reorganization_manifest_20261011.json) maps historical root-level script names to current paths; frozen run bundles retain their original source paths and hashes.

The task is four-way classification of **already segmented** UP/DOWN/LEFT/RIGHT strokes. These experiments do not implement word recognition, an open vocabulary, or automatic movement-boundary detection in a continuous stream.

## Environment

Historical formal training used Python 3.10, PyTorch 1.13.1+cu117, and an RTX 2080 Ti. Create a dedicated environment; no sibling Idea2 project is required by the core runners:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements.txt
```

This installs the historical numerical-library versions and a CPU-compatible PyTorch distribution. CUDA reproduction requires the separately selected CUDA build. Raw C170 segmentation currently uses Python 3.11's hashlib.file_digest and needs a separate compatible preprocessing environment; see [reproduction details](docs/REPRODUCIBILITY.md). Environment installation and new training are not performed by this documentation release.

## Data and a validation-only example

Raw recordings, NPZ datasets, checkpoints, and complete run directories are distributed separately. This Git repository includes collection instructions, processed metadata, and lightweight saved result evidence. Place the C170 bundle at `data/processed_c170_20260926/dataset.npz` or provide `--dataset` explicitly. Verify its provenance against the supplied manifests before training.

From the project root, using a new output directory:

```bash
python scripts/runners/run_c170_experiment.py --phase validation-only --dataset data/processed_c170_20260926/dataset.npz --output-dir outputs/c170_new_validation --train-distance-mm 0 10 20 30 50 60 80 100 120 140 160 --val-distance-mm 5 40 90 130 --test-distance-mm 15 25 110 150 --confirm-physical-order
```

The frozen split excludes 70 mm and the copied B019 window, yielding 879/320/320 train/validation/registered-test windows. The flag confirms a verified physical sensor order; it is not a substitute for checking a new dataset's mapping. The legacy loader decodes/finite-checks the complete NPZ, but this phase trains and selects checkpoints on train/validation without test prediction or scoring.

The C170 registered test was already evaluated for frozen F2. Re-running it creates no new independent evidence. Use a newly collected untouched holdout for future confirmatory evaluation.

## Tests and research practice

```bash
python -m pytest tests
```

Some existing tests require the local data bundle or invoke computational model checks. Consult the reproduction guide before running the complete suite. Preserve raw inputs and existing outputs; fit scales only on training data; compare real labels with matched block-level nulls; report prediction diversity and per-position results; freeze inference before evaluating a new holdout.

All maintained documents, comments, CLI messages, and generated report text use English. Dated documents preserve what was known or planned at their original date; the consolidated report records later outcomes. Historical source hashes refer to pre-translation files, with the migration recorded in [the English conversion ledger](docs/evidence/english_conversion_manifest.json).

## Remote publication

This directory has its own Git repository. See [RELEASE.md](docs/RELEASE.md) for the included/excluded artifacts and commands to attach your remote. No hosting account, repository URL, data-publication permission, or license has been assumed. A license has not been selected; choose one before offering reuse rights.
