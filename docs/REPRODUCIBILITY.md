# Environment and reproduction

Updated 2026-10-01. Reported training scores are historical artifacts. Documentation verification does not launch training or score test waveforms.

## Environments

Formal C170 training: Python 3.10.6, NumPy 1.23.5, SciPy 1.10.1, pandas 1.5.3, PyTorch 1.13.1+cu117, CUDA 11.7, RTX 2080 Ti. requirements.txt pins the numerical/training versions, selects matplotlib3.7.1 for Python3.10, and adds pytest. Generic torch==1.13.1 installation does not guarantee the historical CUDA build; select the appropriate wheel and record driver/library versions.

Raw preparation calls hashlib.file_digest and needs Python3.11+, with compatible NumPy/pandas/SciPy; torch is not required by scripts/data/prepare_c170_session.py. Use a separate preparation environment from processed-data training. This release preserves that existing runtime distinction.

```bash
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -r requirements.txt
python -c "import sys,numpy,scipy,pandas,torch; print(sys.version); print(numpy.__version__,scipy.__version__,pandas.__version__,torch.__version__)"
```

Record environment output for each new run. A clean installation was not performed by this documentation release.

## Data and artifacts

Git excludes raw sessions, dataset.npz, checkpoints and generated outputs. It includes processed metadata and docs/evidence summaries. Review summaries without training data; full checkpoint replay needs the separate artifact bundle.

Provide C170 data/processed_c170_20260926/dataset.npz and verify SHA-256 against original configuration/evidence. For W01, provide data/OpenBCISession_2026-08-27_14-37-15, preserving rejected takes/hashes. Historical absolute paths are provenance; pass current --dataset/--session-dir rather than rewriting historical records.

## CLI and portability

```bash
python scripts/runners/run_c170_experiment.py --help
python scripts/runners/run_c170_ring_cuda_experiment_v2.py --help
python scripts/launchers/launch_c170_ring_cuda_v2.py --help
python scripts/data/prepare_c170_session.py --help
```

Builder help requires its preparation environment. Some launchers embed an operator-specific Python310 path. Ring-v2 and latent-angle launchers accept --python-executable; set it to your interpreter. CUDA launchers require CUDA/nvidia-smi. Use single-run runners directly where a historical launcher is unsuitable. Individual CPU support does not imply a CUDA launcher runs on CPU.

## Reproduction

Use the root README's fixed-split validation command with a nonexistent output directory. Inspect each version's --help and design/plan before launching it; the baseline command does not reproduce every later variant. One-epoch runs are pipeline smoke, not formal scores. Formal budgets and sampler/loss settings must match the actual version.

The registered C170 test was already evaluated for F2. Reproduction does not restore independence. Do not use final-test for routine verification; future confirmation needs new untouched data.

## Existing checks

Full suite, with pytest and its data/model dependencies:

```bash
python -m pytest tests
```

Selected existing contracts also use unittest:

```bash
python -m unittest discover -s tests -p test_c170_ring_cuda_experiment_v2.py
python -m unittest discover -s tests -p test_c170_shared_scale_preprocessing_v3.py
python -m unittest discover -s tests -p test_launch_c170_ring_cuda_v2.py
```

See [release verification](evidence/release_verification.json) for checks actually run, distinguishing them from a full pytest suite, clean install, new training, or historical checkpoint replay.

The October 11 directory reorganization passed all 143 tests in the local PyTorch 2.6.0 environment. The historical environment retained the same 16 failures before and after the moves. See the [organization verification record](plans/2026-10-11-repository-organization.md#verification-outcome) for environments, path checks, preserved artifact hashes, and the existing historical-snapshot limitation. This verifies the reorganization, not a new experimental result or an installation of the pinned historical stack.

## Provenance after translation

Display strings, historical Markdown and configuration prose were converted to English. Numeric arrays/checkpoints/raw recordings remain unchanged. Source comparison ignores string-literal text while retaining numeric constants and syntax structure; exception-match tests were translated with their messages.

Historical source/artifact hashes remain original values. Translated snapshots are not byte-identical executed snapshots; original fail-closed analyzers may reject them. Do not rewrite old hashes or claim complete historical replay succeeded. The [conversion ledger](evidence/english_conversion_manifest.json) records pre/post hashes. A local original-text backup outside the project is excluded from publication. Copied evidence has its own current checksum and original local source path.
