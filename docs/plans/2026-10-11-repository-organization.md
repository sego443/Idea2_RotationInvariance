# Repository Organization Implementation Plan

**Goal:** Organize the existing repository and remove disposable files without losing research evidence.

**Architecture:** Group command-line entry points by responsibility under `scripts/`. Keep reusable models in `src/`, tests in `tests/`, and stable data/output paths. Consolidate execution plans into `docs/plans/`.

**Scope:** The user's repository cleanup request; this project's AGENTS.md governs preservation and English documentation. Execute in the current checkout because local ignored data and outputs are part of the inventory. Do not commit automatically.

## Preservation constraints

- Preserve all recordings, datasets, results, failed runs, test-access markers, and frozen source snapshots.
- Preserve historical evidence manifests and their hashes; record source changes in a separate migration ledger.
- Delete only reproducible caches and operating-system metadata. Identical frozen snapshots and portable evidence copies are intentional redundancy.
- Update maintained commands, imports, source paths, and tests. Dated protocols retain original commands and use the migration map.

## Tasks

- [x] Inventory files, duplicate hashes, imports, and protected artifact hashes.
- [x] Move runners, launchers, analyses, preparation, and synthetic validation scripts into responsibility directories; update live imports and paths.
- [x] Consolidate plan locations and provide a directory guide, script catalog, and old-to-new migration ledger.
- [x] Remove disposable cache/metadata files after checking all resolved paths stay inside this repository.
- [x] Verify existing tests, direct CLI imports, source paths, documentation links, historical artifact checks, and protected file hashes.

## Review focus

Direct script execution must resolve the project root independently of the working directory. Imports must use a single qualified module identity. New source snapshots must use current paths; historical analyzers must continue accepting saved original paths. No cleanup may change recordings or test-access markers. A moved dated plan must still be reachable from the documentation index.

## Verification outcome

All 143 existing tests passed with the local ML2025 Python environment (PyTorch 2.6.0+cu124, NumPy 1.26.4, SciPy 1.15.2). Temporary pytest 8.4.2 dependencies and test artifacts were kept in the project scratch directory and removed afterward. The historical Python 3.10/PyTorch environment produced the same 127 passes and 16 failures before and after the moves; failures concerned temporary-directory access, uint16 conversion, and weights-only loading.

All 66 entry points imported successfully. Six CLI help checks covering all five responsibility directories passed from outside the project root. Maintained Python files parsed, source paths/import targets resolved, live documentation links resolved, and moved function/class signatures stayed unchanged. All 4,768 protected files retained their pre-cleanup hashes; all 47 indexed portable artifacts matched their existing hashes.

Historical v3 snapshot verification still encounters the pre-existing translated source hash difference documented in `docs/evidence/english_conversion_manifest.json`. No historical snapshot or declared hash was changed to bypass this check. The snapshot-reader test checks both path layouts and rejects tampered source.

The migration moved 66 Python scripts and one dated plan. Cleanup removed 272 disposable files (5,521,482 bytes). No source or research artifact was deleted.
