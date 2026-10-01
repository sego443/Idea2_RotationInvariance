# Standalone Git and remote release

Updated2026-10-01. This directory is an independent Git repository with its own .git, main branch, ignore rules, English project instructions, README, code, tests, protocols, metadata and lightweight evidence. Parent Git index/history and other subprojects are not reset, cleaned or committed by this task.

## Packaging

Included: root runners/launchers/analyzers, src, tests, English documentation, collection plans/prompts, processed metadata, requirements, and docs/evidence summaries with checksums. Excluded: raw sessions/rejected takes, dataset NPZ, generated outputs/checkpoints/source snapshots, caches/environments, temporary files and local secrets. Excluded artifacts remain on disk.

Portable evidence is a curated review bundle, not all run artifacts. It includes frozen F2 summary and selected prediction tables; full checkpoint replay and raw segmentation require a separate bundle. Historical machine paths and session identifiers in manifests are provenance, not runtime dependencies.

## English migration

Readable Chinese project text was converted, including locally retained historical snapshots and report generators. Original source/artifact hash fields remain unchanged; docs/evidence/english_conversion_manifest.json records conversion hashes. Translated snapshots cannot silently pass an original byte-identity audit. A recoverable original-text ZIP is stored outside the project and excluded from this English release.

Raw signals, arrays and checkpoints were not edited. Five derived waveform figures were regenerated with English labels using their original selected train/validation/excluded-position windows. Their numerical traces match the original calculation exactly; no registered-test rows were preprocessed. Originals are backed up outside this project. The sixth figure already uses English. See [figure provenance](evidence/figure_conversion_manifest.json) and [figure gallery](FIGURES.md).

## Connect and publish

No remote URL/account was supplied or inferred, and no remote repository was created or pushed. Once a destination exists, run from this project root with the actual URL:

```bash
git rev-parse --show-toplevel
git status --short
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

The top-level path must be this subproject. If origin exists, inspect git remote -v and use the intended destination. Local independent history does not rewrite parent history. The parent may still track historical files beneath this directory; this task does not remove them or convert the project into a submodule.

A license has not been chosen. No authorship, affiliation, credentials, reuse rights or raw-data publication permission is invented. Choose a license and distribute approved raw/checkpoint bundles separately when appropriate.
