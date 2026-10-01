# Rotation Invariance project rules

This is an independent research repository. All maintained documentation, source comments, user-facing strings, and generated report text must be written in English.

- Work only within this project unless a task explicitly spans projects. Preserve existing user changes and the parent repository's state.
- Preserve original recordings and rejected takes. Before batch processing, record input, cropping, channels, filters, labels, output location, and overwrite protection.
- Current files/data and fresh verification take precedence over saved artifacts, historical summaries, designs, and proposals, in that order. Do not describe a plan as an executed result or a historical score as a fresh rerun.
- Four-direction presegmented classification does not establish continuous detection, handwriting transcription, or open-vocabulary recognition.
- Validate physical sensor order. Cyclic reindexing is not the same transformation as moving sensors to new anatomical locations.
- Fit preprocessing only on training data. Split by complete position/recording groups; exclude marked duplicate windows.
- Compare true-label results with matched training-block label permutations and report prediction diversity.
- Repeatedly used validation is exploratory. The existing C170 test has been exposed; future confirmatory claims require a new untouched holdout.
- Report class order, per-class recall, per-position metrics, channel-map strata, seeds, selected epochs, source/data hashes, and the precise test-access boundary.
- Separate mathematical invariance, synthetic numerical checks, same-session physical transfer, rewearing, cross-day, and cross-person evidence.
- Historical source snapshots and manifests retain original run hashes. English conversion changes text bytes; consult docs/evidence/english_conversion_manifest.json before provenance checks. Never rewrite a historical hash to imply the translated source was the executed source.
- Preserve incomplete/failed artifacts and exclusive test-start markers. Do not overwrite runs or silently relabel failures as complete.

Start with README.md and docs/README.md. Detailed design, procedure, results, reproduction, and release documentation live in docs/.
