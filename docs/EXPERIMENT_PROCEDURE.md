# Experimental execution and artifact procedure

Updated 2026-10-01. This describes existing pipelines and historical execution; it does not claim a fresh training run.

## 1. Input identity and segmentation

Record session, participant/wrist/wear, hardware/gain/firmware, sample rate, original channel columns and S0–S3 identity. Preserve raw and rejected files. Hash inputs and retain collection plans/prompts; a historical machine path alone does not identify data.

C170 metadata must agree with X_raw `(1600,500,4)`, 250 Hz, 80 complete/56 short files, one duplicate and two verified EXG maps. Fixed split counts are 879/320/320 after excluding all 70 mm and the B019 copy. Position labels are not measured physical angles.

prepare_c170_session.py validates metadata/channels/prompts, localizes centers and checks spacing/bounds, then writes an exclusive new directory. Localization uses 0.5–12 Hz/75-sample envelope; output windows retain raw samples. Record source rows, detected counts, trimming and the prescribed copy/source. Its hashlib.file_digest call requires Python 3.11+; use a separate preprocessing environment from historical Python 3.10 training. Do not edit the NPZ to fit a model or silently repair labels/counts.

## 2. Preprocessing and access boundary

Use third-order Butterworth 0.5–40 Hz SOS/sosfiltfilt and per-window/channel demeaning. Fit version-specific sensor scales, shared scalar, RMS statistics or feature scaler on train only, store in checkpoints and reuse on validation. Shared scalar and unequal sensor scales have different C4 properties; record the selected version. channel_map supports ordering/strata, not prediction; block/time/repetition are not predictors.

The legacy loader decompresses the full NPZ and finite-checks all arrays including test. Validation-only runners subsequently preprocess/model/score train and validation only. Some diagnostics decode then immediately subset without the global finite check. Report the actual boundary: no test scoring is different from zero compressed-container reads.

## 3. Frozen run matrix

Record model version/arm, true/null targets and permutation, seeds, split, preprocessing, optimizer, epoch budget/patience, losses, sampler, inference rule, source/data hashes and new output path. Typical formal neural configuration: seeds 17/29/43/61/89, AdamW lr/weight_decay .001, max160/patience25. Read each version's manifest: v17 varies regularization; v18 fixes 120-epoch terminal checkpoints.

Group sampling and weights by original position/direction. The corrected sampler covers 879 unique windows once: 19 batches of44 plus one of43. Circular training-position Voronoi weights yield sample weight w_position/(4*n_group), summing to one; each of20 updates uses 20*sum(weight*sample loss). Target shuffling changes complete direction blocks within positions without changing group membership, sampling or weights. Validation targets remain true.

## 4. Smoke, throughput and training

Use separate new directories for real-data one-epoch smoke, fixed-work throughput probes and formal matrices. Inspect counts, uniqueness, finite loss, output shapes, reload agreement and completed artifacts. Extra probe seeds are throughput-only, never formal model-selection candidates.

Historical ring/latent probes compare identical eight-task workloads across 2/4 and sometimes8 workers; distinguish startup from steady epochs. Later batches commonly use <=6 workers, one CPU thread, with RAM admission used+2 GiB <=24 GiB. Choose concurrency from timing/resources. Historical resource readings are not new measurements in this release.

Log epoch loss/CE, F1/recalls/diversity, position/channel strata, updates, timing, selected/actual epochs. Usually select highest validation F1 then lower CE. Save each finished run immediately rather than waiting for the matrix.

## 5. Artifact contract

| Artifact | Meaning |
|---|---|
| config and fingerprint | Frozen version/arm/seed/data contract |
| learning_curve.csv | Epoch losses, validation metrics, timing, updates |
| best_checkpoint.pt / best_weights.npz | Selected state and fitted preprocessing |
| validation_predictions.csv | Sample identity, labels, member outputs, predictions |
| run_manifest.json | Data/source/artifact hashes, counts, runtime, selection and status |
| RUN_COMPLETE.json | Final marker after required artifacts complete |
| source_snapshot/ | Historical dependencies/environment with original hashes |

Names vary by version. A marker alone is insufficient: validate unique planned arm/seed coverage, configuration/hash consistency and prediction identities. Interrupted directories are not completed runs.

## 6. Saved-artifact analysis and inference freeze

Use the matching analyze_/summarize_ script. Keep per-seed, family-vote and fused-system metrics separate and report planned controls/failures. An old-validation maximum is exploratory.

F2 uses complex_gram5, learned_power10 and v7 group_true. Members argmax raw logits/KRR scores; each neural family takes majority labels with lowest-class-ID ties. Two neural family winners and KRR cast three equally weighted votes with the same tie rule. It is hierarchical hard voting. The saved correction preserves the earlier inaccurate description and reproduces320 validation rows; freeze implemented inference rather than an informal name.

v18 uses11 leave-one-training-position-out folds: three neural experts×two seeds×11=66 neural runs, plus11 KRR folds. Fit fusion from OOF training predictions and evaluate old validation afterward. Fixed120-epoch fold endpoints avoid old-validation early stopping, but the validation set's development history still limits independence.

## 7. Test and recovery

C170 registered test was opened for F2 on2026-09-29. Do not rerun it during documentation checks or call it unused. Future confirmation requires new untouched data and frozen model/scales/classes/seeds/inference/scorer. The legacy final-test creates an exclusive started marker; retain it and partial artifacts on failure. Directory markers do not replace data identity/exposure checks across runs.

For training recovery, preserve failed directories and use supported hash/config-checked resume or a separate recovery root. Windows interruptions, cuda/cuda:0 differences and one corrupt v18 NPZ were recorded. The damaged fold original was retained and only that fold rerun; summaries used valid artifacts. Recovery does not authorize rewriting historical source hashes.

## 8. Reporting and translation provenance

Report design versus execution, identity/exclusions/deviations, actual preprocessing/split/model/training/selection/fusion, runtime/recovery, true/null diversity, overall/class/position results, gate and confounds. Distinguish synthetic invariance from physical transfer and W02/v20 proposals from completed work.

English conversion changes snapshot/config text bytes. Existing hashes remain historical. The [conversion manifest](evidence/english_conversion_manifest.json) maps originals to English files. Strict original replay requires the original artifact bundle; fresh runs must hash current source. Raw recordings, arrays and checkpoints are preserved.
