# C170 Shared-Scale C4 Equivariance Experiment (v3)

## Purpose

The v2 validation results did not pass the frozen macro-F1 and per-class recall gate. The next experiment isolates one preprocessing change: replace four channel-specific robust scales with one scalar shared across all channels. Per-channel scaling is not equivariant to cyclic channel rolls and can break the exact C4 symmetry used by the ring readout. This is a single-factor correction to a preprocessing structural gap, not a claim of a novel algorithm contribution.

## Frozen protocol

- Keep the existing order: order-3 Butterworth 0.5–40 Hz `sosfiltfilt` on each window along time, then subtract each trial/channel temporal mean.
- Compute exactly one scale from **training windows only**: `max(median(abs(centered_train)), float32_epsilon)`, reducing across training samples, time, and all four channels. Apply this same scalar to train and validation inputs.
- Use the existing `RingRepresentationClassifier(variant="classification_only", n_classes=4)` architecture, equivalent to v2 `ring_ce`; do not use SupCon.
- Run two variants over seeds 17, 29, 43, 61, and 89: `shared_scale_ce` with true training targets, and `shared_scale_shuffle` with complete training recording-block label permutation. Validation labels remain true.
- Preserve v2 train/validation/registered-test/excluded positions, data loader behavior, original-group 879-sample sampler, 20 weighted CE updates per epoch, AdamW (`lr=0.001`, `weight_decay=0.001`), 160-epoch maximum, patience 25, and validation macro-F1 then lower CE checkpoint selection.
- Do not change raw or processed source data. The existing loader may decode the full NPZ and finite-check all `X_raw` values, including registered-test waveforms. Test windows must not be filtered, demeaned, used to estimate scale, passed to the model, predicted, or scored.
- Preserve the metadata-rich checkpoint and additionally save the selected model `state_dict` as a numeric-only `.npz` artifact. Load selected weights with `numpy.load(..., allow_pickle=False)`, validate exact keys/shapes/dtypes, then call strict `load_state_dict`; never fall back to unrestricted pickle loading.

## Evidence and limits

The implementation must verify that filtering, temporal demeaning, and shared-scalar normalization commute with all four cyclic raw-channel rolls on deterministic synthetic windows, and that the existing ring readout produces matching logits/projections on CUDA. This establishes only the discrete C4 raw-tensor property of this pipeline; it does not establish invariance to arbitrary continuous SO(2) rotations, true wearer rotation, or real-device held-out performance. Report all four validation positions (5, 40, 90, and 130 mm), all four class recalls, and paired per-seed comparison with v2 `ring_ce`; do not select only positions that improve. The final test split remains sealed unless a later validation gate explicitly authorizes evaluation.
