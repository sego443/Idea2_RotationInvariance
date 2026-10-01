# C170 ring C4 and scale commutation check

Date: 2026-09-29. Read-only inspection and CUDA synthetic checks; no dataset, training, or test read.

## Structure

The shared encoder in src/c170_ring_representation.py processes each sensor with shared weights. TemporalRingBlock uses shared self/+1/-1 ring convolutions and per-node GroupNorm, so node tensors are C4 equivariant. Main readout uses node mean/max and directed-edge mean; shifts only reorder these terms. Temporal, classifier, and projection heads are therefore invariant on already standardized tensors in real arithmetic, with float32 reduction errors. Fixed slot_readout has no C4 guarantee.

## Synthetic measurements

Fresh float32 main model on cuda:0, normal synthetic `(4,500,4)` input, shifts 1/2/3: maximum logit error 3.73e-8, projection 8.94e-8. Slot_readout errors: 8.66e-2 and 6.96e-2. These verify structure, not trained or physical performance.

Construct demeaned raw-like X=Z*scales with training scales `[10522.1289,9305.5508,11992.9746,12856.6143]`. Nonuniform fixed per-sensor scales give max difference 0.843656 between roll(X)/scales and roll(X/scales); logit/projection differences 0.00334941/0.00312278. A diagonal scale generally fails to commute with permutation, leaving an end-to-end raw/preprocessing/model C4 gap. These are synthetic differences, not actual signal performance.

Shared scalar 11000 gives zero input difference, logit error <=2.98e-8, projection <=7.45e-8. Algebraically s^-1 I commutes with permutation.

## Historical limitation

The measurements use fresh initialization. Safe checkpoint loading with weights_only=True failed: UnpicklingError: Unsupported operand 71. Automatic approval review rejected weights_only=False because pickle may execute code; the task stopped without bypassing or retrying that load. This report contains no trained-checkpoint C4 result. The [later v3 report](2026-09-29-c170-v3-sharedscale-checkpoint-c4-check.md) is separate evidence.
