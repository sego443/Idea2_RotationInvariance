# C170 temporal phase-invariant v6 design

## Purpose

Test whether learning short temporal filters on the first-harmonic complex signal improves the C170 rotation experiment while preserving exact C4 and ideal global first-harmonic phase invariance. The experiment is a fixed low-capacity comparison; it does not claim a new invariant or real-world continuous rotation performance.

## Frozen input and temporal filter bank

Input is the unchanged v3-filtered, per-window/per-channel demeaned `[B,500,4]` signal normalized by the train-only shared scalar `11123.037109375`. Keep `s=(x0+x1+x2+x3)/4`, `a=(x0-x2)/2`, `b=(x1-x3)/2`, and `z=a+ib`.

Apply eight real causal FIR filters to `a` and `b` with the exact same real weights. Each filter has length 65, no bias, stride 1, and 64 zeros padded on the left. Filter zero is a fixed identity buffer with tap 64 equal to one. The other seven filters begin as delayed deltas at taps `64-τ` for `τ=(1,2,4,8,16,32,64)`. Normalize every filter per forward pass as `w/sqrt(sum(w²)+1e-8)`. In learned arms, the seven filters are trainable; in `fixed_gram`, those same 455 filter parameters are frozen. The identity is never an optimizer parameter.

Let `z_r` be the complex response of filter `r`, and `p_r=|z_r|²`. The ordered 23 features are `asinh(s)`, `log1p(p_r)` for `r=0..7`, then interleaved real/imaginary parts of `q_r=z_0*conj(z_r)/sqrt((1+p_0)(1+p_r))` for `r=1..7`.

Because the same real linear causal filter acts on both components, a constant global phase `z_t→e^(iφ)z_t` commutes with the filter bank; it cancels in `p_r` and `q_r`. A cyclic channel roll also rotates every `z_r` by the same fourth-root-of-unity, which cancels from these features. The FIR bank and the pointwise features are causal: their output at time `t` uses no future sample. The later CNN keeps v4's symmetric convolution padding and whole-window pooling, so the full classifier is not a streaming-causal model.

## Arms and capacity

All four arms use the same 23-input backend and per-seed initialization. The network uses a `23→24` convolutional stem, GroupNorm and SiLU, three depthwise-separable residual blocks with dilations 1/2/4, adaptive average pools at lengths 1/2/4, 24-dimensional population temporal standard deviation, dropout 0.2, and a 192-to-4 classifier.

| Variant | Active features | FIR setting | Training labels |
|---|---|---|---|
| `learned_gram` | All 23 | Seven delayed FIRs trainable | Real labels |
| `fixed_gram` | All 23 | Same seven delayed FIRs frozen | Real labels |
| `learned_power` | Features 0–8 only | Seven delayed FIRs trainable | Real labels |
| `learned_shuffle` | All 23 | Seven delayed FIRs trainable | Whole training recording-block label permutation |

All arms contain 8,763 parameters in total. `fixed_gram` has 8,308 trainable parameters because its 455 FIR weights are frozen; the other arms have 8,763 trainable parameters. This isolates learned temporal weighting from a fixed delayed-delta bank and from power-only readout without adding another head or loss.

## Frozen training and evaluation

Use seeds `(17,29,43,61,89)` and the unchanged train/validation/registered-test position split: 879/320/320 windows, exclude 70 mm and the known duplicate row. Preserve the exact no-replacement sampler, position weights, shared scale, channel maps, and shuffle mapping. Train with the existing per-sample weighted cross-entropy (`20*sum(w_i*CE_i)`, label smoothing zero), AdamW (`lr=0.001`, `weight_decay=0.001`), at most 160 epochs, and patience 25. Select by validation macro-F1, breaking ties with lower validation cross-entropy. Do not use the registered test split for preprocessing, statistics, predictions, scoring, or checkpoint selection.

The primary `learned_gram` validation gate remains mean macro-F1 at least 0.80 and mean recall at least 0.65 for every class. Keep the test split sealed if the gate fails.

## Claim boundary

The design tests a learned temporal filter bank followed by fixed global-phase-invariant readouts, not a first-of-its-kind Fourier phase method. Its ideal invariance assumes one common phase across the complete window and the first-harmonic `z` representation. Four physical sensors alias higher spatial modes; the exact feature symmetry and synthetic checks do not establish arbitrary continuous SO(2), streaming-causal classification, or real wrist-rotation generalization.
