# C170 harmonic-Nyquist v5 design

## Objective

Test whether the four-channel Nyquist component omitted by v4's first-harmonic representation limits C4-invariant classification. This is a capacity-matched, validation-only experiment; it makes no claim of continuous physical rotation invariance.

## Fixed representation

Input is the v3-filtered, per-window/per-channel demeaned `[B,500,4]` signal, divided by the single training-fitted shared scale `11123.037109375`. The original data, channel order, labels, split, and scale fit are unchanged.

The first ten feature channels are exactly v4 `harmonic_gram`: `asinh(s)`, `log1p(p)`, and real/imaginary normalized first-harmonic lag products for lags 1, 4, 16, and 64, where `s=(x0+x1+x2+x3)/4`, `a=(x0-x2)/2`, `b=(x1-x3)/2`, `z=a+ib`, and `p=a²+b²`.

Define the Nyquist component only as an intermediate, `n=(x0-x1+x2-x3)/4`; it is not supplied as a raw feature. Append, in order:

1. `log1p(n²)`.
2. Four scalar lag correlations `n_t*n_(t-τ)/sqrt((1+n_t²)(1+n_(t-τ)²))` for `τ=(1,4,16,64)`, with a zero prefix and no circular temporal wrap.
3. The real and imaginary components of `n*z² / (sqrt(1+n²)*(1+p))`.

This gives 17 inputs. Under a cyclic sensor shift by `k`, `z→i^k z` and `n→(-1)^k n`. All 17 features are invariant because the sign of `n` cancels the sign of `z²` in the coupling. When `z` receives an ideal common phase `e^(iφ)` while `n` stays fixed, the coupling receives `e^(i2φ)`; the coupled representation therefore gives up v4's ideal first-harmonic U(1) invariance. The experiment tests the C4/Nyquist tradeoff only.

## Arms and matched model

All arms use the same 17-input network, initialization rule, and optimization:

| Variant | Active features | Training targets |
|---|---|---|
| `nyquist_coupled` | 0–16 | Real labels |
| `nyquist_uncoupled` | 0–14; coupling slots 15–16 zero | Real labels |
| `gram_control` | 0–9; Nyquist slots 10–16 zero | Real labels |
| `nyquist_shuffle` | 0–16 | Complete training recording-block labels permuted within position |

The convolutional backend is unchanged from v4 except its input channels increase from 12 to 17: 24-channel stem, three depthwise-separable residual blocks with dilations 1/2/4, adaptive mean pools at lengths 1/2/4, 24-dimensional population temporal standard deviation, dropout 0.2, and a 192-to-4 classifier. It has 7,012 trainable parameters. The `gram_control` arm is retrained with this same capacity; historical v4 `harmonic_gram` is reported only as a paired-seed historical reference, not treated as the matched control.

## Frozen training protocol

Use seeds `(17,29,43,61,89)`, the unchanged 879/320/320 train/validation/registered-test position split, exclude 70 mm and the known duplicate row, and draw each training sample exactly once per epoch in 20 batches. Preserve the v3 shared training scale and channel mapping. Optimize weighted per-sample cross-entropy with label smoothing zero using AdamW (`lr=0.001`, `weight_decay=0.001`), up to 160 epochs with patience 25. Select by validation macro-F1, breaking ties with lower validation cross-entropy. Train the four arms with identical per-seed parameter initialization and batch index order.

## Evaluation boundary

The primary `nyquist_coupled` gate is mean validation macro-F1 at least 0.80 and mean validation recall at least 0.65 for each class. Report paired seed differences against both `nyquist_uncoupled` and `gram_control`, all four validation positions, shuffle prediction diversity, and the historical v4 `harmonic_gram` comparison. Do not score or use the registered test split unless the validation gate passes and a later explicit release is issued. The loader may decode the complete compressed NPZ and perform its approved whole-array finite integrity check; only train/validation rows enter preprocessing, scale fitting, training, validation selection, or predictions.

## Claims and limits

This is a small C4-invariant representation experiment that restores an invariant involving the sampled Nyquist component. Four sensors omit and alias spatial harmonics, so the constructed features do not guarantee arbitrary continuous SO(2) or real wrist-rotation invariance. The coupling deliberately breaks ideal `z`-only continuous phase invariance when `n` is fixed. Results are specific to this dataset and split until tested on genuine held-angle recordings.
