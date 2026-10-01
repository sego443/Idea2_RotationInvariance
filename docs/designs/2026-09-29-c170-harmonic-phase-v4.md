# C170 Harmonic Phase Classifier v4

## Goal

Test whether a small first-harmonic representation with lagged phase relations improves the existing C170 ring CE result. This is a controlled representation comparison, not a claim of a novel invariant or demonstrated physical rotation generalization.

## Frozen data and training protocol

Use the v3 C170 loader, filtering, per-window temporal demeaning, and one train-fitted scalar shared by S0–S3. The recorded scalar is 11123.037109375; v4 recomputes it from the unchanged training split and aborts if the value differs. Validation uses the train scalar. The complete NPZ may be decoded and X_raw finite-checked, including registered-test waveforms, but test samples do not enter preprocessing, scaling, model input, prediction, or scoring.

Keep the fixed split and sample counts: train positions 0, 10, 20, 30, 50, 60, 80, 100, 120, 140, 160 (879 windows); validation positions 5, 40, 90, 130 (320); registered test positions 15, 25, 110, 150 (320); exclude 70 mm and the known duplicate row. Use the 879-example no-replacement sampler, 20 optimizer updates per epoch, position-weighted per-example CE with label smoothing 0, AdamW at 0.001 with weight decay 0.001, up to 160 epochs, patience 25, and seeds 17, 29, 43, 61, 89. Select by validation macro-F1, breaking ties by lower validation CE. Save numeric state arrays in best_weights.npz and load only with allow_pickle=False and strict state-dict validation.

## Representation

For normalized input x with four channels, compute:

- s = (x0 + x1 + x2 + x3) / 4
- a = (x0 - x2) / 2, b = (x1 - x3) / 2, z = a + i b, and p = a² + b²
- first two channels: asinh(s), log1p(p)
- for tau in 1, 4, 16, 64, append Re(c_tau), Im(c_tau), where c_tau,t = z_t conjugate(z_(t-tau)) / sqrt((1+p_t)(1+p_(t-tau))). Set the first tau values to zero; do not wrap around.
- final two channels: asinh(a), asinh(b)

The resulting ordered tensor has 12 channels. The four fixed masks are:

| Variant | Active features | Role |
|---|---|---|
| harmonic_gram | indices 0–9 | Primary candidate; retains lagged phase relations and removes instantaneous phase |
| harmonic_power | indices 0–1 | Tests whether power-only features suffice |
| harmonic_phase | indices 0–11 | Phase-sensitive comparison |
| harmonic_shuffle | indices 0–9 | Primary mask with complete recording-block training-target permutation |

All variants use the same 5,932-parameter network: Conv1d 12→24, kernel 9, stride 2, padding 4, no bias; GroupNorm(4,24), SiLU; three residual blocks with depthwise kernel 7 and dilations 1, 2, 4, each followed by GroupNorm, SiLU, pointwise 24→24 without bias, GroupNorm, residual add and SiLU. Concatenate adaptive average pools at lengths 1, 2, and 4 (168 values) with the full-time population standard deviation (24 values); apply dropout 0.2 and Linear(192,4). Optional diagnostics expose the L2-normalized 192-dimensional pooled representation without a learned projection head.

All arms have identical trainable parameter shapes and receive identical initial weights for each seed. Only the fixed input mask and the shuffle arm's training targets differ.

## Invariance boundary

A cyclic permutation of the four channels induces a discrete C4 phase rotation of z, which leaves s, p, and every lagged product c_tau unchanged. The features therefore support the exact discrete channel-roll property.

An ideal continuous-rotation interpretation requires a strict spatial bandlimit |m| ≤ 1 over equally spaced sensor angles. Under that assumption, one common angular shift multiplies the first harmonic z by a scalar phase and leaves p and c_tau invariant. Four real sensor samples also admit the Nyquist alternating mode m=2; omitted Nyquist content and higher-harmonic aliasing can make z a mixture of modes with different phase shifts. Continuous SO(2) invariance and physical wearable rotation performance are therefore unverified. Reflection maps z to its conjugate and c_tau to its conjugate; retaining Im(c_tau) preserves signed chirality.

## Evaluation and reporting

Run four variants × five seeds. Report mean and population SD for validation macro-F1, each class recall, all four validation positions, selected train-target metrics and train/validation gaps, per-seed paired differences against harmonic_power, harmonic_phase, shuffle, and the prior v2 ring_ce run, plus shuffle prediction counts and unique-class diversity. Apply the existing candidate gate to harmonic_gram: five-seed mean macro-F1 at least 0.80 and mean recall at least 0.65 for every direction. Report observed evidence without choosing a favorable position. Do not evaluate the registered test set unless the gate passes and the parent separately authorizes that phase.

## Prior-art context

Phase-sensitive spectral invariants and rotation-invariant bispectral representations predate this experiment. These references motivate the feature family but do not establish novelty or non-novelty of the exact four-channel selected-lag design.

1. Ramakrishna Kakarala, “The Bispectrum as a Source of Phase-Sensitive Invariants for Fourier Descriptors: A Group-Theoretic Approach,” Journal of Mathematical Imaging and Vision 44 (2012), DOI: 10.1007/s10851-012-0330-6; [preprint](https://arxiv.org/abs/0902.0196).
2. Ramakrishna Kakarala and Dansheng Mao, “A Theory of Phase-Sensitive Rotation Invariance with Spherical Harmonic and Moment-Based Representations,” CVPR 2010; [paper](https://iacl.ece.jhu.edu/proceedings/cvpr2010/papers/1059.pdf).
3. Zhizhen Zhao and Amit Singer, “Rotationally Invariant Image Representation for Viewing Direction Classification in Cryo-EM,” Journal of Structural Biology 186(1):153–166 (2014), DOI: 10.1016/j.jsb.2014.03.003; [preprint](https://arxiv.org/abs/1309.7643).
