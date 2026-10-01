# Synthetic-only v4 symmetry check

Run: `E:\MyCode\codex\Idea2\Idea2_RotationInvariance\outputs\c170_harmonic_phase_v4_formal_cuda_20260929\harmonic_gram__seed-17` (harmonic_gram, seed 17); selected epoch: 28; config SHA-256: `0012e8862b8c5e092f78b4a2509542a83eaecaf27c3c49bd6ab9c71efc38b578`; manifest SHA-256: `5a38e655f089ef78588973886f8d641885ab4379845a7088ca93ca62603a2a52`; RUN_COMPLETE SHA-256: `707574db6b2889a50eed1f1f85fb1f916dd1564022a03932f3e77b5489c51f57`; safe NPZ SHA-256: `615711a5547eac1db9c633ee3e7e60243ec42a2af331663ec3a387aac1b715d3`.
Environment: NVIDIA GeForce RTX 2080 Ti, PyTorch 1.13.1+cu117 (CUDA 11.7), 2 CPU threads, batch size 2.
Input preprocessing: v3 0.5–40 Hz order-3 zero-phase filter, per-window/per-channel demean, and fixed train-derived shared scalar 11123.037109375.

All inputs below are deterministic synthetic raw-scale waveforms with no labels. The script loads only the completed run manifest and numeric `best_weights.npz`; it does not open the dataset or calculate test metrics.

| Cyclic channel roll | Max abs logit error | Max abs normalized embedding error | Max cosine gap |
|---:|---:|---:|---:|
| 0 | 0.000e+00 | 0.000e+00 | 1.192e-07 |
| 1 | 9.537e-07 | 5.960e-08 | 0.000e+00 |
| 2 | 9.537e-07 | 5.960e-08 | 0.000e+00 |
| 3 | 1.907e-06 | 5.960e-08 | 5.960e-08 |

| Synthetic first-harmonic common phase | Max abs logit error | Max abs normalized embedding error | Max cosine gap |
|---:|---:|---:|---:|
| 0.37 rad | 1.431e-06 | 7.451e-08 | 1.192e-07 |
| 1.13 rad | 2.384e-06 | 7.451e-08 | 5.960e-08 |
| 2.41 rad | 1.311e-06 | 7.451e-08 | 1.192e-07 |

The channel-roll check tests the model's discrete C4 behavior on these constructed inputs. The phase check rotates only the synthetic first-harmonic `(a,b)` pair; it is an idealized mathematical action, not a physical wrist rotation or evidence of general SO(2) invariance. Continuous-angle invariance requires the strict spatial bandlimit `|m|<=1`; four-sensor Nyquist and aliased higher harmonics are not covered. This check is diagnostic only and does not alter model selection or the validation/test boundary.
