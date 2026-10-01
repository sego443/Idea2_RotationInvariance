# Synthetic-only v5 Nyquist symmetry check

Environment: NVIDIA GeForce RTX 2080 Ti, PyTorch 1.13.1+cu117 (CUDA 11.7), 2 CPU threads, batch size 2.

| Arm | Formal run path | Selected epoch | Runner SHA-256 | Model SHA-256 | Source bundle SHA-256 | RUN_COMPLETE SHA-256 | Manifest SHA-256 | Safe NPZ SHA-256 |
|---|---|---:|---|---|---|---|---|---|
| nyquist_coupled | `E:\MyCode\codex\Idea2\Idea2_RotationInvariance\outputs\c170_harmonic_nyquist_v5_formal_cuda_20260929\nyquist_coupled__seed-17` | 6 | `3578c484f21f9151ddbf936d96fb2b92cfd34d7c8bfeecc48fd6594650fa5b7e` | `7eda573cf276cb322dedaf74a47db6691388078dc2925d73a908e8ddf4a5a39a` | `ea4a7083c41e958cad2c85fbfe2af4028d808ce049102837e7a7514273a989c8` | `00ec0631c69f4e999a1dea658575d25ba038e08c9da396a8247c7eab489eba35` | `820297df966f1e73fa0a1c8cedd0355e66cb88dcae697ab2f16ec4ce73873d49` | `8c3a5c810f81f2e4240ff3efd64355a1a67e10bf66d20ab48710eda2349d9fb9` |
| nyquist_uncoupled | `E:\MyCode\codex\Idea2\Idea2_RotationInvariance\outputs\c170_harmonic_nyquist_v5_formal_cuda_20260929\nyquist_uncoupled__seed-17` | 2 | `3578c484f21f9151ddbf936d96fb2b92cfd34d7c8bfeecc48fd6594650fa5b7e` | `7eda573cf276cb322dedaf74a47db6691388078dc2925d73a908e8ddf4a5a39a` | `ea4a7083c41e958cad2c85fbfe2af4028d808ce049102837e7a7514273a989c8` | `33a8550211de4140a9435576e529d9cce00849b5f04f2d35409c9e199ea4e85b` | `5e35a4a488a100e2955a536363f5bd94659d22d2b229f0cd01e7185a09409d27` | `0a094536dff3f18cfb7f20a62b1d658595f3525ce278f524e324d08afcb5fa7d` |

The script loaded only the completed-run manifests and numeric `best_weights.npz` files, then passed two deterministic unlabeled synthetic raw-scale windows through the v3 filter, per-window/per-channel demean, and each checkpoint's fixed train-derived scalar 11123.037109375. It did not open the dataset or calculate test metrics.

| Arm | Transform | Max abs logit error | Max abs normalized embedding error | Max embedding cosine gap |
|---|---|---:|---:|---:|
| nyquist_coupled | C4 roll 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| nyquist_coupled | C4 roll 1 | 4.768e-07 | 5.960e-08 | 5.960e-08 |
| nyquist_coupled | C4 roll 2 | 4.768e-07 | 5.215e-08 | 0.000e+00 |
| nyquist_coupled | C4 roll 3 | 4.768e-07 | 2.980e-08 | 0.000e+00 |
| nyquist_coupled | fixed-n z phase 0.37 rad | 7.418e-02 | 1.217e-02 | 1.205e-03 |
| nyquist_coupled | fixed-n z phase 1.13 rad | 2.258e-01 | 3.019e-02 | 6.393e-03 |
| nyquist_coupled | fixed-n z phase 2.41 rad | 3.542e-01 | 3.618e-02 | 9.373e-03 |
| nyquist_uncoupled | C4 roll 0 | 0.000e+00 | 0.000e+00 | 1.192e-07 |
| nyquist_uncoupled | C4 roll 1 | 2.384e-07 | 5.960e-08 | 1.192e-07 |
| nyquist_uncoupled | C4 roll 2 | 2.384e-07 | 5.960e-08 | 1.192e-07 |
| nyquist_uncoupled | C4 roll 3 | 2.384e-07 | 4.470e-08 | 1.192e-07 |
| nyquist_uncoupled | fixed-n z phase 0.37 rad | 2.980e-07 | 8.196e-08 | 5.960e-08 |
| nyquist_uncoupled | fixed-n z phase 1.13 rad | 2.533e-07 | 6.706e-08 | 0.000e+00 |
| nyquist_uncoupled | fixed-n z phase 2.41 rad | 3.576e-07 | 6.706e-08 | 0.000e+00 |

The C4 rows measure discrete cyclic channel rolls on these constructed inputs. The phase rows rotate only the ideal first-harmonic `(a,b)` pair while holding the synthetic Nyquist component `n` fixed. The uncoupled arm is expected to be invariant up to float roundoff; the coupled arm contains `n*z²`, which rotates by `exp(i*2*phi)`, so its observed change is reported rather than treated as a failure. This is a mathematical negative control, not a physical wrist-rotation or general SO(2) test. Four-sensor Nyquist and aliasing remain limitations.
