# C170 harmonic-Nyquist v5 implementation and execution plan

1. Add the isolated 17-feature model and four fixed masks, retaining the v4 backend and its seed-reset initialization rule.
2. Add an independent runner that reuses the frozen v2 sampler/objective/evaluation helpers and v3 train-only shared-scale preprocessing/safe NPZ helpers. It must record exact source/data hashes, write per-epoch curves and validation predictions, and mark completion only after safe-weight reload and artifact hashes pass.
3. Run focused formula, lag-boundary, C4-invariance, non-U(1) negative-control, mask/capacity, exact-sampler, safe-NPZ round-trip, manifest, and forward checks in the frozen Python 3.10 / PyTorch 1.13.1+cu117 environment.
4. Freeze v5 source, create a source/environment snapshot, and hand off CUDA smoke runs for `nyquist_coupled` and `nyquist_shuffle`. Do not begin the formal matrix before the parent authorizes it.
5. When authorized, run the fixed four-arm by five-seed matrix. Keep the test split sealed; summarize validation means, population standard deviations, per-class recalls, all four positions, matched seed differences, prediction diversity, and the historical v4 comparison.
6. Release no test evaluation unless the primary arm meets both validation gates and the parent explicitly authorizes the later test phase.

No hyperparameter search, preprocessing change, data edit, continuous-rotation claim, or modification to v4/v3 code is in scope.
