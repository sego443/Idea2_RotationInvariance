# C170 temporal phase-invariant v6 implementation and execution plan

1. Add the isolated 23-channel temporal-FIR model with fixed identity buffer, seven normalized learnable-or-frozen delta-initialized filters, and the four registered masks.
2. Add an independent runner that uses the frozen v3 train-only preprocessing/safe-NPZ helpers and v2 sampler, weighted CE, validation, and early stopping. Record exact source/data/config hashes, per-epoch curves, selected-checkpoint train evaluation, validation predictions, and completion hashes.
3. Run focused formula, C4/ideal-global-phase invariance, delta-lag equivalence, FIR causality, learnable/frozen gradient and parameter-count, identical backend initialization, safe NPZ round-trip including the identity buffer, manifest, and four-arm CUDA-forward checks in Python 3.10 / PyTorch 1.13.1+cu117.
4. Freeze source and send the runner contract to the launcher/review agents. Create an independent source/environment snapshot and run the approved one-epoch CUDA smoke arms in separate output roots.
5. Begin the fixed four-arm by five-seed formal matrix only after the parent authorizes it. Keep test data sealed; report validation means, population standard deviations, per-class recall, all validation positions, paired seed comparisons, and prediction diversity.
6. Do not evaluate the registered test split unless the primary validation gate passes and the parent explicitly authorizes a later test phase.

Do not modify or overwrite v4/v5 files or outputs. Do not add losses, heads, orthogonality constraints, preprocessing changes, extra features, or hyperparameter search. Describe the causal scope as the FIR/front-end features only; the full CNN classifier uses symmetric padding and whole-window pooling.
