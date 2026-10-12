# v22 time-window and amplitude-normalization ablations (fresh exploratory CUDA runs)

Date: 2026-10-12 (Asia/Hong_Kong). Remote source pulled with `git pull --ff-only` to `79a03c5`. This report describes new local execution, separate from the historical v21 Apple M2 scores.

All **81/81 planned runs completed**: 66 true-label runs (11 variants × 2 protocols × 3 seeds) and 15 matched block-permuted nulls (11 C170 variants and the first 4 W01 variants). Both requested ablations are implemented by the single v22 grid. No W02 acquisition, S2 freeze, or W02 evaluation was performed.

The fresh full/shared baseline has W01 ZERO macro-F1 **0.233** and same-recording CAL τ=1 **0.910**. The largest W01 ZERO mean is **0.337** for `prefix_200/shared`; the largest CAL mean is **0.944** for `full/group_sensor`. These are exploratory rankings across reused data, not independent model-selection evidence.

## Execution and verification

- Runtime: Python 3.10.18, PyTorch 2.6.0+cu124, NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.7.0, Windows, RTX 2080 Ti, NVIDIA driver 591.86. Four concurrent CUDA jobs, two CPU threads per job.
- B2 signed CNN; same-group episodic weight 1; training τ=4; augmentation enabled; batch size 64; AdamW lr 0.001, weight decay 0.001; fixed 120 epochs. The final epoch 120 is selected for every run; held-out scores do not choose checkpoints. Real seeds: 17, 29, 43; null model seed: 17; block-permutation seed: 17.
- Fresh relevant tests: 18 passed. Full suite: 167 passed. The full/shared CUDA smoke retry completed one epoch; its scores are excluded. Original smoke directory is preserved as failed/incomplete.
- The first CUDA smoke failed before evaluation because NumPy default integer labels are int32 on Windows. The C170 and W01 loaders now return explicit int64 labels. Two regression cases failed on cross-entropy before the fix and passed afterward. This changes dtype, not label values, split, training budget, or preprocessing. The episode-loss test fixture also uses explicit int64 labels.
- Fresh audit checked all 81 completion markers, full 1–120 epoch curves with finite losses, source/data hashes, prediction counts and valid class IDs, calibration exclusion, duplicate exclusion, and recalculated macro-F1. Portable evidence contains per-run and pooled per-position metrics, per-class recall, prediction counts, channel-map strata and artifact hashes. Local checkpoints and raw predictions remain in the ignored output root.

Run command:

```powershell
& 'C:/Users/56569/Miniforge3/envs/ML2025/python.exe' -m scripts.launchers.launch_v22_window_norm_ablation --output-root outputs/v22_window_norm_ablation_cuda_20261012 --workers 4 --device cuda --cpu-threads 2
```

## Data, preprocessing and access boundary

Class order is **UP, DOWN, LEFT, RIGHT** (IDs 0–3). Original recordings and rejected takes were unchanged. Inputs are existing raw two-second windows, 500 samples at 250 Hz, four stored sensor channels. C170 has 1,600 windows, with its one marked copied window excluded. W01 has 240 windows, 60 per class.

- Fixed split: 879 training windows from positions 0/10/20/30/50/60/80/100/120/140/160 mm; 70 mm excluded. Old-validation positions 5/40/90/130 mm and exposed-test positions 15/25/110/150 mm are evaluated together after training, then reported separately (288 scored windows each).
- W01 transfer: train on all 1,599 nonduplicate C170 windows, including 70 mm; evaluate all 12 W01 angles (0–330° in 30° steps), 144 scored windows.
- CAL support is repetitions 1–2 of each direction (8 strokes/group); score later repetitions only. CAL uses true evaluation support labels even for the training-label nulls. C170/W01 each direction occupies its own recording, so support and scored strokes share recording identity. CALNB uses another held-out position 10–20 mm away. UNSUP is transductive, using other unlabeled scored strokes in the group.
- Each visible span receives its own third-order 0.5–40 Hz Butterworth zero-phase filter and temporal demeaning; cut edges are tapered over 20 samples and hidden samples zero-padded. Prefixes retain [0,200/250/300); return-only retains [250,500). Outbound-valley chooses the envelope valley between two strongest bursts using the full window (fallback 250), so it is oracle segmentation. Even fixed prefixes come from already localized peak-centred windows and do not establish causal continuous detection.
- Shared normalization uses a training-fitted scalar. Window-global/window-sensor RMS use each individual visible input. **group_sensor uses each evaluation group’s full unlabeled input pool, including scored windows, to fit per-channel medians.** The user explicitly instructed this run to follow the original v22 design and authorized this transductive diagnostic on 2026-10-12. It must not be described as train-only preprocessing or deployable online without an equivalent input pool.
- The loaders decode the full NPZ before optimization; only training groups and their labels enter training. No held-out checkpoint selection occurs. Evaluation labels enter CAL support and subsequent scoring; group_sensor and UNSUP access evaluation inputs as described. C170 registered test was already exposed by F2 on 2026-09-29 and by later development. W01 and old validation were repeatedly used. No untouched holdout is accessed here.

C170 EXG maps 0,2,4,6 and 0,1,2,3 were previously user-confirmed as physical S0,S1,S2,S3; the map transition crosses 70 mm. W01 stores EXG 0,1,2,3; the current data audit validates channel columns but does not independently revalidate anatomical sensor order. B2 is a signed CNN without C4 pooling; neither cyclic channel indexing nor these scores prove physical rotation invariance.

Data SHA-256:

- C170: `d57a7fef61ae4930b3a91fd917ce44ef481a7a84e58b261deb44a3040569e349`
- W01: `29246cf45f8fb17ac398ddf1d5c58dbef1df88970bb8739fad8513fa110b9179`

Both match the existing S1 freeze manifest. Executed source hashes are in each fresh run manifest and the verified evidence. Historical source hashes and the English conversion ledger are unchanged.

## Time-window ablation

Values are mean macro-F1 over three seeds. CAL means CAL τ=1. All rows in this table use the shared training scale.

| Window | Normalization | C170 old val ZERO / CAL | C170 exposed test ZERO / CAL | W01 ZERO / CAL |
|---|---|---:|---:|---:|
| full | shared | 0.438 / 0.765 | 0.678 / 0.856 | 0.233 / 0.910 |
| prefix_200 | shared | 0.439 / 0.670 | 0.480 / 0.622 | 0.337 / 0.673 |
| prefix_250 | shared | 0.385 / 0.782 | 0.559 / 0.765 | 0.278 / 0.877 |
| prefix_300 | shared | 0.398 / 0.727 | 0.561 / 0.836 | 0.212 / 0.914 |
| outbound_valley | shared | 0.479 / 0.861 | 0.559 / 0.752 | 0.261 / 0.759 |
| return_only | shared | 0.344 / 0.763 | 0.382 / 0.785 | 0.250 / 0.931 |

## Normalization ablation and combinations

| Window | Normalization | C170 old val ZERO / CAL | C170 exposed test ZERO / CAL | W01 ZERO / CAL |
|---|---|---:|---:|---:|
| full | shared | 0.438 / 0.765 | 0.678 / 0.856 | 0.233 / 0.910 |
| full | window_global | 0.435 / 0.772 | 0.594 / 0.898 | 0.256 / 0.940 |
| full | window_sensor | 0.457 / 0.739 | 0.511 / 0.793 | 0.217 / 0.889 |
| full | group_sensor | 0.456 / 0.839 | 0.616 / 0.927 | 0.211 / 0.944 |
| outbound_valley | window_global | 0.492 / 0.797 | 0.541 / 0.701 | 0.243 / 0.794 |
| outbound_valley | group_sensor | 0.559 / 0.829 | 0.566 / 0.718 | 0.263 / 0.792 |

The group_sensor rows have the additional transductive access described above. Full seed means/SD (population SD, ddof=0), majority votes, UNSUP τ=1, CALNB τ=4, nulls and opposite-direction error shares are in [the complete score table](../evidence/v22_ablation_20261012/score_table.md). All evaluated τ values 1/4/16 and all position/channel strata are in the verified evidence.

## Interpretation of the two ablations

1. **Shortening the window does not consistently improve transfer.** The 200-sample prefix gives the largest W01 ZERO mean (0.337), but its W01 CAL mean drops to 0.673. It also lowers C170 exposed-test ZERO/CAL relative to full/shared, and has no planned W01 null. This is a lead for a new experiment, not a validated replacement.
2. **The return half contains substantial same-recording calibration information.** Return-only W01 CAL is 0.931, while ZERO is 0.250. Full-window calibration therefore cannot be attributed exclusively to the outward movement. The oracle outbound cut improves C170 old-validation CAL but worsens exposed-test and W01 CAL.
3. **Normalization helps calibration more than zero-calibration semantics.** Full/group_sensor has the largest CAL seed mean among full-window normalization variants on all three reported subsets, but its additional transductive input pool is part of that setting. Its W01 ZERO remains low; window-global and window-sensor normalization also do not produce strong ZERO transfer. Seed differences are not confidence intervals for new wears, days or people.
4. **Zero-calibration cross-session performance remains weak.** Even the highest exploratory W01 ZERO mean is 0.337; no ablation establishes the project’s physical rotation-generalization gate. The strong CAL scores still require labeled support from the same direction recording, and the matched CAL nulls must be considered before attributing that performance to correctly labeled global training.

## Matched nulls, recall and prediction diversity

Nulls permute complete direction-recording label blocks within every training wear group, preserving within-block repetition dependence. One null seed per planned variant is a diagnostic rather than a permutation distribution or a significance test. Seven W01 variants have no planned null; they must not inherit another variant’s control result.

The table shows the three-seed true-label mean and, separately, the matched seed-17 true/null pair. A strong CAL null shows that labeled local support can support template matching without correctly labeled global training; a single matched null does not establish equivalence.

| Window / norm | Subset | True mean ZERO / CAL | True seed 17 ZERO / CAL | Null seed 17 ZERO / CAL |
|---|---|---:|---:|---:|
| full / shared | c170_val | 0.438 / 0.765 | 0.448 / 0.756 | 0.184 / 0.685 |
| full / shared | c170_test | 0.678 / 0.856 | 0.759 / 0.868 | 0.143 / 0.883 |
| full / shared | w01 | 0.233 / 0.910 | 0.292 / 0.972 | 0.185 / 0.897 |
| outbound_valley / shared | c170_val | 0.479 / 0.861 | 0.518 / 0.899 | 0.209 / 0.613 |
| outbound_valley / shared | c170_test | 0.559 / 0.752 | 0.566 / 0.742 | 0.196 / 0.633 |
| outbound_valley / shared | w01 | 0.261 / 0.759 | 0.261 / 0.809 | 0.185 / 0.806 |
| full / window_global | c170_val | 0.435 / 0.772 | 0.358 / 0.775 | 0.268 / 0.727 |
| full / window_global | c170_test | 0.594 / 0.898 | 0.596 / 0.872 | 0.199 / 0.790 |
| full / window_global | w01 | 0.256 / 0.940 | 0.214 / 0.881 | 0.260 / 0.979 |
| full / group_sensor | c170_val | 0.456 / 0.839 | 0.439 / 0.863 | 0.198 / 0.679 |
| full / group_sensor | c170_test | 0.616 / 0.927 | 0.692 / 0.921 | 0.201 / 0.938 |
| full / group_sensor | w01 | 0.211 / 0.944 | 0.178 / 0.986 | 0.248 / 0.944 |

Recall and prediction counts below are pooled over the three seeds, in class order UP/DOWN/LEFT/RIGHT. They describe individual-seed predictions, not the majority-vote predictions.

| Variant | W01 mode | Recall UP / DOWN / LEFT / RIGHT | Prediction counts | Opposite-direction share of errors |
|---|---|---|---|---:|
| full/shared | zero | 0.185 / 0.093 / 0.343 / 0.407 | 104 / 27 / 101 / 200 | 0.287 |
| full/shared | cal_tau1 | 0.889 / 0.898 / 0.880 / 0.972 | 110 / 98 / 104 / 120 | 0.282 |
| prefix_200/shared | zero | 0.120 / 0.565 / 0.296 / 0.463 | 36 / 190 / 84 / 122 | 0.246 |
| prefix_200/shared | cal_tau1 | 0.463 / 0.685 / 0.685 / 0.889 | 66 / 109 / 108 / 149 | 0.239 |
| full/group_sensor | zero | 0.204 / 0.074 / 0.296 / 0.315 | 119 / 27 / 130 / 156 | 0.256 |
| full/group_sensor | cal_tau1 | 0.926 / 0.944 / 0.981 / 0.926 | 106 / 103 / 112 / 111 | 0.208 |

## Position and channel-map detail

The following compact view compares fresh baseline and the W01 ZERO-leading variant. These are macro-F1 values recomputed from predictions pooled across three seeds, not seed-mean F1. Every other variant, null, mode and per-class detail is preserved in the evidence.

| Variant | Position / angle | ZERO F1 | CAL τ=1 F1 |
|---|---|---:|---:|
| full/shared | C170_P005 | 0.716 | 0.706 |
| full/shared | C170_P015 | 0.797 | 0.871 |
| full/shared | C170_P025 | 0.931 | 0.991 |
| full/shared | C170_P040 | 0.497 | 0.661 |
| full/shared | C170_P090 | 0.193 | 0.905 |
| full/shared | C170_P110 | 0.483 | 0.736 |
| full/shared | C170_P130 | 0.307 | 0.738 |
| full/shared | C170_P150 | 0.413 | 0.811 |
| full/shared | W01_A000 | 0.352 | 0.944 |
| full/shared | W01_A030 | 0.185 | 0.972 |
| full/shared | W01_A060 | 0.105 | 0.889 |
| full/shared | W01_A090 | 0.165 | 0.892 |
| full/shared | W01_A120 | 0.289 | 0.858 |
| full/shared | W01_A150 | 0.269 | 0.944 |
| full/shared | W01_A180 | 0.296 | 0.760 |
| full/shared | W01_A210 | 0.334 | 0.789 |
| full/shared | W01_A240 | 0.086 | 1.000 |
| full/shared | W01_A270 | 0.173 | 0.944 |
| full/shared | W01_A300 | 0.259 | 1.000 |
| full/shared | W01_A330 | 0.023 | 0.912 |
| prefix_200/shared | C170_P005 | 0.432 | 0.503 |
| prefix_200/shared | C170_P015 | 0.648 | 0.830 |
| prefix_200/shared | C170_P025 | 0.521 | 0.590 |
| prefix_200/shared | C170_P040 | 0.450 | 0.722 |
| prefix_200/shared | C170_P090 | 0.387 | 0.589 |
| prefix_200/shared | C170_P110 | 0.467 | 0.553 |
| prefix_200/shared | C170_P130 | 0.425 | 0.810 |
| prefix_200/shared | C170_P150 | 0.214 | 0.501 |
| prefix_200/shared | W01_A000 | 0.105 | 0.614 |
| prefix_200/shared | W01_A030 | 0.205 | 0.794 |
| prefix_200/shared | W01_A060 | 0.306 | 0.741 |
| prefix_200/shared | W01_A090 | 0.439 | 0.553 |
| prefix_200/shared | W01_A120 | 0.337 | 0.669 |
| prefix_200/shared | W01_A150 | 0.346 | 0.715 |
| prefix_200/shared | W01_A180 | 0.373 | 0.681 |
| prefix_200/shared | W01_A210 | 0.263 | 0.684 |
| prefix_200/shared | W01_A240 | 0.411 | 0.625 |
| prefix_200/shared | W01_A270 | 0.249 | 0.608 |
| prefix_200/shared | W01_A300 | 0.433 | 0.476 |
| prefix_200/shared | W01_A330 | 0.206 | 0.607 |

Pooled channel-map strata (macro-F1):

| Variant | Protocol | EXG map | ZERO | CAL τ=1 |
|---|---|---|---:|---:|
| full/shared | fixed_split | 0,1,2,3 | 0.369 | 0.805 |
| full/shared | fixed_split | 0,2,4,6 | 0.746 | 0.814 |
| full/shared | w01_transfer | 0,1,2,3 | 0.244 | 0.910 |
| prefix_200/shared | fixed_split | 0,1,2,3 | 0.390 | 0.628 |
| prefix_200/shared | fixed_split | 0,2,4,6 | 0.517 | 0.666 |
| prefix_200/shared | w01_transfer | 0,1,2,3 | 0.340 | 0.672 |

Map strata are confounded with C170 position/acquisition order; they are diagnostics, not controlled causal comparisons of channel wiring.

## Evidence and preservation

- [Preflight record](../evidence/v22_ablation_20261012/preflight.json): inputs, preprocessing, labels, environment, output policy and access boundary.
- [Score summary JSON](../evidence/v22_ablation_20261012/summary.json) and [full score table](../evidence/v22_ablation_20261012/score_table.md).
- [Verified run evidence](../evidence/v22_ablation_20261012/verified_run_evidence.json): all 81 run manifests, final epochs, data/source/artifact hashes, per-position metrics and channel-map strata.
- [Verification record](../evidence/v22_ablation_20261012/verification.json). Local output root: `outputs/v22_window_norm_ablation_cuda_20261012` (ignored by Git).

No historical/incomplete artifacts were overwritten or relabeled. The original failed smoke remains `outputs/v22_cuda_smoke_20261012`; its completed retry remains separate and is excluded from all tables.

This evidence addresses presegmented four-direction classification on existing single-participant recordings. C170 is same-session position transfer. W01 is a different session/day; it is reused cross-session evidence with same-recording calibration, not a new independent rewearing trial. There is no fresh mathematical invariance claim, untouched physical rewearing validation, cross-person evidence, continuous detection, transcription or open-vocabulary recognition. A new untouched W02 holdout is still required for confirmatory claims.
