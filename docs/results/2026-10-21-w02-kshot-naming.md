# W02 k-shot cluster naming: exploratory results

Run and written on 2026-10-21. **All results are exploratory.** This is a fresh execution of the unchanged `explore_w02_kshot_naming` entry point from code commit `2039d19086562d997f4552b29ffff4176b8590bb`, using the repeatedly explored W02 development wears.

Three labeled prefix strokes per recording are a promising reduction from four: `cluster_zero` reaches mean macro-F1 **0.969** across all four choices of three classes (range **0.963–0.980**), compared with **0.971** using four labeled strokes. Two labels give **0.907** on average, with a much wider **0.798–0.980** range. These descriptive comparisons do not establish equivalence, a deployment-ready calibration budget, or a confirmatory success.

## Execution and access boundary

- Data were supplied through the separate private data repository, commit `76bca414ac24c8d17ca7e47379f1a96650668dfe`. The copied development NPZ matches the published split hash exactly.
- Wears: Part A W01–W06 and Part B W08/W09/W11; one participant P01, two acquisition days (2026-10-13/14). There are 54 trial recordings, 1,080 presegmented strokes, 432 prefix strokes and **648 scored body strokes** (162 per class). Each recording has 8 prefix and 12 body strokes. The 63 REST windows are unused.
- **Class and confusion-matrix order: UP, DOWN, LEFT, RIGHT (0–3).** Each body has three strokes of each class. For budget b, support is the first prefix stroke of each of b distinct classes, using every class subset: 1/4/6/4/1 subsets for b=0/1/2/3/4. Thus b is the total number of labels per recording, not shots per class. The same subset is applied to all recordings in a row.
- Leave-one-wear-out training uses eight complete wears and scores the ninth. PCA, the borrowed precision and zero-model standardization/classifier are fitted on training wears only. The method intentionally adapts to unlabeled held-out recordings: tangent references, recording-level scaling/clustering and body refinement use held-out unlabeled signals. This is transductive analysis of a complete recording.
- The supplied repository contains **no sealed W07/W10/W12 data**, and the runner refuses those wear identifiers. No sealed W02 signals, raw recordings, or C170 data were opened by this run. This work does not consume the future W02 sealed evaluation. The C170 registered test remains historically exposed.
- No source, acquisition record or signal was changed. The output directory and log were absent before execution; the runner creates the directory with `exist_ok=False`. Original output is retained at `outputs/w02_explore_20261015/kshot_naming_v1`, with the adjacent `.log`, preflight and exit records. Execution ran from 06:55:54 to 06:57:38 UTC (14:55:54–14:57:38 Hong Kong), exiting with code 0.

## Features and method interpretation

Input is the existing 500-sample, 2-second window at 250 Hz, localized on the prompt grid as described in the [segmentation record](2026-10-15-w02-grid-segmentation.md); no raw window re-cut was performed. The recipe applies a third-order zero-phase 0.5–12 Hz Butterworth filter, temporal demeaning and lags 0,5,…,30 samples to form a regularized 28×28 covariance, then tangent-space features.

`cluster_zero` fits four-cluster k-means separately to all 20 strokes of each recording, after local tangent mapping and z-scoring. A one-to-one Hungarian assignment names clusters using the zero-model log probabilities and large penalties for contradicting support labels. `cluster_flat` omits the zero-model prior; unresolved assignments depend on arbitrary ties. The zero model is logistic regression (C=0.1) trained on the other wears, with per-recording unlabeled re-centering and angle-dependent cyclic channel reindexing. The exact implemented shift convention is retained in the hashed source; it was not changed to match a verbal angle convention.

The anchored methods borrow PCA-30 features and a Ledoit-Wolf within-class precision from training wears, then perform two soft body-refinement passes. Missing classes start either at zero-model-weighted means (`anchored_zero`) or at successively farthest body strokes (`anchored_farthest`), with zero-model naming of missing slots for the latter. At b=4 both reduce to `rec1_borrowed` and attain **1.000 prediction agreement** with its separately computed reference on all 648 body strokes.

**Physical-order and channel strata:** the acquisition table records map `0,1,2,3` for all 54 development trial recordings (and their REST recordings). All reported metrics therefore belong to this one channel-map stratum; no alternate map is available. The NPZ lacks a channel-map field, so this is a check against acquisition metadata, not a new physical inspection of sensor placement. Cyclic reindexing does not demonstrate invariance to changing anatomical contacts. Recording/stroke identities are unique; the W02 archive has no marked-duplicate flag. No C170 duplicate window is involved.

## Budget curves

Each cell is **mean macro-F1 [minimum, maximum across class subsets]**. Each subset's F1 pools the 648 body predictions. The subset mean is not a pooled prediction ensemble or an estimate for randomly selected labels. There is no bootstrap interval for the subset mean.

| Method | b=0 | b=1 | b=2 | b=3 | b=4 |
|---|---:|---:|---:|---:|---:|
| `cluster_zero` | 0.601 [0.601, 0.601] | 0.786 [0.644, 0.912] | 0.907 [0.798, 0.980] | 0.969 [0.963, 0.980] | 0.971 [0.971, 0.971] |
| `cluster_flat` | 0.247 [0.247, 0.247] | 0.480 [0.450, 0.529] | 0.722 [0.674, 0.753] | 0.961 [0.944, 0.981] | 0.957 [0.957, 0.957] |
| `anchored_zero` | 0.570 [0.570, 0.570] | 0.607 [0.546, 0.694] | 0.681 [0.610, 0.790] | 0.782 [0.720, 0.810] | 0.933 [0.933, 0.933] |
| `anchored_farthest` | 0.561 [0.561, 0.561] | 0.677 [0.581, 0.768] | 0.767 [0.653, 0.829] | 0.844 [0.792, 0.916] | 0.933 [0.933, 0.933] |

The four-label anchored baseline freshly reproduces F1 0.933358. Its bootstrap interval [0.897642, 0.965956] belongs to this run; the [October 15 report](2026-10-15-w02-exploratory-development.md) contains historical results, with a different bootstrap sample count. No historical score is treated as a rerun.

**Why fewer labels sometimes score higher:** assignment accuracy is not monotonic in the number of constraints. Support strokes can lie in the same k-means cluster and conflict with a one-to-one labeling. Finite penalties and tie handling can then change the assignment. The saved outputs do not count support collisions, so they do not establish the cause of each non-monotonic row.

## Class subsets, recall and prediction diversity

The following rows describe the leading naming method and the four-label anchored reference. Confidence intervals use 2,000 bootstrap samples of complete wears, seed 20261015. They are pointwise exploratory intervals and do not correct for selection across methods or subsets. U/D/L/R vectors follow the class order above. Naming failures count recordings where body accuracy falls at least 0.25 below the best permutation of that recording's predicted labels.

| Method | b / labeled classes | F1 [95% CI] | Accuracy | Recall U/D/L/R | Prediction counts U/D/L/R | Naming failures / 54 |
|---|---|---:|---:|---|---|---:|
| `cluster_zero` | 0 / none | 0.601 [0.447, 0.752] | 0.602 | 0.407/0.494/0.907/0.599 | 163/161/164/160 | 33 |
| `cluster_zero` | 1 / UP | 0.912 [0.846, 0.957] | 0.912 | 0.944/0.901/0.963/0.840 | 163/161/164/160 | 9 |
| `cluster_zero` | 2 / UP+DOWN | 0.954 [0.930, 0.975] | 0.954 | 0.951/0.975/0.981/0.907 | 161/162/166/159 | 4 |
| `cluster_zero` | 2 / UP+LEFT | 0.927 [0.856, 0.977] | 0.927 | 0.944/0.901/1.000/0.864 | 163/161/166/158 | 7 |
| `cluster_zero` | 2 / UP+RIGHT | 0.964 [0.941, 0.985] | 0.965 | 0.944/0.957/1.000/0.957 | 163/161/166/158 | 3 |
| `cluster_zero` | 2 / DOWN+LEFT | 0.818 [0.707, 0.925] | 0.819 | 0.654/0.975/1.000/0.648 | 161/162/166/159 | 18 |
| `cluster_zero` | 2 / DOWN+RIGHT | 0.980 [0.957, 0.995] | 0.980 | 0.969/0.975/1.000/0.975 | 161/163/166/158 | 1 |
| `cluster_zero` | 2 / LEFT+RIGHT | 0.798 [0.690, 0.899] | 0.798 | 0.611/0.605/1.000/0.975 | 163/161/166/158 | 21 |
| `cluster_zero` | 3 / UP+DOWN+LEFT | 0.963 [0.941, 0.985] | 0.963 | 0.951/0.975/1.000/0.926 | 161/162/166/159 | 3 |
| `cluster_zero` | 3 / UP+DOWN+RIGHT | 0.971 [0.952, 0.989] | 0.971 | 0.951/0.994/1.000/0.938 | 161/163/166/158 | 2 |
| `cluster_zero` | 3 / UP+LEFT+RIGHT | 0.964 [0.941, 0.985] | 0.965 | 0.944/0.957/1.000/0.957 | 163/161/166/158 | 3 |
| `cluster_zero` | 3 / DOWN+LEFT+RIGHT | 0.980 [0.957, 0.995] | 0.980 | 0.969/0.975/1.000/0.975 | 161/163/166/158 | 1 |
| `cluster_zero` | 4 / UP+DOWN+LEFT+RIGHT | 0.971 [0.952, 0.989] | 0.971 | 0.951/0.994/1.000/0.938 | 161/163/166/158 | 2 |
| `anchored_zero` | 4 / UP+DOWN+LEFT+RIGHT | 0.933 [0.898, 0.966] | 0.934 | 0.901/0.901/0.994/0.938 | 154/157/169/168 | 0 |

All 64 real rows predict all four classes; the full [CSV](../evidence/w02_kshot_naming_20261021/row_metrics.csv) and [JSON](../evidence/w02_kshot_naming_20261021/results.json) retain every class count, recall, null score, confusion matrix, per-wear/per-angle F1 and naming-failure identifier. Balanced counts alone do not imply correct naming: zero-label `cluster_zero` predicts 160–164 strokes per class while still failing naming in 33/54 recordings. Anchored missing-class initialization also produces substantial imbalance in some rows (for example, `anchored_zero` with UP+DOWN+RIGHT predicts 107/111/339/91).

One UP label is the best one-label `cluster_zero` subset here (0.912), but the other choices reach only 0.644–0.798. The best two-label subset, DOWN+RIGHT (0.980), was selected on these same development data. It should not be generalized into a two-label guarantee. Three labels give the strongest consistency across the tested subsets. `cluster_flat` also reaches 0.944–0.981 with three labels: direct support constraints resolve much of the naming ambiguity without the zero-model prior. Anchored methods generalize less successfully to missing classes.

The diagnostic `cluster_oracle` gives F1 **0.989197**, accuracy 0.989198, recall 0.988/0.994/1.000/0.975, counts 161/163/166/158 and bootstrap interval [0.981506, 0.995380]. Its naming permutation is chosen using the true labels of **all 20 strokes**, before scoring only the body. It is a label-informed clustering reference, not an available prediction method or a strict body-optimal upper bound. It differs from the historical per-wear clustering purity experiment.

## Wear and angle breakdown

The table uses the fixed UP+DOWN+LEFT subset for three-label `cluster_zero` (the first enumerated subset, not a best-subset selection), alongside zero-label naming, four-label naming and the anchored reference. All rows below belong to channel map `0,1,2,3`. Each wear contributes 72 body strokes.

| Wear | Angle | cluster_zero b=0 | cluster_zero b=3 U+D+L | cluster_zero b=4 | anchored b=4 |
|---|---:|---:|---:|---:|---:|
| A_W01 | 193° | 0.765 | 0.986 | 0.986 | 0.898 |
| A_W02 | 3° | 0.765 | 0.972 | 0.972 | 0.820 |
| A_W03 | 225° | 0.430 | 0.916 | 0.986 | 0.885 |
| A_W04 | 289° | 0.417 | 1.000 | 1.000 | 0.958 |
| A_W05 | 34° | 0.292 | 1.000 | 1.000 | 0.986 |
| A_W06 | 87° | 0.750 | 0.917 | 0.917 | 0.930 |
| B_W08 | 3° | 0.292 | 0.917 | 0.917 | 0.972 |
| B_W09 | 225° | 0.972 | 0.972 | 0.972 | 0.945 |
| B_W11 | 34° | 0.736 | 0.986 | 0.986 | 1.000 |

Angle-level F1 pools predictions at the same measured angle; it is not the arithmetic mean of wear F1 values. Angles are confounded with a small number of wear episodes.

| Angle | Body strokes | cluster_zero b=0 | cluster_zero b=3 U+D+L | cluster_zero b=4 | anchored b=4 |
|---|---:|---:|---:|---:|---:|
| 3° | 144 | 0.528 | 0.944 | 0.944 | 0.896 |
| 34° | 144 | 0.512 | 0.993 | 0.993 | 0.993 |
| 87° | 72 | 0.750 | 0.917 | 0.917 | 0.930 |
| 193° | 72 | 0.765 | 0.986 | 0.986 | 0.898 |
| 225° | 144 | 0.701 | 0.944 | 0.979 | 0.917 |
| 289° | 72 | 0.417 | 1.000 | 1.000 | 0.958 |

These results mix within-day and cross-day training wears in every fold. They support an exploratory rewear analysis within this participant and two-day collection. They do not isolate day-held-out transfer, cross-person performance, mathematical invariance or pure physical rotation effects.

## Matched nulls

For seeds 0–4 the zero model is retrained with a separate class-label permutation within each complete training recording, and support labels are relabeled by a seeded four-class permutation per recording. The feature geometry, class partitions within each recording and train/test wears are preserved. Borrowed precision is computed once from real training labels; a within-recording class renaming preserves those within-class partitions, so this control tests semantic naming under preserved class structure. It does not establish a wholly label-free metric.

Each cell below is the mean of the five null scores over all class subsets at that budget. The full seed values are in the CSV/JSON. These five seeds are descriptive controls; no multiple-comparison-adjusted significance claim is made.

| Method | b=0 | b=1 | b=2 | b=3 | b=4 |
|---|---:|---:|---:|---:|---:|
| `cluster_zero` | 0.221 | 0.229 | 0.233 | 0.235 | 0.237 |
| `cluster_flat` | 0.247 | 0.244 | 0.241 | 0.232 | 0.230 |
| `anchored_zero` | 0.223 | 0.225 | 0.233 | 0.241 | 0.239 |
| `anchored_farthest` | 0.220 | 0.230 | 0.234 | 0.237 | 0.239 |

`cluster_flat` b=0 uses neither support nor the zero prior, so its null equals its real result (0.247). For `cluster_zero` b=3 the real subset mean is 0.969 versus null mean 0.235. Null prediction diversity is not saved by this entry point; reported diversity and per-class/per-position metrics apply to real rows only.

## Reproduction, provenance and verification

Environment: Windows, Python 3.10.6, NumPy 1.23.5, SciPy 1.10.1, scikit-learn 1.3.2. The project-local `.venv` inherits existing system packages; it is not a clean installation of all pinned historical dependencies. The [package list](../evidence/w02_kshot_naming_20261021/packages.txt) records the available environment. No GPU or neural training is used; selected epochs are **not applicable**. K-means uses seed 0 and 20 initializations, null seeds are 0–4, and the bootstrap uses seed 20261015 with 2,000 samples per real row. PCA's `random_state` is unspecified in the unchanged source; exact numerical replay is therefore not guaranteed across processes or library versions.

PowerShell command used (the Python module and experiment arguments match the requested command):

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& .\.venv\Scripts\python.exe -u -m scripts.analysis.explore_w02_kshot_naming --output-dir outputs/w02_explore_20261015/kshot_naming_v1 2>&1 |
  Tee-Object -FilePath outputs/w02_explore_20261015/kshot_naming_v1.log
# Capture $LASTEXITCODE immediately after the pipeline; the recorded value was 0.
```

Choose a new output/log name for a rerun; do not overwrite this run. Before the original command, input and output absence were checked and the [preflight](../evidence/w02_kshot_naming_20261021/preflight.json) was written. The [run record](../evidence/w02_kshot_naming_20261021/run.json), [exit record](../evidence/w02_kshot_naming_20261021/exit.json), [stdout](../evidence/w02_kshot_naming_20261021/stdout.log), [artifact hashes](../evidence/w02_kshot_naming_20261021/manifest.json) and [verification](../evidence/w02_kshot_naming_20261021/verification.json) accompany the report.

| Input/source | SHA-256 |
|---|---|
| Development dataset | `f5ec8530ab51a3114e6b7909678ec8ca2ebe1bf8073f4c326433c04af43ea764` |
| `explore_w02_kshot_naming.py` | `383a00a0ffbc3d75fb6114b0679a1f5659a6f2a0bf29aafee1d0eb492ceeefe5` |
| `src/w02_cov_fewshot.py` | `0043aacc09853cb90af98b91736c6f3fe9dd16a1ece62ca0f91bee6dc554af8a` |
| `src/v21_data.py` | `6f03523ce716046cbee555aa0990bec8b42d3e725e25bfa057e741028e8829e5` |
| Fresh `results.json` | `b7fa176a2c849cda9b281925f16877c8598ca6dc15d4de9be07e3ca6ad18f440` |

Verification covered all 64 real method/budget/subset rows, 320 null scores and 20 budget summaries. Accuracy, macro-F1, per-class recall and prediction counts were independently recomputed from every stored confusion matrix; all checks passed. Both four-label anchored agreement checks equal 1.0. The focused existing recipe tests passed **2/2** with `OPENBLAS_NUM_THREADS=1`. A prior parallel test attempt failed during NumPy DLL initialization (WinError 1114 / fatal code 0xc0000008); its failure is retained in the verification record. The same NumPy import and tests passed with one OpenBLAS thread; the exact OS-level cause was not established. No experiment failure was overwritten, and no source fix was made. This is not a full repository test run or an independent rerun from saved per-stroke predictions (the entry point does not save those predictions).

## Research limits and next decision

Three labels per recording merit a frozen follow-up recipe. The present evidence does not show reliable zero-label transfer, and two-label success depends strongly on the selected directions. No new sealed-test recipe or threshold was frozen by this task, and no test was launched. The existing development wears were already used for feature and method selection; bootstrap intervals do not restore independence. A future frozen evaluation must retain the precise sealed-test access boundary and its exclusive start marker.

This task is classification of four **already segmented** directions, using future unlabeled strokes from the same recording. It does not establish first-stroke online prediction, continuous boundary detection, handwriting transcription, open-vocabulary recognition or rotation invariance. Labels are reduced, while acquisition still contains the entire eight-stroke prefix and twelve-stroke body; collection time was not experimentally reduced. Cross-person transfer and a fully day-held-out evaluation remain untested.
