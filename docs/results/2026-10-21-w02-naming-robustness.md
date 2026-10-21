# W02 cluster naming under incomplete and unbalanced pools

Run and written on 2026-10-21. **Exploratory development results only.** This is a fresh execution of the unchanged `scripts.analysis.explore_w02_naming_robustness` module. It probes the complete, balanced recording assumption behind the [preceding k-shot naming experiment](2026-10-21-w02-kshot-naming.md).

The complete-pool three-label result does not carry over uniformly to smaller pools. With the fixed UP+DOWN+LEFT support subset, `cluster_zero` gives accuracy/F1 **0.963/0.963** on the complete recording, but **0.736/0.722** on only the first four body strokes plus those three support strokes. Retaining the other unlabeled prefix strokes and choosing the number of clusters change the outcome materially. This is a robustness stress study, not a new confirmatory result or a claim of label-free recognition.

## Run, input and test-access boundary

Code commit: `c74b3dfc4d014c5130684ad49dc524cb52000c83`. Private data repository commit: `76bca414ac24c8d17ca7e47379f1a96650668dfe`. Start: `2026-10-21T07:47:54.012664+00:00`; finish: `2026-10-21T07:55:42.9631036Z`; exit code: **0**. The command used all defaults, yielding **1,600 real rows**, **8,000 matched-null rows** and **1,088 condition-family summaries**. All 9,600 prediction vectors were saved.

Input is the same verified development NPZ as the previous experiment: 1,080 already segmented strokes from P01, Parts A/B on 2026-10-13/14; nine wears (A W01–W06, B W08/W09/W11), 54 trial recordings, 432 prefix strokes and 648 available body strokes. Class order is **UP, DOWN, LEFT, RIGHT** (IDs 0–3), with 162 available body strokes per class. The 63 REST windows are unused. Body strokes are scored in their original recording order after applying the selected perturbation.

The supplied dataset excludes sealed W07/W10/W12, and the entry point rejects them. Only development signals were loaded; no sealed W02 data, raw recordings, C170 data or historical C170 test was opened. No sealed-test start marker was created. Existing C170 test exposure is unchanged. No accepted/rejected recording was modified.

Before batch processing, the [preflight](../evidence/w02_naming_robustness_20261021/preflight.json) recorded input, cropping, channels, filters, labels, source/data hashes, output and overwrite protection. Both the run directory and adjacent log were absent. The runner uses `mkdir(exist_ok=False)`; original run files remain in `outputs/w02_explore_20261015/naming_robustness_v1` with adjacent log/preflight/exit records. No failed run was overwritten.

## What was varied

- **Body conditions (25):** all 12 strokes; first 4 or first 8; randomly retain 9 or 6 strokes (25%/50% removal, five seeds each); remove the first 1, 2 or 3 body strokes of each specified class. All four class-removal directions are included. Removing three makes that class absent from scored bodies.
- **Prefix modes:** `with` retains all eight prefix strokes, with only selected support labels revealed. `without` retains only the selected labeled support strokes. The selected support is always the first prefix stroke of each chosen class. Body-class removal does not remove prefix/support strokes, so a class absent from the body can still be present in the clustered pool.
- **Cluster count:** `fixed4`, or the best silhouette score over k=2,3,4 subject to k < pool size. In a four-stroke pool silhouette cannot choose k=4; fixed4 makes each point its own cluster. Cluster count is chosen from signals, without scoring-label access.
- **Eight pre-specified support subsets:** none; UP; UP+DOWN; UP+RIGHT; DOWN+RIGHT; UP+DOWN+LEFT; UP+DOWN+RIGHT; all four. This is not all possible subsets at each budget. In particular, only two of four three-class subsets are evaluated here; no all-subset mean is claimed.
- **Naming:** `cluster_zero` combines support constraints with a zero-model log-probability prior; `cluster_flat` uses only support constraints. Rectangular Hungarian assignment gives each cluster a distinct class. Penalties are finite; differently labeled support strokes sharing a cluster cannot all be satisfied. Support-collision counts and the chosen-k histogram are saved for each real row.

Features are unchanged: existing 500-sample (2 s at 250 Hz) grid-localized windows, no new raw cropping; third-order zero-phase 0.5–12 Hz band-pass, temporal demeaning, lags 0,5,…,30 and regularized 28×28 covariance. The k-means representation uses the selected pool's tangent reference and local feature z-scoring. The zero model uses angle-dependent cyclic channel reindexing and per-recording tangent features, with logistic regression C=0.1 and max_iter=3000. Its scaler and classifier use the other eight complete wears only. Held-out reference/scaling and zero-model inference use only the selected pool, making the analysis explicitly transductive. This runner uses no borrowed precision or PCA, even though it imports shared helpers from the previous script.

Acquisition metadata records channel map **0,1,2,3** for every development trial/REST recording. All results below are in this single map stratum. The NPZ has no map field, so validation is against acquisition metadata, not a new physical inspection. Cyclic reindexing does not establish transfer to new anatomical contacts. Recording/stroke identities are unique; the W02 archive has no marked-duplicate field.

## Baseline reproduction and subset dependence

The 16 real `full / with / fixed4` rows reproduce the corresponding previous k-shot rows in accuracy, macro-F1 and prediction counts. The nulls use a different support-permutation RNG schedule in this script, so they are newly computed controls, not a reproduction of the previous seed-by-seed nulls.

The following table gives **accuracy / macro-F1** for `cluster_zero` on the full body. The first column retains all eight prefix strokes; the second retains only labeled support. These columns score the same 648 body strokes.

| Support labels | with / fixed4 | without / fixed4 | with / silhouette | without / silhouette |
|---|---:|---:|---:|---:|
| none | 0.602 / 0.601 | 0.583 / 0.583 | 0.582 / 0.581 | 0.562 / 0.559 |
| UP | 0.912 / 0.912 | 0.872 / 0.872 | 0.878 / 0.877 | 0.823 / 0.820 |
| UP+DOWN | 0.954 / 0.954 | 0.866 / 0.866 | 0.910 / 0.909 | 0.830 / 0.828 |
| UP+RIGHT | 0.965 / 0.964 | 0.921 / 0.921 | 0.917 / 0.917 | 0.890 / 0.891 |
| DOWN+RIGHT | 0.980 / 0.980 | 0.926 / 0.926 | 0.923 / 0.923 | 0.881 / 0.881 |
| UP+DOWN+LEFT | 0.963 / 0.963 | 0.931 / 0.930 | 0.920 / 0.919 | 0.863 / 0.860 |
| UP+DOWN+RIGHT | 0.971 / 0.971 | 0.918 / 0.919 | 0.927 / 0.928 | 0.883 / 0.883 |
| UP+DOWN+LEFT+RIGHT | 0.971 / 0.971 | 0.958 / 0.958 | 0.927 / 0.928 | 0.901 / 0.902 |

## Pool-size and missing-class stress

The next table fixes support to **UP+DOWN+LEFT**, avoiding best-subset selection. Cells are **accuracy / macro-F1 over true classes present**. Random-drop cells average five removal seeds; they are not independent wear replications or confidence intervals. Each row scores only retained body strokes. Consequently, differences across rows change both the inference pool and the scored sample set; they are not paired accuracy losses on a common set of strokes.

| Body condition | Scored strokes per seed | with / fixed4 | without / fixed4 | with / silhouette | without / silhouette |
|---|---:|---:|---:|---:|---:|
| full | 648 | 0.963 / 0.963 | 0.931 / 0.930 | 0.920 / 0.919 | 0.863 / 0.860 |
| first4 | 216 | 0.935 / 0.930 | 0.736 / 0.722 | 0.898 / 0.894 | 0.764 / 0.721 |
| first8 | 432 | 0.972 / 0.972 | 0.847 / 0.845 | 0.938 / 0.936 | 0.810 / 0.807 |
| drop25 | 486 | 0.950 / 0.950 | 0.876 / 0.876 | 0.889 / 0.888 | 0.830 / 0.828 |
| drop50 | 324 | 0.942 / 0.942 | 0.808 / 0.808 | 0.898 / 0.896 | 0.773 / 0.762 |
| dropclass_UP_3 | 486 | 0.949 / 0.965 | 0.854 / 0.901 | 0.870 / 0.897 | 0.753 / 0.794 |
| dropclass_DOWN_3 | 486 | 0.957 / 0.969 | 0.897 / 0.912 | 0.895 / 0.910 | 0.815 / 0.825 |
| dropclass_LEFT_3 | 486 | 0.870 / 0.892 | 0.665 / 0.731 | 0.817 / 0.856 | 0.574 / 0.640 |
| dropclass_RIGHT_3 | 486 | 0.920 / 0.952 | 0.774 / 0.859 | 0.918 / 0.943 | 0.846 / 0.898 |

The following comparisons retain only labeled support (`without`) and use fixed4. They show the two tested three-label subsets beside the four-label reference, for both naming methods. Random-drop accuracy is averaged across five removal seeds.

| Method / body condition | UP+DOWN+LEFT | UP+DOWN+RIGHT | All four |
|---|---:|---:|---:|
| `cluster_zero` / full | 0.931 | 0.918 | 0.958 |
| `cluster_zero` / first4 | 0.736 | 0.759 | 0.856 |
| `cluster_zero` / first8 | 0.847 | 0.866 | 0.917 |
| `cluster_zero` / drop25 | 0.876 | 0.877 | 0.936 |
| `cluster_zero` / drop50 | 0.808 | 0.785 | 0.875 |
| `cluster_flat` / full | 0.929 | 0.903 | 0.946 |
| `cluster_flat` / first4 | 0.736 | 0.750 | 0.856 |
| `cluster_flat` / first8 | 0.856 | 0.833 | 0.914 |
| `cluster_flat` / drop25 | 0.875 | 0.860 | 0.916 |
| `cluster_flat` / drop50 | 0.805 | 0.775 | 0.863 |

## Class recall, prediction diversity and cluster diagnostics

Selected rows below use `cluster_zero`, **without** the other prefix strokes, **fixed4**, support **UP+DOWN+LEFT**. The full [row CSV](../evidence/w02_naming_robustness_20261021/row_metrics.csv) contains every condition/subset, both methods, all real class recalls/counts and all five null prediction-count vectors.

| Body condition | True counts U/D/L/R | Recall U/D/L/R | Predicted counts U/D/L/R | Naming failures / 54 | Support collisions / 54 | Mean null accuracy |
|---|---|---|---|---:|---:|---:|
| full | 162/162/162/162 | 0.932/0.914/1.000/0.877 | 163/157/169/159 | 5 | 2 | 0.272 |
| first4 | 65/45/70/36 | 0.615/0.600/0.914/0.778 | 42/30/73/71 | 25 | 2 | 0.248 |
| first8 | 113/110/109/100 | 0.823/0.773/1.000/0.790 | 101/94/122/115 | 12 | 1 | 0.253 |
| drop50_s0 | 82/82/85/75 | 0.890/0.732/0.976/0.733 | 77/61/102/84 | 9 | 2 | 0.269 |
| dropclass_UP_3 | 0/162/162/162 | absent/0.796/0.981/0.784 | 53/129/162/142 | 14 | 9 | 0.290 |
| dropclass_DOWN_3 | 162/0/162/162 | 0.877/absent/0.963/0.852 | 156/16/159/155 | 8 | 4 | 0.255 |
| dropclass_LEFT_3 | 162/162/0/162 | 0.716/0.722/absent/0.556 | 138/120/85/143 | 27 | 10 | 0.263 |
| dropclass_RIGHT_3 | 162/162/162/0 | 0.747/0.586/0.988/absent | 124/96/160/106 | 16 | 1 | 0.242 |

Macro-F1 explicitly averages only classes present among the scored true labels; absent-class recall is null. Predictions into an absent class can still occur, reducing recall of present classes. These three-class F1 values should not be interpreted as the original balanced four-class endpoint. Accuracy is retained throughout for comparison. Per-wear/per-angle F1 also uses the true classes present within that stratum.

Naming failure means a recording's actual body accuracy is at least 0.25 below its best permutation of predicted class IDs. On short bodies this is a coarse diagnostic. Support collision means two differently labeled support points share a cluster; it measures incompatible constraints, not every possible source of error. Counts for selected first4 rows illustrate cluster-count tradeoffs:

| Prefix / k mode | Support | Chosen k histogram (recordings) | Support collisions / 54 | Accuracy | F1 |
|---|---|---|---:|---:|---:|
| with / fixed4 | UP+DOWN+LEFT | k4: 54 | 1 | 0.935 | 0.930 |
| with / fixed4 | UP+DOWN+LEFT+RIGHT | k4: 54 | 7 | 0.935 | 0.933 |
| without / fixed4 | UP+DOWN+LEFT | k4: 54 | 2 | 0.736 | 0.722 |
| without / fixed4 | UP+DOWN+LEFT+RIGHT | k4: 54 | 16 | 0.856 | 0.846 |
| with / silhouette | UP+DOWN+LEFT | k2: 2, k3: 8, k4: 44 | 3 | 0.898 | 0.894 |
| with / silhouette | UP+DOWN+LEFT+RIGHT | k2: 2, k3: 8, k4: 44 | 14 | 0.898 | 0.894 |
| without / silhouette | UP+DOWN+LEFT | k2: 11, k3: 28, k4: 15 | 14 | 0.764 | 0.721 |
| without / silhouette | UP+DOWN+LEFT+RIGHT | k2: 8, k3: 19, k4: 27 | 32 | 0.815 | 0.799 |

Silhouette selection can merge directions and reduce the number of class names that can be assigned, even when four labeled supports exist. Four-class clustering can also split small or class-incomplete pools. Neither selection rule is guaranteed to recover the semantic class count. Support collisions and chosen-k histograms are original runner outputs; prediction replay verifies their ranges/totals, but cannot independently reconstruct them because cluster assignments were not saved.

## Wear, position and channel-map strata

The following columns use `cluster_zero / without / fixed4 / UP+DOWN+LEFT`. Each cell is **accuracy / present-class F1**. All belong to map `0,1,2,3`. Full/first8/first4 score 72/48/24 body strokes per wear, respectively. The full JSON retains per-wear/per-position metrics for all 1,600 real rows.

| Wear | Angle | Full body | First 8 | First 4 |
|---|---:|---:|---:|---:|
| A_W01 | 193° | 0.903 / 0.901 | 0.917 / 0.909 | 0.583 / 0.552 |
| A_W02 | 3° | 0.875 / 0.874 | 0.729 / 0.720 | 0.708 / 0.732 |
| A_W03 | 225° | 0.917 / 0.917 | 0.833 / 0.824 | 0.792 / 0.723 |
| A_W04 | 289° | 1.000 / 1.000 | 0.979 / 0.977 | 0.875 / 0.848 |
| A_W05 | 34° | 1.000 / 1.000 | 0.875 / 0.879 | 0.708 / 0.662 |
| A_W06 | 87° | 0.833 / 0.832 | 0.604 / 0.566 | 0.583 / 0.579 |
| B_W08 | 3° | 0.917 / 0.917 | 0.854 / 0.848 | 0.750 / 0.705 |
| B_W09 | 225° | 0.944 / 0.944 | 0.917 / 0.913 | 0.833 / 0.838 |
| B_W11 | 34° | 0.986 / 0.986 | 0.917 / 0.913 | 0.792 / 0.723 |

Angle rows pool predictions at each measured angle. Three angles occur in two wears, and the others in one. They do not isolate angle effects from wear/contact differences.

| Angle | Full body | First 8 | First 4 |
|---|---:|---:|---:|
| 3° | 0.896 / 0.895 | 0.792 / 0.783 | 0.729 / 0.732 |
| 34° | 0.993 / 0.993 | 0.896 / 0.896 | 0.750 / 0.693 |
| 87° | 0.833 / 0.832 | 0.604 / 0.566 | 0.583 / 0.579 |
| 193° | 0.903 / 0.901 | 0.917 / 0.909 | 0.583 / 0.552 |
| 225° | 0.931 / 0.930 | 0.875 / 0.870 | 0.812 / 0.791 |
| 289° | 1.000 / 1.000 | 0.979 / 0.977 | 0.875 / 0.848 |

## Nulls, provenance and verification

The zero classifier is retrained with per-recording block class permutations on training wears, and support class IDs are relabeled by a seeded four-class permutation for each held-out recording. Each real/null pair uses the same retained body and pool. This preserves within-recording class partitions and signal geometry while breaking consistent direction naming. For `cluster_flat` with no support, the null is identical to the real prediction because the method uses neither labels nor the zero prior. Null accuracy/F1/counts are reported for all five seeds; null per-wear/per-angle metrics are not in the original JSON, but predictions and sample metadata are retained for further audit.

K-means: seed 0, n_init=20. Body-removal seeds: 0–4, seeded with recording index. Null seeds: 0–4, with support permutations seeded by `[seed, recording_index]`. No bootstrap intervals, trained neural epochs or epoch selection apply. Models and preprocessing choices remain exploratory on repeatedly used development wears. There are only nine wear units and one participant; thousands of conditions are not thousands of independent experiments.

Environment is the existing project-local Python 3.10.6 environment with NumPy 1.23.5, SciPy 1.10.1 and scikit-learn 1.3.2 on Windows, inheriting system packages. OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS were each set to 1. No GPU was used. The [package snapshot](../evidence/w02_naming_robustness_20261021/packages.txt) is provided; no clean historical-environment installation is claimed.

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& .\.venv\Scripts\python.exe -u -m scripts.analysis.explore_w02_naming_robustness --output-dir outputs/w02_explore_20261015/naming_robustness_v1 2>&1 |
  Tee-Object -FilePath outputs/w02_explore_20261015/naming_robustness_v1.log
# $LASTEXITCODE was captured immediately after the pipeline: 0.
```

Use a fresh output directory and log for any rerun. Original results/prediction NPZ/log bytes and derived audit tables are included in the [evidence bundle](../evidence/w02_naming_robustness_20261021/README.md). The published prediction NPZ contains only integer class predictions and true class IDs, with no waveform signals. Sample metadata maps prediction indices to recording, wear, angle and prefix status. Artifact SHA-256 values and sizes are in the [manifest](../evidence/w02_naming_robustness_20261021/manifest.json).

| Input/source | SHA-256 |
|---|---|
| Development dataset | `f5ec8530ab51a3114e6b7909678ec8ca2ebe1bf8073f4c326433c04af43ea764` |
| `scripts/analysis/explore_w02_naming_robustness.py` | `2094fdc141e87ad04df828b2a1fb791cef0f05b38882e9fce548df54b06cd9fb` |
| `scripts/analysis/explore_w02_kshot_naming.py` | `383a00a0ffbc3d75fb6114b0679a1f5659a6f2a0bf29aafee1d0eb492ceeefe5` |
| `src/w02_cov_fewshot.py` | `0043aacc09853cb90af98b91736c6f3fe9dd16a1ece62ca0f91bee6dc554af8a` |
| `src/v21_data.py` | `6f03523ce716046cbee555aa0990bec8b42d3e725e25bfa057e741028e8829e5` |
| Fresh `results.json` | `f0b6345de0ac674eae0b18e1c7216e11ac95da4c83f4abfffff7748f4799c37c` |
| Fresh `predictions.npz` | `9474e87989c7010596e46cb3910c295647e5dc9eea5172b9b19c847b3e451721` |

Verification independently reconstructed every scored mask for all 25 perturbations, recomputed all real accuracy/F1/recalls/counts/per-wear/per-angle metrics and naming failures from saved predictions, and recomputed every null accuracy/F1/count vector. All 1,600 real rows, 8,000 null rows and 1,088 summary rows passed, including 16 real baseline comparisons with the previous run. Derived confusion matrices are retained. All original predictions have 1,080 entries, with -1 for unscored strokes; no prefix stroke is scored. Input/source hashes match the run records. The existing focused covariance-recipe tests passed **2/2** (1.40 s). This is saved-prediction replay and focused testing, not a second model execution, a full repository test suite or a sealed evaluation.

## Interpretation and limits

The preceding k-shot result relied on enough unlabeled strokes to establish recording-local geometry and naming. Shortening the pool or omitting the unused prefix weakens several three-label configurations. Dynamic cluster count is an additional modeling choice, not an automatic solution to missing classes. Four labels can help anchor naming in smaller pools, but do not eliminate collisions or all errors. The result should guide a future frozen recipe and calibration/acquisition design rather than set a new acceptance claim.

First-m and random-drop conditions change the class balance naturally within the retained sample; class-targeted removal is a label-aware synthetic stress intervention, not a measured natural missing-class rate. The retained prefix/support can still reveal classes removed from the body. This experiment does not isolate how much degradation comes from reduced pool geometry versus changing the scoring population.

All folds mix training wears from both acquisition days. This is same-participant exploratory rewear evidence over two days, with transductive within-recording adaptation. It is not a day-held-out or cross-person experiment, proof of mathematical/physical rotation invariance, first-stroke online control, continuous detection, handwriting transcription or open-vocabulary recognition. No new test recipe or success threshold was frozen, and the sealed holdout remains unopened.
