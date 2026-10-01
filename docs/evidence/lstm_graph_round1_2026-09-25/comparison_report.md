# Four independent LSTMs with graph-attention fusion: round-one re-evaluation

This run reuses the 240 round-one windows from 2026-08-27, four physical channels, and the fixed angle split.
The original test-angle results are already known; this comparison is exploratory and requires new W02 data for independent confirmation.

## Conclusion

After staged tuning, the new model has unseen-angle macro-F1 0.339, lower than the length-matched CNN by 0.158, lower than the original cyclic-orbit CNN by 0.331.
The same four LSTMs with mean fusion achieve 0.392; graph attention scores lower in this test by 0.053.
The weakest class, up, has mean recall only 0.160.The model fits the training samples, but does not improve cross-angle recognition or meet the predefined gate.

## Model and selection procedure

Each channel has an independent bidirectional LSTM. The four nodes interact through scaled dot-product attention at each time step;
node pooling forms a temporal feature h(t), classified by an MLP or a small temporal CNN.
Angle labels are used only for splits and statistics; angle is not a model input.
First compare LSTM hidden sizes 8/16 and depths 1/2 using validation angles;
then compare 1/2 attention heads, fully connected/ring edges, and MLP/CNN classification heads.
Use two fixed seeds per tuning candidate; retrain the selected combination with the original five seeds.
Selected on validation angles: `lstm_graph_cnn`, parameters `{'hidden_size': 16, 'num_layers': 1, 'graph_heads': 2, 'graph_topology': 'ring'}`. Evaluated 12 candidate configurations.

Reuse the original 0.5–40 Hz filtering and training-angle channel scales;
antialias and resample the 500-point movement windows 4:1 to 125 points before the LSTM.
The length-matched CNN control also uses the 125-point input.

## Five-seed results

| Model | Train macro-F1 | Validation macro-F1 | Unseen-angle macro-F1 | Test accuracy | Minimum class recall on test |
|---|---:|---:|---:|---:|---:|
| New model: LSTM + graph attention | 0.958 | 0.461 | 0.339 ± 0.034 | 0.360 | 0.160 |
| Same LSTM + mean fusion | 0.887 | 0.451 | 0.392 ± 0.098 | 0.413 | 0.107 |
| Length-matched raw CNN | 0.983 | 0.479 | 0.497 ± 0.062 | 0.523 | 0.293 |
| Original raw CNN, 500 points | 0.985 | 0.437 | 0.495 ± 0.106 | 0.523 | 0.240 |
| Original cyclic-orbit CNN, 500 points | 0.940 | 0.441 | 0.670 ± 0.026 | 0.683 | 0.427 |
| Original harmonic CNN, 500 points | 0.993 | 0.531 | 0.478 ± 0.040 | 0.493 | 0.213 |

Accuracy by test angle: 90°=0.500, 210°=0.310, 330°=0.270.
Mean recall (up/down/left/right): 0.160, 0.560, 0.227, 0.493.

## Controls and interpretation limits

Shuffled-label test macro-F1 of the new model: 0.198; Rest-window test macro-F1: 0.171.
The predefined round-one gate is unseen regular-angle macro-F1 >= 0.80 and recall >= 0.65 for every class.
Graph-attention weights are internal model coefficients and cannot be interpreted directly as physical sensor contributions.
The 12 angles were collected sequentially; baseline drift and wearing state remain limitations.

Per-configuration validation scores, per-seed confusion matrices, per-sample predictions, and checkpoints are saved in this directory.
