"""Compile the dated English report from verified fresh-run evidence."""
import json
from pathlib import Path

EVIDENCE = Path('docs/evidence/v22_ablation_20261012')
summary = json.loads((EVIDENCE / 'summary.json').read_text())
audit = json.loads((EVIDENCE / 'verified_run_evidence.json').read_text())
preflight = json.loads((EVIDENCE / 'preflight.json').read_text(encoding='utf-8-sig'))
assert audit['verified_runs'] == 81


def metric(window, norm, subset, mode, null=False):
    protocol = 'w01_transfer' if subset == 'w01' else 'fixed_split'
    key = f'{protocol}__{window}__{norm}' + ('__null' if null else '')
    return summary[key]['subsets'][subset]['modes'][mode]


def table(variants):
    lines = ['| Window | Normalization | C170 old val ZERO / CAL | C170 exposed test ZERO / CAL | W01 ZERO / CAL |',
             '|---|---|---:|---:|---:|']
    for window, norm in variants:
        cells = []
        for subset in ['c170_val', 'c170_test', 'w01']:
            z, c = [metric(window, norm, subset, m)['seed_mean'] for m in ['zero', 'cal_tau1']]
            cells.append(f'{z:.3f} / {c:.3f}')
        lines.append(f'| {window} | {norm} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


baseline_z = metric('full', 'shared', 'w01', 'zero')['seed_mean']
baseline_c = metric('full', 'shared', 'w01', 'cal_tau1')['seed_mean']
real = {k: v for k, v in summary.items() if not k.endswith('__null')}
w01 = {k: v for k, v in real.items() if k.startswith('w01_transfer')}
best_zero = max(w01, key=lambda k: w01[k]['subsets']['w01']['modes']['zero']['seed_mean'])
best_cal = max(w01, key=lambda k: w01[k]['subsets']['w01']['modes']['cal_tau1']['seed_mean'])
_, bz_window, bz_norm = best_zero.split('__')
_, bc_window, bc_norm = best_cal.split('__')
bz = metric(bz_window, bz_norm, 'w01', 'zero')
bc = metric(bc_window, bc_norm, 'w01', 'cal_tau1')

lines = [
    '# v22 time-window and amplitude-normalization ablations (fresh exploratory CUDA runs)',
    '',
    'Date: 2026-10-12 (Asia/Hong_Kong). Remote source pulled with `git pull --ff-only` to `79a03c5`. '
    'This report describes new local execution, separate from the historical v21 Apple M2 scores.',
    '',
    f'All **81/81 planned runs completed**: 66 true-label runs (11 variants × 2 protocols × 3 seeds) and '
    '15 matched block-permuted nulls (11 C170 variants and the first 4 W01 variants). '
    'Both requested ablations are implemented by the single v22 grid. No W02 acquisition, S2 freeze, or W02 evaluation was performed.',
    '',
    f'The fresh full/shared baseline has W01 ZERO macro-F1 **{baseline_z:.3f}** and same-recording CAL τ=1 '
    f'**{baseline_c:.3f}**. The largest W01 ZERO mean is **{bz["seed_mean"]:.3f}** '
    f'for `{bz_window}/{bz_norm}`; the largest CAL mean is **{bc["seed_mean"]:.3f}** '
    f'for `{bc_window}/{bc_norm}`. These are exploratory rankings across reused data, not independent model-selection evidence.',
    '',
    '## Execution and verification',
    '',
    '- Runtime: Python 3.10.18, PyTorch 2.6.0+cu124, NumPy 1.26.4, SciPy 1.15.2, scikit-learn 1.7.0, '
    'Windows, RTX 2080 Ti, NVIDIA driver 591.86. Four concurrent CUDA jobs, two CPU threads per job.',
    '- B2 signed CNN; same-group episodic weight 1; training τ=4; augmentation enabled; batch size 64; '
    'AdamW lr 0.001, weight decay 0.001; fixed 120 epochs. The final epoch 120 is selected for every run; '
    'held-out scores do not choose checkpoints. Real seeds: 17, 29, 43; null model seed: 17; block-permutation seed: 17.',
    '- Fresh relevant tests: 18 passed. Full suite: 167 passed. The full/shared CUDA smoke retry completed '
    'one epoch; its scores are excluded. Original smoke directory is preserved as failed/incomplete.',
    '- The first CUDA smoke failed before evaluation because NumPy default integer labels are int32 on Windows. '
    'The C170 and W01 loaders now return explicit int64 labels. Two regression cases failed on cross-entropy '
    'before the fix and passed afterward. This changes dtype, not label values, split, training budget, or preprocessing. '
    'The episode-loss test fixture also uses explicit int64 labels.',
    '- Fresh audit checked all 81 completion markers, full 1–120 epoch curves with finite losses, source/data hashes, '
    'prediction counts and valid class IDs, calibration exclusion, duplicate exclusion, and recalculated macro-F1. '
    'Portable evidence contains per-run and pooled per-position metrics, per-class recall, prediction counts, '
    'channel-map strata and artifact hashes. Local checkpoints and raw predictions remain in the ignored output root.',
    '',
    'Run command:',
    '',
    '```powershell',
    "& 'C:/Users/56569/Miniforge3/envs/ML2025/python.exe' -m scripts.launchers.launch_v22_window_norm_ablation --output-root outputs/v22_window_norm_ablation_cuda_20261012 --workers 4 --device cuda --cpu-threads 2",
    '```',
    '',
    '## Data, preprocessing and access boundary',
    '',
    'Class order is **UP, DOWN, LEFT, RIGHT** (IDs 0–3). Original recordings and rejected takes were unchanged. '
    'Inputs are existing raw two-second windows, 500 samples at 250 Hz, four stored sensor channels. '
    'C170 has 1,600 windows, with its one marked copied window excluded. W01 has 240 windows, 60 per class.',
    '',
    '- Fixed split: 879 training windows from positions 0/10/20/30/50/60/80/100/120/140/160 mm; '
    '70 mm excluded. Old-validation positions 5/40/90/130 mm and exposed-test positions 15/25/110/150 mm '
    'are evaluated together after training, then reported separately (288 scored windows each).',
    '- W01 transfer: train on all 1,599 nonduplicate C170 windows, including 70 mm; evaluate '
    'all 12 W01 angles (0–330° in 30° steps), 144 scored windows.',
    '- CAL support is repetitions 1–2 of each direction (8 strokes/group); score later repetitions only. '
    'CAL uses true evaluation support labels even for the training-label nulls. C170/W01 each direction '
    'occupies its own recording, so support and scored strokes share recording identity. CALNB uses another '
    'held-out position 10–20 mm away. UNSUP is transductive, using other unlabeled scored strokes in the group.',
    '- Each visible span receives its own third-order 0.5–40 Hz Butterworth zero-phase filter and temporal demeaning; '
    'cut edges are tapered over 20 samples and hidden samples zero-padded. Prefixes retain [0,200/250/300); '
    'return-only retains [250,500). Outbound-valley chooses the envelope valley between two strongest bursts '
    'using the full window (fallback 250), so it is oracle segmentation. Even fixed prefixes come from '
    'already localized peak-centred windows and do not establish causal continuous detection.',
    '- Shared normalization uses a training-fitted scalar. Window-global/window-sensor RMS use each individual '
    'visible input. **group_sensor uses each evaluation group’s full unlabeled input pool, including scored windows, '
    'to fit per-channel medians.** The user explicitly instructed this run to follow the original v22 design '
    'and authorized this transductive diagnostic on 2026-10-12. It must not be described as train-only preprocessing '
    'or deployable online without an equivalent input pool.',
    '- The loaders decode the full NPZ before optimization; only training groups and their labels enter training. '
    'No held-out checkpoint selection occurs. Evaluation labels enter CAL support and subsequent scoring; '
    'group_sensor and UNSUP access evaluation inputs as described. C170 registered test was already exposed '
    'by F2 on 2026-09-29 and by later development. W01 and old validation were repeatedly used. '
    'No untouched holdout is accessed here.',
    '',
    'C170 EXG maps 0,2,4,6 and 0,1,2,3 were previously user-confirmed as physical S0,S1,S2,S3; '
    'the map transition crosses 70 mm. W01 stores EXG 0,1,2,3; the current data audit validates channel columns '
    'but does not independently revalidate anatomical sensor order. B2 is a signed CNN without C4 pooling; '
    'neither cyclic channel indexing nor these scores prove physical rotation invariance.',
    '',
    'Data SHA-256:',
    '',
    f'- C170: `{preflight["data_sha256"]["c170"]}`',
    f'- W01: `{preflight["data_sha256"]["w01"]}`',
    '',
    'Both match the existing S1 freeze manifest. Executed source hashes are in each fresh run manifest and '
    'the verified evidence. Historical source hashes and the English conversion ledger are unchanged.',
    '',
    '## Time-window ablation',
    '',
    'Values are mean macro-F1 over three seeds. CAL means CAL τ=1. All rows in this table use the shared training scale.',
    '',
    table([(w, 'shared') for w in ['full', 'prefix_200', 'prefix_250', 'prefix_300', 'outbound_valley', 'return_only']]),
    '',
    '## Normalization ablation and combinations',
    '',
    table([(w, n) for w, n in [('full', 'shared'), ('full', 'window_global'), ('full', 'window_sensor'),
                              ('full', 'group_sensor'), ('outbound_valley', 'window_global'), ('outbound_valley', 'group_sensor')]]),
    '',
    'The group_sensor rows have the additional transductive access described above. Full seed means/SD '
    '(population SD, ddof=0), majority votes, UNSUP τ=1, CALNB τ=4, nulls and opposite-direction error shares '
    'are in [the complete score table](../evidence/v22_ablation_20261012/score_table.md). '
    'All evaluated τ values 1/4/16 and all position/channel strata are in the verified evidence.',
    '',
    '## Interpretation of the two ablations',
    '',
    f'1. **Shortening the window does not consistently improve transfer.** The 200-sample prefix gives '
    f'the largest W01 ZERO mean ({metric("prefix_200", "shared", "w01", "zero")["seed_mean"]:.3f}), '
    f'but its W01 CAL mean drops to {metric("prefix_200", "shared", "w01", "cal_tau1")["seed_mean"]:.3f}. '
    'It also lowers C170 exposed-test ZERO/CAL relative to full/shared, and has no planned W01 null. '
    'This is a lead for a new experiment, not a validated replacement.',
    f'2. **The return half contains substantial same-recording calibration information.** Return-only '
    f'W01 CAL is {metric("return_only", "shared", "w01", "cal_tau1")["seed_mean"]:.3f}, '
    f'while ZERO is {metric("return_only", "shared", "w01", "zero")["seed_mean"]:.3f}. '
    'Full-window calibration therefore cannot be attributed exclusively to the outward movement. '
    'The oracle outbound cut improves C170 old-validation CAL but worsens exposed-test and W01 CAL.',
    '3. **Normalization helps calibration more than zero-calibration semantics.** Full/group_sensor '
    'has the largest CAL seed mean among full-window normalization variants on all three reported subsets, but its additional transductive '
    'input pool is part of that setting. Its W01 ZERO remains low; window-global and window-sensor '
    'normalization also do not produce strong ZERO transfer. Seed differences are not confidence '
    'intervals for new wears, days or people.',
    f'4. **Zero-calibration cross-session performance remains weak.** Even the highest exploratory '
    f'W01 ZERO mean is {bz["seed_mean"]:.3f}; no ablation establishes the project’s physical '
    'rotation-generalization gate. The strong CAL scores still require labeled support from the '
    'same direction recording, and the matched CAL nulls must be considered before attributing '
    'that performance to correctly labeled global training.',
    '',
    '## Matched nulls, recall and prediction diversity',
    '',
    'Nulls permute complete direction-recording label blocks within every training wear group, preserving '
    'within-block repetition dependence. One null seed per planned variant is a diagnostic rather than '
    'a permutation distribution or a significance test. Seven W01 variants have no planned null; '
    'they must not inherit another variant’s control result.',
    '',
    'The table shows the three-seed true-label mean and, separately, the matched seed-17 '
    'true/null pair. A strong CAL null shows that labeled local support can support template matching '
    'without correctly labeled global training; a single matched null does not establish equivalence.',
    '',
    '| Window / norm | Subset | True mean ZERO / CAL | True seed 17 ZERO / CAL | Null seed 17 ZERO / CAL |',
    '|---|---|---:|---:|---:|',
]
for window, norm in [('full', 'shared'), ('outbound_valley', 'shared'), ('full', 'window_global'), ('full', 'group_sensor')]:
    for subset in ['c170_val', 'c170_test', 'w01']:
        z, c = [metric(window, norm, subset, m)['seed_mean'] for m in ['zero', 'cal_tau1']]
        sz, sc = [metric(window, norm, subset, m)['per_seed'][0] for m in ['zero', 'cal_tau1']]
        nz, nc = [metric(window, norm, subset, m, True)['seed_mean'] for m in ['zero', 'cal_tau1']]
        lines.append(f'| {window} / {norm} | {subset} | {z:.3f} / {c:.3f} | {sz:.3f} / {sc:.3f} | {nz:.3f} / {nc:.3f} |')
lines += ['', 'Recall and prediction counts below are pooled over the three seeds, in class order UP/DOWN/LEFT/RIGHT. '
          'They describe individual-seed predictions, not the majority-vote predictions.', '',
          '| Variant | W01 mode | Recall UP / DOWN / LEFT / RIGHT | Prediction counts | Opposite-direction share of errors |',
          '|---|---|---|---|---:|']
for window, norm in dict.fromkeys([('full', 'shared'), (bz_window, bz_norm), (bc_window, bc_norm)]):
    for mode in ['zero', 'cal_tau1']:
        m = metric(window, norm, 'w01', mode)
        recalls = ' / '.join(f'{m["recall"][c]:.3f}' for c in ['UP', 'DOWN', 'LEFT', 'RIGHT'])
        counts = ' / '.join(str(m['prediction_counts'][c]) for c in ['UP', 'DOWN', 'LEFT', 'RIGHT'])
        lines.append(f'| {window}/{norm} | {mode} | {recalls} | {counts} | {m["opposite_error_share"]:.3f} |')
lines += ['', '## Position and channel-map detail', '',
          'The following compact view compares fresh baseline and the W01 ZERO-leading variant. '
          'These are macro-F1 values recomputed from predictions pooled across three seeds, not seed-mean F1. '
          'Every other variant, null, mode and per-class detail is preserved in the evidence.', '',
          '| Variant | Position / angle | ZERO F1 | CAL τ=1 F1 |', '|---|---|---:|---:|']
for window, norm in dict.fromkeys([('full', 'shared'), (bz_window, bz_norm)]):
    for protocol in ['fixed_split', 'w01_transfer']:
        values = audit['aggregate'][f'{protocol}__{window}__{norm}__False']
        for group, modes in values['per_position_pooled'].items():
            lines.append(f'| {window}/{norm} | {group} | {modes["zero"]["macro_f1"]:.3f} | {modes["cal_tau1"]["macro_f1"]:.3f} |')
lines += ['', 'Pooled channel-map strata (macro-F1):', '',
          '| Variant | Protocol | EXG map | ZERO | CAL τ=1 |', '|---|---|---|---:|---:|']
for window, norm in dict.fromkeys([('full', 'shared'), (bz_window, bz_norm)]):
    for protocol in ['fixed_split', 'w01_transfer']:
        values = audit['aggregate'][f'{protocol}__{window}__{norm}__False']
        for mapping, modes in values['channel_map_strata_pooled'].items():
            lines.append(f'| {window}/{norm} | {protocol} | {mapping} | {modes["zero"]["macro_f1"]:.3f} | {modes["cal_tau1"]["macro_f1"]:.3f} |')
lines += ['', 'Map strata are confounded with C170 position/acquisition order; they are diagnostics, '
          'not controlled causal comparisons of channel wiring.', '',
          '## Evidence and preservation', '',
          '- [Preflight record](../evidence/v22_ablation_20261012/preflight.json): inputs, preprocessing, labels, environment, output policy and access boundary.',
          '- [Score summary JSON](../evidence/v22_ablation_20261012/summary.json) and [full score table](../evidence/v22_ablation_20261012/score_table.md).',
          '- [Verified run evidence](../evidence/v22_ablation_20261012/verified_run_evidence.json): all 81 run manifests, final epochs, data/source/artifact hashes, per-position metrics and channel-map strata.',
          '- [Verification record](../evidence/v22_ablation_20261012/verification.json). Local output root: `outputs/v22_window_norm_ablation_cuda_20261012` (ignored by Git).',
          '', 'No historical/incomplete artifacts were overwritten or relabeled. The original failed smoke remains '
          '`outputs/v22_cuda_smoke_20261012`; its completed retry remains separate and is excluded from all tables.',
          '', 'This evidence addresses presegmented four-direction classification on existing single-participant '
          'recordings. C170 is same-session position transfer. W01 is a different session/day; it is reused '
          'cross-session evidence with same-recording calibration, not a new independent rewearing trial. '
          'There is no fresh mathematical invariance claim, untouched physical rewearing validation, cross-person '
          'evidence, continuous detection, transcription or open-vocabulary recognition. A new untouched W02 '
          'holdout is still required for confirmatory claims.', '']
output = Path('docs/results/2026-10-12-v22-window-normalization-ablation.md')
output.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
print(json.dumps({'best_w01_zero': best_zero, 'best_w01_cal': best_cal, 'report': str(output)}))
