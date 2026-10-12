"""Verify completed fresh runs and export portable research evidence without waveforms."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from scripts.launchers.launch_v22_window_norm_ablation import planned_runs

ROOT = Path('outputs/v22_window_norm_ablation_cuda_20261012')
DEST = Path('docs/evidence/v22_ablation_20261012')
CLASSES = ['UP', 'DOWN', 'LEFT', 'RIGHT']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(rows, mode):
    y = np.array([int(r['true']) for r in rows])
    p = np.array([int(r[mode]) for r in rows])
    return {'n': len(y), 'macro_f1': float(f1_score(y, p, labels=range(4), average='macro')),
            'recall': {c: float(np.mean(p[y == i] == i)) for i, c in enumerate(CLASSES)},
            'prediction_counts': dict(zip(CLASSES, np.bincount(p, minlength=4).tolist())),
            'n_predicted_classes': int(len(np.unique(p)))}


def main():
    complete = json.loads((ROOT / 'GRID_COMPLETE.json').read_text())
    assert not complete['failed']
    preflight = json.loads((DEST / 'preflight.json').read_text(encoding='utf-8-sig'))
    assert not (ROOT / 'GRID_FAILED.json').exists()
    for kind, path in preflight['input'].items():
        assert sha(Path(path)) == preflight['data_sha256'][kind]
    with np.load('data/processed_c170_20260926/dataset.npz') as data:
        maps = np.array([','.join(map(str, row)) for row in data['channel_map']])
        duplicate = data['is_duplicate']
    runs, grouped = [], {}
    for name, _ in planned_runs():
        path = ROOT / name
        assert (path / 'RUN_COMPLETE.json').is_file(), name
        manifest = json.loads((path / 'run_manifest.json').read_text())
        args = manifest['arguments']
        assert args['epochs'] == '120'
        assert args['device'] == 'cuda'
        assert manifest['n_train'] == (879 if args['protocol'] == 'fixed_split' else 1599)
        for kind, value in manifest['data_sha256'].items():
            assert value == preflight['data_sha256'][kind], (name, kind)
        for source, value in manifest['source_sha256'].items():
            assert sha(Path(source)) == value, (name, source)
        with (path / 'learning_curve.csv').open() as handle:
            curve = list(csv.DictReader(handle))
        assert [int(r['epoch']) for r in curve] == list(range(1, 121))
        assert all(np.isfinite(float(r[k])) for r in curve for k in ['ce', 'episode', 'elapsed_s'])
        with (path / 'predictions.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == (576 if args['protocol'] == 'fixed_split' else 144), name
        assert all(int(r['repetition']) > 2 for r in rows)
        if args['protocol'] == 'fixed_split':
            assert all(not duplicate[int(r['sample_index'])] for r in rows)
        modes = [k for k in rows[0] if k == 'zero' or '_tau' in k]
        assert all(0 <= int(r[m]) < 4 for r in rows for m in modes)
        metrics = json.loads((path / 'metrics.json').read_text())
        evaluation = 'c170_heldout' if args['protocol'] == 'fixed_split' else 'w01'
        for mode in modes:
            assert abs(describe(rows, mode)['macro_f1'] - metrics[evaluation][mode]['macro_f1']) < 1e-12
        per_position = {g: {m: describe([r for r in rows if r['group'] == g], m) for m in modes}
                        for g in sorted({r['group'] for r in rows})}
        strata = {}
        for row in rows:
            mapping = maps[int(row['sample_index'])] if row['source'] == 'C170' else '0,1,2,3'
            strata.setdefault(str(mapping), []).append(row)
        channel_strata = {s: {m: describe(rs, m) for m in modes} for s, rs in strata.items()}
        runs.append({'run': name, 'manifest': manifest, 'selected_epoch': 120,
                     'artifact_sha256': {f: sha(path / f) for f in ['checkpoint.pt', 'predictions.csv', 'metrics.json',
                         'run_manifest.json', 'learning_curve.csv', 'RUN_COMPLETE.json']},
                     'per_position': per_position, 'channel_map_strata': channel_strata})
        key = (args['protocol'], args['window'], args['norm'], args['null_seed'] != 'None')
        grouped.setdefault(key, []).append((int(args['seed']), rows))
    assert len(runs) == 81
    aggregate = {}
    for key, values in grouped.items():
        values.sort()
        identity = lambda rs: [(r['source'], r['sample_index'], r['group'], r['true']) for r in rs]
        assert all(identity(rs) == identity(values[0][1]) for _, rs in values), key
        assert [seed for seed, _ in values] == ([17] if key[-1] else [17, 29, 43]), key
        modes = [k for k in values[0][1][0] if k == 'zero' or '_tau' in k]
        pooled = [row for _, rs in values for row in rs]
        positions = sorted({r['group'] for r in pooled})
        aggregate['__'.join(map(str, key))] = {
            'seeds': [seed for seed, _ in values],
            'per_position_pooled': {g: {m: describe([r for r in pooled if r['group'] == g], m) for m in modes}
                                    for g in positions},
            'channel_map_strata_pooled': {s: {m: describe([r for r in pooled if
                (maps[int(r['sample_index'])] if r['source'] == 'C170' else '0,1,2,3') == s], m) for m in modes}
                for s in sorted({maps[int(r['sample_index'])] if r['source'] == 'C170' else '0,1,2,3' for r in pooled})}}
    payload = {'verified_runs': len(runs), 'classes': CLASSES, 'runs': runs, 'aggregate': aggregate,
               'additional_source_sha256': {str(p): sha(p) for p in [
                   Path('src/c170_complex_orbit_model_v11.py'), Path('src/c170_harmonic_phase_model_v4.py'),
                   Path('src/c170_shared_scale_preprocessing_v3.py'),
                   Path('scripts/launchers/launch_v22_window_norm_ablation.py'),
                   Path('scripts/analysis/summarize_v22_window_norm_ablation.py')]}}
    output = DEST / 'verified_run_evidence.json'
    assert not output.exists()
    output.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps({'verified_runs': len(runs), 'evidence_bytes': output.stat().st_size,
                      'total_gpu_process_runtime_hours': sum(r['manifest']['runtime_s'] for r in runs) / 3600}))


if __name__ == '__main__':
    main()
