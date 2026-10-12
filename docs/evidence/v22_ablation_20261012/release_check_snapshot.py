import hashlib
import json
import re
from pathlib import Path

root = Path('docs/evidence/v22_ablation_20261012')
verification = json.loads((root / 'verification.json').read_text(encoding='utf-8-sig'))
for name, expected in verification['evidence_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
report = Path(verification['report']['path'])
assert hashlib.sha256(report.read_bytes()).hexdigest() == verification['report']['sha256']
for doc in [report, root / 'score_table.md', Path('README.md'), Path('docs/README.md'),
            Path('docs/EXPERIMENT_RESULTS.md'), Path('docs/evidence/README.md')]:
    for target in re.findall(r'\]\(([^)]+)\)', doc.read_text(encoding='utf-8')):
        if not target.startswith(('http:', 'https:', '#')):
            assert (doc.parent / target.split('#')[0]).exists(), (str(doc), target)
audit = json.loads((root / 'verified_run_evidence.json').read_text())
assert audit['verified_runs'] == 81
assert len(audit['runs']) == 81
assert all(r['selected_epoch'] == 120 for r in audit['runs'])
summary = json.loads((root / 'summary.json').read_text())
assert sum(v['n_runs'] for k, v in summary.items() if not k.endswith('__null')) == 66
assert sum(v['n_runs'] for k, v in summary.items() if k.endswith('__null')) == 15
assert not any(Path('outputs/v22_window_norm_ablation_cuda_20261012').glob('*__retry*'))
print('Release verification passed: evidence/report hashes, Markdown targets, 81 runs, 66 real/15 null, final epoch 120.')
