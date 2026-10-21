#!/usr/bin/env python3
"""rank_checks.py's 30% vs 50% comparison (483 common pairs, 50% mask) with the extra
onset checkpoints of DECISIONS D9 added: research/e2/{coordinator,dynamics}/e1-checkpoints/.
Each extra checkpoint is first checked against the SHA-256 in its run's e1 manifest and
against the step stored inside it.

uv run --locked python research/e2/audit/shares_dense.py OUT_DIR
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rank_checks as rc  # noqa: E402

EXTRA = [rc.root/'research/e2/coordinator/e1-checkpoints', rc.root/'research/e2/dynamics/e1-checkpoints']


def path(run, step):
    return next(p for d in (run, *(e/run.name for e in EXTRA)) if (p := d/f'checkpoint_{step:06d}.npz').exists())


def check_extra():
    out = []
    for f in sorted(f for e in EXTRA for f in e.glob('*/checkpoint_*.npz')):
        run = next(r for r in rc.RUNS if r.name == f.parent.name)
        manifest = json.loads((run/'manifest.json').read_text())['files'][f.name]
        with np.load(f) as z:
            step = int(z['step'])
        out.append(dict(file=str(f.relative_to(rc.root)), sha256_matches_manifest=hashlib.sha256(f.read_bytes()).hexdigest() ==
                        manifest['sha256'], bytes_match=f.stat().st_size == manifest['bytes'],
                        step_matches_name=step == int(f.stem.split('_')[1]), not_in_run_dir=not (run/f.name).exists()))
    return out


rc.steps = lambda run: sorted({int(f.stem.split('_')[1]) for d in (run, *(e/run.name for e in EXTRA))
                               for f in d.glob('checkpoint_*.npz')})


def logits(run, step, d):
    with np.load(path(run, step)) as z:
        return rc.ind.forward((z['U'], z['V'], z['W']), *rc.ind.one_hot_inputs(d['a'], d['b'], rc.P))[0]


rc.logits = logits

if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    checks = check_extra()
    (out/'extra-checkpoints.json').write_text(json.dumps(dict(
        files=len(checks), all_pass=all(all(v for k, v in c.items() if k != 'file') for c in checks), checks=checks)) + '\n')
    rows, chance = rc.shares()
    (out/'shares-dense.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in [chance] + rows))
