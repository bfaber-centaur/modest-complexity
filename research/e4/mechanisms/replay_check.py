#!/usr/bin/env python3
"""Check each D7 dense replay against the e3 run it replays (DECISIONS D7).

For every checkpoint update the replay shares with the e3 run, all arrays (weights, Adam moments,
step) must be bitwise equal. e3 checkpoints not published in Git are read from the local e3 run
directory when present, after their SHA-256 matches the e3 manifest. Metric rows up to the replay's
last update must be equal except the wall-clock fields. Config differences are listed.

uv run --locked --group analysis python research/e4/mechanisms/replay_check.py --local runs/e4 [--e3-local runs/e3]
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT/'research/e4/mechanisms'
E3 = ROOT/'research/e3/mechanisms/runs'
CLOCK = ('seconds', 'elapsed_after_diagnostics')
BOOKKEEPING = ('out', 'run_id', 'purpose', 'session_label', 'steps', 'checkpoint_steps', 'runtime', 'host',
               'diagnostic_schedule')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replays():
    plan = json.loads((ROOT/'research/e4/plan.json').read_text())
    return [r for r in plan['runs'] if r['owner'] == 'mechanisms' and r['purpose'] == 'diagnostic-replay']


def e3_checkpoint(run_id, name, e3_local):
    """The e3 file, from Git if published, else from the local run directory, hash-checked."""
    entry = json.loads((E3/run_id/'manifest.json').read_text())['files'].get(name)
    if entry is None:
        return None
    path = E3/run_id/name if entry['published'] else e3_local/run_id/name
    if not path.exists():
        return None
    if sha256(path) != entry['sha256']:
        raise ValueError(f'{path}: hash differs from the e3 manifest')
    return path


def rows(path):
    return {x['step']: {k: v for k, v in x.items() if k not in CLOCK}
            for x in map(json.loads, Path(path).read_text().splitlines())}


def check(r, local, e3_local):
    run, ref = local/r['run_id'], r['replay_of']
    names = sorted(p.name for p in run.glob('checkpoint_*.npz'))
    compared, unavailable, unequal = [], [], []
    for name in names:
        path = e3_checkpoint(ref, name, e3_local)
        if path is None:
            continue
        with np.load(run/name) as x, np.load(path) as y:
            same = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
        (compared if same else unequal).append(int(name[11:17]))
    for name, entry in json.loads((E3/ref/'manifest.json').read_text())['files'].items():
        step = int(name[11:17]) if name.startswith('checkpoint_') else None
        if step is not None and step <= r['steps'] and e3_checkpoint(ref, name, e3_local) is None:
            unavailable.append(step)
    mine, theirs = rows(run/'metrics.jsonl'), rows(E3/ref/'metrics.jsonl')
    metric_diff = [s for s in mine if mine[s] != theirs.get(s)]
    c1, c2 = (json.loads(p.read_text()) for p in (run/'config.json', E3/ref/'config.json'))
    config_diff = sorted(k for k in set(c1) | set(c2) if k not in BOOKKEEPING and c1.get(k) != c2.get(k))
    runtime = json.loads((run/'runtime.json').read_text())
    return dict(run_id=r['run_id'], replay_of=ref, status=runtime.get('status'),
                completed_steps=runtime.get('completed_steps'), checkpoints=len(names),
                bitwise_equal_steps=compared, unequal_steps=unequal, e3_checkpoints_unavailable=unavailable,
                metric_rows_compared=len(mine), metric_rows_differing=metric_diff,
                config_keys_differing=config_diff, git_commit=c1['runtime']['git_commit'],
                git_status=c1['runtime']['git_status'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--local', type=Path, required=True, help='Local root of the replay run directories')
    ap.add_argument('--e3-local', type=Path, default=ROOT/'runs/e3')
    ap.add_argument('--out', type=Path, default=HERE/'results/replay_check.json')
    args = ap.parse_args()
    out = [check(r, args.local, args.e3_local) for r in replays()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    for x in out:
        print(x['run_id'], len(x['bitwise_equal_steps']), 'equal', x['unequal_steps'], 'unequal,',
              len(x['metric_rows_differing']), 'metric rows differ, config', x['config_keys_differing'])
    if any(x['unequal_steps'] or x['metric_rows_differing'] or x['config_keys_differing'] for x in out):
        raise SystemExit('a replay differs from its e3 run')


if __name__ == '__main__':
    main()
