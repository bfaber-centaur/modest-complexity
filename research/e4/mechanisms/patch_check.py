#!/usr/bin/env python3
"""Check that the restricted-decay patch leaves training unchanged without its flags (DECISIONS D14).

Compares a diagnostic rerun of a D7 replay command at the patched code with the published D7 run:
every checkpoint file's SHA-256 against the published manifest (all 89, published or not; equal
files mean bitwise-equal weights, moments and step), and every metric row except wall-clock fields.

uv run --locked --group analysis python research/e4/mechanisms/patch_check.py RERUN_DIR RUN_ID
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CLOCK = ('seconds', 'elapsed_after_diagnostics')


def rows(path):
    return {x['step']: {k: v for k, v in x.items() if k not in CLOCK}
            for x in map(json.loads, Path(path).read_text().splitlines())}


def main():
    rerun, run_id = Path(sys.argv[1]), sys.argv[2]
    pub = HERE/'runs'/run_id
    manifest = json.loads((pub/'manifest.json').read_text())['files']
    names = sorted(n for n in manifest if n.startswith('checkpoint_'))
    differ = [n for n in names if hashlib.sha256((rerun/n).read_bytes()).hexdigest() != manifest[n]['sha256']]
    mine, theirs = rows(rerun/'metrics.jsonl'), rows(pub/'metrics.jsonl')
    config = json.loads((rerun/'config.json').read_text())
    out = dict(run_id=run_id, rerun_code_commit=config['runtime']['git_commit'], rerun_tree_status=config['runtime']['git_status'],
               decay_projection=config.get('decay_projection'), checkpoints_compared=len(names), checkpoints_differing=differ,
               metric_rows_compared=len(theirs), metric_rows_differing=[s for s in theirs if mine.get(s) != theirs[s]],
               extra_rerun_files=sorted(p.name for p in rerun.glob('checkpoint_*.npz') if p.name not in manifest))
    (HERE/'results/patch_check.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
    if differ or out['metric_rows_differing'] or out['extra_rerun_files']:
        raise SystemExit('the patched code does not reproduce the published run')


if __name__ == '__main__':
    main()
