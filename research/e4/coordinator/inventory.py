#!/usr/bin/env python3
"""Inventory of the e3 edited-start continuations that e4 starts from.

For each of the 15 e3 mechanisms runs (arms keep/filt/sham at wd 1, filt/sham at wd 0):
published checkpoint update counts, whether each holds the Adam state (moments and
step), the metrics cadence, the first logged update with train accuracy 1 (a label-free
reference time), the logged common-set peak is NOT computed here (it needs held-out data).
Also checks that the wd-0 and wd-1 runs from the same start file differ in config only
in wd and in run bookkeeping.

uv run --locked python research/e4/coordinator/inventory.py
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT/'research/e3/mechanisms/runs'
BOOKKEEPING = {'run_id', 'out', 'runtime', 'host', 'session_label', 'wd', 'expect_init_sha256', 'init_weights'}


def main():
    out = {}
    for run in sorted(RUNS.iterdir()):
        cfg = json.loads((run/'config.json').read_text())
        rows = [json.loads(r) for r in (run/'metrics.jsonl').read_text().splitlines()]
        cks = sorted(int(p.stem.split('_')[1]) for p in run.glob('checkpoint_*.npz'))
        with np.load(run/f'checkpoint_{cks[-1]:06d}.npz') as z:
            state = sorted(z.files)
        fit = next((r['step'] for r in rows if r['step'] > 0 and r['train'] == 1.0), None)
        out[run.name] = dict(wd=cfg['wd'], init_weights_sha256=cfg['init_weights_sha256'], init_age=cfg['init_age'],
                             metrics_every=cfg['log_every'], metrics_rows=len(rows), last_step=rows[-1]['step'],
                             checkpoints=cks, checkpoint_keys=state, first_train_acc_1=fit,
                             manifest_sha256=hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest())
    pairs = {}
    for name in out:
        if '-wd0-' in name:
            ref = name.replace('-wd0-', '-')
            a = json.loads((RUNS/name/'config.json').read_text())
            b = json.loads((RUNS/ref/'config.json').read_text())
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in BOOKKEEPING)
            pairs[name] = dict(reference=ref, same_start=a['init_weights_sha256'] == b['init_weights_sha256'],
                               config_differences_beyond_wd_and_bookkeeping=diff)
    report = dict(runs=out, wd0_vs_wd1=pairs)
    (ROOT/'research/e4/coordinator/inventory.json').write_text(json.dumps(report, indent=1) + '\n')
    for k, v in out.items():
        print(k, v['wd'], v['first_train_acc_1'], v['checkpoints'])
    for k, v in pairs.items():
        print(k, v)


if __name__ == '__main__':
    main()
