"""Compare the c2-co-verify-* runs (c2train resumed from an r1 ckpt-1000) with the r1 run they
continue: r1's metrics keys at every logged step 1000..10000, and every array of ckpt-10000
(weights, Adam moments, step count) against the recovered original.
uv run --locked python research/composition-c2/coordinator/verify_resume.py"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research'/'regimes'/'common'))
import nets  # noqa: E402

PAIRS = {'c2-co-verify-inter-s51200-a1': ('composition', 'r1-comp-rnn-inter-s51200-a1'),
         'c2-co-verify-final-s51200-a1': ('composition', 'r1-comp-rnn-final-s51200-a1'),
         'c2-co-verify-untiedro-s51200-a1': ('review', 'r1-rev-comp-rnn-untied-s51200-a1')}
KEYS = ('step', 'train', 'held', 'held_final_pred')
out = {}
for run, (lane, src) in PAIRS.items():
    ours = [{k: r[k] for k in KEYS} for r in map(json.loads, (ROOT/'runs'/'c2'/run/'metrics.jsonl').read_text().splitlines())]
    theirs = [{k: r[k] for k in KEYS} for r in map(json.loads, (ROOT/'research'/'regimes'/lane/'runs'/src/'metrics.jsonl').read_text().splitlines())]
    theirs = [r for r in theirs if r['step'] >= 1000]
    (pa, sa, _), (pb, sb, _) = nets.load(ROOT/'runs'/'c2'/run/'ckpt-10000.npz'), nets.load(ROOT/'research'/'composition-c2'/'recovered'/src/'ckpt-10000.npz')
    arrays = sa['t'] == sb['t'] and pa.keys() == pb.keys() and all(
        np.array_equal(x[k], y[k]) for x, y in ((pa, pb), (sa['m'], sb['m']), (sa['v'], sb['v'])) for k in x)
    out[run] = dict(source=src, logged_steps=len(ours), metrics_identical=ours == theirs, ckpt_10000_arrays_identical=bool(arrays))
print(json.dumps(out, indent=1))
assert all(v['metrics_identical'] and v['ckpt_10000_arrays_identical'] for v in out.values())
