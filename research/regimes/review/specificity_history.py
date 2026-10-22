"""R3 counterexample (QUESTION.md, Revisions; REVIEW.md §3): is A's linear
decodability after B specific to A, or would any 3-bit parity read out as well?

1. Replay check: each reviewer replay's end-of-B snapshot (metrics and both
   readouts) must equal the lane's summary.json snapshot of the same branch.
2. The lane's own readout recipe (history.readout: train-standardized features,
   fresh depth-0 head from seed 53300, Adam lr 1e-2, 2,000 updates, A's 256
   training rows) is refit with only the target changed: C1 = x7^x8^x9 and
   C2 = x0^x1^x7, on the untrained trunk, ckA and every end-of-B trunk.

uv run --locked python research/regimes/review/specificity_history.py LANE_COMMIT RUN_ROOT
"""
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

from lanes import check, lane_module, FAIL, ROOT

commit, root = sys.argv[1], Path(sys.argv[2])
H = lane_module(commit, 'history', 'history.py')
import nets  # noqa: E402  (the lane commit's copy, on sys.path via lane_module)

X = H.domain(); train, test = H.split()
par = lambda *b: X[:, list(b)].sum(1).astype(int) % 2
TARGETS = dict(A=par(0, 1, 2), C1=par(7, 8, 9), C2=par(0, 1, 7))


def lane_summary(run):
    return json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                               f'{commit}:research/regimes/history/runs/{run}/summary.json']))


def readouts(p):
    Hid = nets.forward(p, X)[0][1]
    return {k: H.readout(Hid, y, train, test)['2000']['test_acc'] for k, y in TARGETS.items()}


same, rows = True, []
for s in (53200, 53201, 53202):
    ckA, _, _ = nets.load(io.BytesIO(subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', f'{commit}:research/regimes/history/runs/r1-hist-A-s{s}-a1/ckpt_t2000.npz'])))
    p0 = nets.init(10, (128,), {'A': 2, 'B': 2}, s)
    rows.append((s, 'untrained', readouts(p0))); rows.append((s, 'end of A (ckA)', readouts(ckA)))
    for b in ('rel', 'unrel'):
        for d in (1000, 10000):
            mine = root/f'r1-rev-hist-replay-{b}-d{d}-s{s}-a1'
            snap = next(x for x in json.loads((mine/'summary.json').read_text())['snapshots'] if x['label'] == 'phase_end')
            theirs = next(x for x in lane_summary(f'r1-hist-{b}-d{d}-s{s}-a1')['snapshots']
                          if x['label'] == 'phase_end' and x['phase'] == 1)
            same &= all(snap[k] == theirs[k] for k in ('metrics', 'readout', 'readout_null', 'step', 't'))
            p, _, _ = nets.load(mine/snap['ckpt'])
            rows.append((s, f'end of B{b} d={d}', readouts(p)))
check('every replayed end-of-B snapshot equals the lane snapshot (metrics, readout, null: exact float equality)', same)
print('held-out readout accuracy (lane recipe, 2,000 updates), targets A | C1 = x7^x8^x9 | C2 = x0^x1^x7')
for s, what, r in rows:
    print(f'  seed {s}  {what:22s} A {r["A"]:.3f}   C1 {r["C1"]:.3f}   C2 {r["C2"]:.3f}   A-C1 {r["A"]-r["C1"]:+.3f}')
print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
