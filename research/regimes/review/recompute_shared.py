"""Recompute R2's frozen-readout numbers from published pretrained.npz and run
the symmetry counterexample (QUESTION.md, Revisions; REVIEW.md §2).

The readout is rebuilt here: a fresh copy of the init 'new' head, Adam lr 3e-3,
2,000 full-batch updates on unstandardized last-layer features of D_n. Its y2
result must equal the lane's logged readout. The same readout is then fitted for
(x1^x3)&x4 and (x2^x3)&x4, which a y1 network has no reason to treat differently.

uv run --locked python research/regimes/review/recompute_shared.py LANE_COMMIT
"""
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np

from lanes import check, FAIL, ROOT
sys.path.insert(0, str(ROOT/'research/regimes/common'))
import nets  # noqa: E402

I = np.arange(1024)
x = lambda k: (I >> (k-1)) & 1
X = np.stack([x(k) for k in range(1, 11)], 1).astype(float)
TARGETS = {'y2 = (x1^x2)&x4': (x(1) ^ x(2)) & x(4), "y2' = (x1^x3)&x4": (x(1) ^ x(3)) & x(4),
           "y2'' = (x2^x3)&x4": (x(2) ^ x(3)) & x(4)}
NS = (32, 64, 128, 512)


def published(commit, run):
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'archive', commit, f'research/regimes/shared/runs/{run}'])
    t = tarfile.open(fileobj=io.BytesIO(raw))
    return {Path(m.name).name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}


def readout(H, y, W, b, idx, test1, updates=2000, lr=3e-3):
    q = {'W_new': W.copy(), 'b_new': b.copy()}; st = nets.adam_state(q)
    for _ in range(updates):
        nets.adam_step(q, nets.loss_grad(q, H[idx], {'new': y[idx]})[2], st, lr)
    z = H @ q['W_new'] + q['b_new']
    return float(np.mean(z[test1].argmax(1) == y[test1]))


def split(seed):
    o = np.random.default_rng(seed).permutation(1024)
    return o[:512], o[512:]


def main(commit):
    worst = 0.
    for r in range(3):
        run = f'r1-share-rel-s5220{r}-a1'; F = published(commit, run)
        cfg = json.loads(F['config.json']); pool, test = split(cfg['data_seed'])
        test1 = test[x(4)[test] == 1]
        with np.load(io.BytesIO(F['pretrained.npz'])) as z:
            p = {k[2:]: z[k] for k in z.files if k.startswith('p.')}
        p0 = nets.init(10, cfg['hidden'], {'pre': 2, 'new': 2}, cfg['init_seed'])
        check(f'{run}: new head in pretrained.npz equals its init (phase 1 never trains it)',
              np.array_equal(p['W_new'], p0['W_new']) and np.array_equal(p['b_new'], p0['b_new']))
        logged = {row['n']: row['new_test_acc_x4is1'] for row in map(json.loads, gzip.decompress(F['metrics.jsonl.gz']).splitlines())
                  if row.get('tag') == 'readout' and row['step'] == 2000}
        out = {}
        for tag, net in (('y1-pretrained', p), ('untrained', p0)):
            H = nets.forward(net, X)[0][-1]
            for name, y in TARGETS.items():
                out[tag, name] = [readout(H, y, p['W_new'], p['b_new'], pool[:n], test1) for n in NS]
        mine = out['y1-pretrained', 'y2 = (x1^x2)&x4']
        worst = max(worst, max(abs(a - logged[n]) for a, n in zip(mine, NS)))
        for (tag, name), accs in out.items():
            print(f'INFO {run} {tag:13s} {name:18s} test x4=1 acc at n={NS}: {[round(a, 3) for a in accs]}')
    check('the y2 readout rebuilt here equals the lane-logged readout at n = 32, 64, 128, 512 on every replicate',
          worst < 1e-12, f'max |diff| {worst:.1e}')


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
