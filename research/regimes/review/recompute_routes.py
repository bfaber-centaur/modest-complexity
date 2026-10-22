"""Recompute R4's M2 numbers from published artifacts (REVIEW.md §4).

For every published run: verify MANIFEST.sha256, read final.npz with plain
np.load, run a forward pass written here, score it with oracle_routes'
separate metric code, and compare with the run's last metrics row and the
routes_at_T.csv row. Then re-derive README's trajectory claims from metrics.jsonl.

uv run --locked python research/regimes/review/recompute_routes.py LANE_COMMIT
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np

from lanes import check, lane_module, FAIL, ROOT
from oracle_routes import decode, my_metrics

KEYS = ['train_acc', 'train_acc_exceptions', 'acc_reliable', 'acc_reversed', 'acc_independent',
        'flip_s', 'flip_rule', 'flip_irr', 'delta_s', 'delta_rule', 'absdelta_s', 'absdelta_rule', 'absdelta_irr']


def published(commit):
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'archive', commit, 'research/regimes/routes/runs',
                                   'research/regimes/routes/figures/routes_at_T.csv'])
    t = tarfile.open(fileobj=io.BytesIO(raw))
    return {Path(m.name).relative_to('research/regimes/routes').as_posix(): t.extractfile(m).read()
            for m in t.getmembers() if m.isfile()}


def main(commit):
    F = published(commit)
    man = [l.split() for l in F['runs/MANIFEST.sha256'].decode().splitlines() if l.strip()]
    check(f'MANIFEST.sha256: all {len(man)} entries match the published bytes',
          all(hashlib.sha256(F['runs/' + p]).hexdigest() == h for h, p in man))
    T = lane_module(commit, 'routes'); d = T.domain(); X = d['X']; dec = [decode(r) for r in X]
    held = np.array([q[2] not in {dec[i][2] for i in T.training_set('rho1')['idx']} for q in dec])
    at_T = {r['run_id']: r for r in csv.DictReader(io.StringIO(F['figures/routes_at_T.csv'].decode()))}
    worst, n = 0., 0
    rows = {}
    for path in sorted(p for p in F if p.endswith('final.npz')):
        rid = path.split('/')[1]
        cfg = json.loads(F[f'runs/{rid}/config.json']); cond = cfg['condition']
        z = np.load(io.BytesIO(F[path]))
        W1, b1, Wy, by = (z[k] for k in ('p.W1', 'p.b1', 'p.W_y', 'p.b_y'))
        logit = np.maximum(X @ W1 + b1, 0) @ Wy + by
        mine = my_metrics(logit[:, 1] - logit[:, 0], T.training_set(cond), held, X, dec, cond)
        traj = [json.loads(l) for l in F[f'runs/{rid}/metrics.jsonl'].decode().splitlines()]
        last = traj[-1]; rows[rid] = (cond, traj)
        assert last['step'] == cfg['updates']
        for k in KEYS:
            if last.get(k) is None:
                continue
            for other in (last[k], float(at_T[rid][k]) if rid in at_T else last[k]):
                worst = max(worst, abs(mine[k] - other) / max(1., abs(other)))
        n += 1
    check(f'{n} runs: final.npz rescored with independent code equals the last metrics row and routes_at_T.csv',
          worst < 1e-9, f'worst relative diff {worst:.1e}')
    main_runs = {r: v for r, v in rows.items() if '-s5420' in r}
    check('21 initial-comparison runs at T = 20000 (7 conditions x seeds 54200-54202)', len(main_runs) == 21)
    first = lambda traj, f: next((r['step'] for r in traj if f(r)), None)
    for cond in ('rho31of32', 'rho15of16', 'rho7of8', 'rho3of4'):
        tr = [v[1] for r, v in main_runs.items() if v[0] == cond]
        sc = [first(t, lambda r: r['flip_s'] >= .95) for t in tr]
        fit = [first(t, lambda r: r['train_acc'] == 1.) for t in tr]
        rev_fit = [next(r['acc_reversed'] for r in t if r['step'] == f) for t, f in zip(tr, fit)]
        cross = [first(t, lambda r: r['acc_reversed'] > .5) for t in tr]
        loss_cross = [next((r['train_loss'] for r in t if r['step'] == c), None) for t, c in zip(tr, cross)]
        print(f'INFO {cond}: flip_s>=0.95 first at {sc}; train fit at {fit} (reversed there {[round(x, 3) for x in rev_fit]}); '
              f'reversed>0.5 first at {cross} (train loss {[None if x is None else f"{x:.1e}" for x in loss_cross]}); '
              f'final delta_rule {[round(t[-1]["delta_rule"], 1) for t in tr]} delta_s {[round(t[-1]["delta_s"], 1) for t in tr]}')
        # Is the rule's contribution still rising over the last decade of updates?
        late = [(t[-1]['delta_rule'] - next(r for r in t if r['step'] >= 2000)['delta_rule']) for t in tr]
        print(f'INFO {cond}: delta_rule change from step ~2000 to 20000: {[round(x, 1) for x in late]}')
    rows_all = [r for v in main_runs.values() for r in v[1] if r['acc_reliable'] == 1.]
    check(f'identity: flip_s = 1 - reversed and flip_rule = reversed whenever reliable = 1 ({len(rows_all)} logged rows)',
          all(abs(r['flip_s'] - (1 - r['acc_reversed'])) < 1e-12 and abs(r['flip_rule'] - r['acc_reversed']) < 1e-12 for r in rows_all))
    # Counterexample (QUESTION.md, Revisions): at rho = 15/16, does reversed-environment
    # correctness vary by held-out context or by rule pattern?
    rev = np.array([held[i] and dec[i][1] != dec[i][3] for i in range(len(X))])
    ctxs = sorted({dec[i][2] for i in range(len(X)) if rev[i]}); pats = sorted({dec[i][0] for i in range(len(X))})
    for rid in sorted(r for r, v in main_runs.items() if v[0] == 'rho15of16'):
        z = np.load(io.BytesIO(F[f'runs/{rid}/final.npz']))
        logit = np.maximum(X @ z['p.W1'] + z['p.b1'], 0) @ z['p.W_y'] + z['p.b_y']
        ok = {(dec[i][2], dec[i][0]): int((logit[i, 1] > logit[i, 0]) == dec[i][3]) for i in np.flatnonzero(rev)}
        R = np.array([[ok[(c, a)] for a in pats] for c in ctxs], float)
        m = R.mean(); tot = ((R - m)**2).sum()
        sc = 8*((R.mean(1) - m)**2).sum()/tot; sp = 32*((R.mean(0) - m)**2).sum()/tot
        print(f'INFO {rid}: reversed acc {m:.3f}; variance share context {sc:.2f}, rule pattern {sp:.2f}, '
              f'interaction {1-sc-sp:.2f} (shuffled null about 0.12 / 0.03); contexts all-correct '
              f'{int((R.mean(1) == 1).sum())}, all-wrong {int((R.mean(1) == 0).sum())} of {len(ctxs)}; '
              f'per-pattern acc {np.round(R.mean(0), 2).tolist()}')


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
