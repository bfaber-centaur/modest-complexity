"""R4 M4 checks (REVIEW.md §4) and the optimizer-stall counterexample
(QUESTION.md, Revisions). Analysis only, from published files at a lane commit.

1. Registry: interrupted a1 runs, manual events, a2 reruns, plan order.
2. step20000.npz of every follow-up run equals the initial run's final.npz
   (weights, both moments, t); a1 partial metrics equal the a2 reruns line for line.
3. final.npz at 200k rescored with independent code; follow-up table numbers.
4. Counterexample: decompose the last Adam step at 20k and 200k, and the share of
   the exact-arithmetic gradient weight carried by float64-saturated rows.

uv run --locked python research/regimes/review/check_routes_m4.py LANE_COMMIT
"""
import hashlib
import io
import json
import subprocess
import sys
import numpy as np

from lanes import check, lane_module, FAIL, ROOT
from oracle_routes import decode, my_metrics
from recompute_routes import KEYS, published

CONDS, SEEDS = ('rho1', 'rho31of32', 'rho15of16'), (54200, 54201, 54202)


def long_id(c, s):
    return f'r1-route-long-{c}-s{s}-' + ('a2' if s == 54202 and c != 'rho15of16' else 'a1')


def main(commit):
    F = published(commit)
    when = lambda c: subprocess.check_output(['git', '-C', str(ROOT), 'show', '-s', '--format=%cI', c], text=True).strip()
    reg = [json.loads(l) for l in subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', f'{commit}:research/regimes/routes/registry.jsonl']).decode().splitlines()]
    reg = [r for r in reg if '-long-' in r['run_id']]
    ids = sorted({r['run_id'] for r in reg})
    ev = {i: [r for r in reg if r['run_id'] == i] for i in ids}
    check(f'registry: {len(ids)} follow-up IDs, each with one running and one terminal event, no ID reused',
          all([r['status'] for r in e] in (['running', 'complete'], ['running', 'failed']) for e in ev.values()))
    manual = [i for i, e in ev.items() if e[-1].get('logged_by') == 'manual']
    print('INFO hand-written terminal events:', {i: (ev[i][-1]['status'], ev[i][-1]['returncode'], ev[i][-1]['reason'][:60] + '...') for i in manual})
    dt = lambda e: (np.datetime64(e[-1]['time'][:19]) - np.datetime64(e[0]['time'][:19])).astype(int)
    check('the manual events record the wall time between launch and the last metrics write',
          all(abs(dt(ev[i]) - ev[i][-1]['wall_seconds_including_startup']) <= 1 for i in manual))
    first = min(r['time'] for r in reg if r['status'] == 'running')
    a2 = min(r['time'] for r in reg if r['status'] == 'running' and r['run_id'].endswith('-a2'))
    check(f'order: follow-up predictions and plan 5710c35 ({when("5710c35")}) < first launch ({first}); '
          f'a2 plan f79f170 ({when("f79f170")}) < first a2 launch ({a2})', when('5710c35') < first and when('f79f170') < a2)

    man = [l.split(None, 2) for l in F['runs/MANIFEST.sha256'].decode().splitlines() if l.strip()]
    pub = [(h, p) for h, p, *rest in man if not rest]
    files = {p[5:] for p in F if p.startswith('runs/') and not p.endswith('MANIFEST.sha256')}
    check(f'MANIFEST.sha256: {len(pub)} published entries match their bytes, every published run file is listed '
          f'({len(man) - len(pub)} entries marked local-only)',
          all(hashlib.sha256(F['runs/' + p]).hexdigest() == h for h, p in pub) and files <= {p for _, p in pub})

    # 2. Branch-point identity and a1 = a2.
    load = lambda path: dict(np.load(io.BytesIO(F[path])))
    same = []
    for i in ids:
        c, s = i.split('-long-')[1].rsplit('-s', 1)[0], int(i.rsplit('-s', 1)[1][:5])
        a, b = load(f'runs/{i}/step20000.npz'), load(f'runs/r1-route-{c}-s{s}-a1/final.npz')
        keys = [k for k in a if k != 'meta']
        same.append(set(keys) == {k for k in b if k != 'meta'} and all(np.array_equal(a[k], b[k]) for k in keys) and int(a['t']) == 20000)
    check(f'{len(ids)} follow-up runs: step20000.npz equals the initial run final.npz (all p., m., v. arrays and t = 20000)', all(same))
    for c in ('rho1', 'rho31of32'):
        a1 = F[f'runs/r1-route-long-{c}-s54202-a1/metrics.jsonl'].decode().splitlines()
        b2 = F[f'runs/r1-route-long-{c}-s54202-a2/metrics.jsonl'].decode().splitlines()
        missing = [f for f in ('final.npz', 'runtime.json') if f'runs/r1-route-long-{c}-s54202-a1/{f}' in F]
        check(f'{c} s54202: the {len(a1)} a1 metrics lines (to step {json.loads(a1[-1])["step"]}) equal the first {len(a1)} a2 lines; '
              f'a1 publishes no final.npz/runtime.json', a1 == b2[:len(a1)] and not missing)

    # 3. Rescore final.npz at 200k; the follow-up table.
    T = lane_module(commit, 'routes'); d = T.domain(); X = d['X']; dec = [decode(r) for r in X]
    held = np.array([q[2] not in {dec[i][2] for i in T.training_set('rho1')['idx']} for q in dec])
    worst, M = 0., {}
    for c in CONDS:
        for s in SEEDS:
            i = long_id(c, s); z = load(f'runs/{i}/final.npz')
            logit = np.maximum(X @ z['p.W1'] + z['p.b1'], 0) @ z['p.W_y'] + z['p.b_y']
            M[c, s] = [json.loads(l) for l in F[f'runs/{i}/metrics.jsonl'].decode().splitlines()]
            mine = my_metrics(logit[:, 1] - logit[:, 0], T.training_set(c), held, X, dec, c)
            worst = max([worst] + [abs(mine[k] - M[c, s][-1][k]) / max(1., abs(M[c, s][-1][k])) for k in KEYS if M[c, s][-1].get(k) is not None])
    check('9 follow-up final.npz (200k) rescored with independent code equal the last metrics row', worst < 1e-9, f'worst relative diff {worst:.1e}')
    share = lambda r: r['delta_rule'] / (abs(r['delta_s']) + abs(r['delta_rule']))
    at = lambda m, step: next(r for r in m if r['step'] >= step)
    for c in CONDS:
        rows = [(M[c, s], s) for s in SEEDS]
        print(f'INFO {c}: reversed 20k->39k->200k', [(at(m, 20000)['acc_reversed'], at(m, 38858)['acc_reversed'], m[-1]['acc_reversed']) for m, _ in rows])
        print(f'INFO {c}: rule share 20k->39k->200k', [tuple(round(share(at(m, k)), 3) for k in (20000, 38858, 200000)) for m, _ in rows])
        print(f'INFO {c}: displacement/update at steps {at(rows[0][0], 20000)["step"]}, 38858, 200000:',
              [tuple(f'{at(m, k)["mean_update_norm"]:.2g}' for k in (20000, 38858, 200000)) for m, _ in rows])
        print(f'INFO {c}: median sqrt(vhat) at 20177 and 200000; saturated fraction at 200000; min/median margin at 200000',
              [(f'{at(m, 20000)["adam_sqrt_vhat_median"]:.1g}', f'{m[-1]["adam_sqrt_vhat_median"]:.1g}', m[-1]['train_frac_saturated'],
                round(m[-1]['train_margin_min'], 1), round(m[-1]['train_margin_median'], 1)) for m, _ in rows])
        if c == 'rho15of16':
            print('INFO rho15of16: first step with reversed > 0.5', [next((r['step'] for r in m if r['acc_reversed'] > .5), None) for m, _ in rows])

    # 4. Counterexample: what sets the late step size?
    print('INFO last Adam step from the saved state (lr 1e-3, eps 1e-8): |step|; share of |step|^2 on coords with sqrt(vhat) >= eps; '
          '|step| / |(lr/eps) mhat|; |step with eps = 0| / |step|; saturated rows share of sum(1 - p) (exact)')
    for c in CONDS:
        for s in SEEDS:
            i = long_id(c, s); ts = T.training_set(c)
            for tag, path in (('20k', f'runs/{i}/step20000.npz'), ('200k', f'runs/{i}/final.npz')):
                z = load(path); t = int(z['t']); ks = [k[2:] for k in z if k.startswith('p.')]
                mh = np.concatenate([z[f'm.{k}'].ravel() for k in ks]) / (1 - .9**t)
                sv = np.sqrt(np.concatenate([z[f'v.{k}'].ravel() for k in ks]) / (1 - .999**t))
                step = 1e-3 * mh / (sv + 1e-8)
                big = np.sum(step[sv >= 1e-8]**2) / np.sum(step**2)
                gd = np.linalg.norm(1e-3 / 1e-8 * mh); noeps = np.linalg.norm(1e-3 * mh / np.where(sv > 0, sv, np.inf))
                Xt = X[ts['idx']]; tgt = ts['target']
                zt = np.maximum(Xt @ z['p.W1'] + z['p.b1'], 0) @ z['p.W_y'] + z['p.b_y']
                m = (zt[:, 1] - zt[:, 0]) * (2*tgt - 1)
                q = 1 / (1 + np.exp(m))                       # 1 - p_target without rounding to 0
                e = np.exp(zt - zt.max(1, keepdims=True)); sat = (e / e.sum(1, keepdims=True))[np.arange(len(tgt)), tgt] == 1.
                print(f'  {i} {tag:4s} t={t}: |step| {np.linalg.norm(step):.2g}; share sqrt(vhat)>=eps {big:.3g}; '
                      f'step/GD-like {np.linalg.norm(step)/gd:.3f}; eps=0/step {noeps/np.linalg.norm(step):.3g}; '
                      f'saturated {sat.mean():.3f} of rows carrying {q[sat].sum()/q.sum():.1e} of sum(1-p); min margin {m.min():.1f}')


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
