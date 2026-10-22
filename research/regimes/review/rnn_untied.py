"""Run R1's own train.py (read from a lane commit, unedited) for the reviewer's
`rnn-untied` counterexample arm (QUESTION.md, Revisions). MODE=untied replaces
the lane's three RNN functions so that h1, h2 and h3 each get their own readout
V1, V2, V3, all initialized as copies of the lane's V; the transition and
everything else are the lane's. MODE=tied runs the lane unchanged (replay check).
Usage, through run_jobs.py:
  rnn_untied.py LANE_COMMIT MODE -- <train.py arguments>
"""
import hashlib
import json
import sys
import numpy as np

from lanes import lane_module

commit, mode = sys.argv[1:3]
rest = sys.argv[sys.argv.index('--')+1:]
T = lane_module(commit, 'composition', 'train.py')
tied_init, tied_forward, tied_grad = T.rnn_init, T.rnn_forward, T.rnn_loss_grad
STEPS = tuple(enumerate(T.STEPS, 1))


def untied_init(width, seed):
    p = tied_init(width, seed); V, bv = p.pop('V'), p.pop('bv')
    for t, _ in STEPS:
        p[f'V{t}'], p[f'bv{t}'] = V.copy(), bv.copy()
    return p


def untied_forward(p, d):
    H = tied_forward({**p, 'V': p['V3'], 'bv': p['bv3']}, d)[0]  # the lane's transition
    return H, {h: H[t] @ p[f'V{t}'] + p[f'bv{t}'] for t, h in STEPS}


def untied_grad(p, d, targets):
    """The lane's rnn_loss_grad with readout V_t at step t (same operations, same order)."""
    H, logits = untied_forward(p, d)
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dH = [np.zeros_like(h) for h in H]
    total = 0.
    for t, h in STEPS:
        if h not in targets:
            continue
        y = targets[h]; n = len(y)
        z = logits[h] - logits[h].max(1, keepdims=True)
        e = np.exp(z); sm = e / e.sum(1, keepdims=True)
        total += float(np.mean(np.log(e.sum(1)) - z[np.arange(n), y]))
        sm[np.arange(n), y] -= 1; sm /= n
        g[f'V{t}'] += H[t].T @ sm; g[f'bv{t}'] += sm.sum(0); dH[t] += sm @ p[f'V{t}'].T
    eye_o = np.eye(T.task.N_OP)
    for t, slot in zip((3, 2, 1), 'cba'):
        dz = dH[t] * (H[t] > 0)
        g['A'] += H[t-1].T @ dz; g['B'] += eye_o[d[slot]].T @ dz; g['bh'] += dz.sum(0)
        dH[t-1] += dz @ p['A'].T
    dz = dH[0] * (H[0] > 0)
    g['Ws'] = np.eye(T.task.N_SYM)[d['s']].T @ dz; g['bs'] = dz.sum(0)
    return total, g


def checks(seed):
    """At init the untied net is the lane's net; and its gradient matches central differences."""
    P = T.task.draw_ops()[0]; d = T.task.rows(P, T.task.split()[1])
    y = {h: d[h] for _, h in STEPS}
    pt, pu = tied_init(128, seed), untied_init(128, seed)
    (lt, gt), (lu, gu) = tied_grad(pt, d, y), untied_grad(pu, d, y)
    same_init = (lt == lu and all(np.array_equal(gt[k], gu[k]) for k in gt if k not in ('V', 'bv'))
                 and all(np.array_equal(gt[k], gu[f'{k}1'] + gu[f'{k}2'] + gu[f'{k}3']) for k in ('V', 'bv'))
                 and all(np.array_equal(tied_forward(pt, d)[1][h], untied_forward(pu, d)[1][h]) for _, h in STEPS))
    r = np.random.default_rng(55100); q = untied_init(6, 55101)
    q = {k: v + r.normal(0, .3, v.shape) for k, v in q.items()}  # readouts no longer equal
    small = {k: v[:40] for k, v in d.items()}; ys = {h: small[h] for _, h in STEPS}
    g = untied_grad(q, small, ys)[1]; worst = 0.
    for k, v in q.items():
        for i in np.ndindex(v.shape):
            old = v[i]; v[i] = old + 1e-6; up = untied_grad(q, small, ys)[0]
            v[i] = old - 1e-6; down = untied_grad(q, small, ys)[0]; v[i] = old
            worst = max(worst, abs((up - down)/2e-6 - g[k][i]))
    return dict(identical_to_lane_at_init=bool(same_init), finite_difference_worst_abs_error=worst,
                finite_difference_entries=int(sum(v.size for v in q.values())))


if mode == 'untied':
    seed = int(rest[rest.index('--init-seed')+1])
    result = checks(seed); print(json.dumps(result), flush=True)
    assert result['identical_to_lane_at_init'] and result['finite_difference_worst_abs_error'] < 1e-7, result
    T.rnn_init, T.rnn_forward, T.rnn_loss_grad = untied_init, untied_forward, untied_grad
else:
    assert mode == 'tied'; result = None
sys.argv = ['train.py', *rest]
T.main()
out = rest[rest.index('--out')+1]
cfg = json.load(open(f'{out}/config.json'))
cfg['reviewer'] = dict(lane_commit=commit, mode=mode, wrapper_sha256=hashlib.sha256(open(__file__, 'rb').read()).hexdigest(),
                       checks=result, note='seeds reused from the lane runs on purpose, to pair this arm with theirs')
json.dump(cfg, open(f'{out}/config.json', 'w'), indent=1)
