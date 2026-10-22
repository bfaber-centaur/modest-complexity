"""Reviewer's oracle and split checks for R4 (REVIEW.md §4), written from the
brief and routes/QUESTION.md before reading the lane's code.

Everything is rebuilt from the lane's X matrix alone, in plain Python where it
matters: parity, s, contexts, the split, the exceptions. The lane module is used
only as a black box that hands over its arrays, and its `metrics` is compared
with a separate calculation on random margins.

uv run --locked python research/regimes/review/oracle_routes.py LANE_COMMIT
(LANE_COMMIT is checked out read-only into a temp dir; nothing in the lane's
directory is touched.)
"""
from itertools import product
import sys
import numpy as np

from lanes import check, lane_module, FAIL

COND_K = dict(rho1=0, rho31of32=1, rho15of16=2, rho7of8=4, rho3of4=8, rho1of2=16)


def decode(row):
    b = [int(v > 0) for v in row]                 # +1 is bit 1
    rule, s, ctx = tuple(b[0:3]), b[3], tuple(b[4:10])
    return rule, s, ctx, (rule[0] + rule[1] + rule[2]) % 2


def main(commit):
    T = lane_module(commit, 'routes')
    d = T.domain(); X = d['X']
    check('domain is all 1024 distinct ±1 rows of length 10',
          X.shape == (1024, 10) and set(np.unique(X)) == {-1., 1.} and len({tuple(r) for r in X}) == 1024)
    dec = [decode(r) for r in X]
    check('lane labels equal parity of positions 0-2 (plain Python)', all(int(d['y'][i]) == dec[i][3] for i in range(1024)))
    check('lane s equals position 3', all(int(d['s'][i]) == dec[i][1] for i in range(1024)))
    # a and c must be bijections onto rule patterns and contexts.
    a_map = {}; c_map = {}
    for i, (rule, s, ctx, y) in enumerate(dec):
        a_map.setdefault(int(d['a'][i]), set()).add(rule); c_map.setdefault(int(d['c'][i]), set()).add(ctx)
    check('a and c are bijections onto the 8 rule patterns and 64 contexts',
          len(a_map) == 8 and len(c_map) == 64 and all(len(v) == 1 for v in (*a_map.values(), *c_map.values()))
          and len({next(iter(v)) for v in c_map.values()}) == 64)
    check('index(a, c, s) addresses the matching row',
          all(int(T.index(d['a'][i], d['c'][i], d['s'][i])) == i for i in range(1024)))

    ev = T.evaluation_sets()
    held = np.asarray(ev['held'], bool)
    train_ctx = set()
    for cond in COND_K:
        ts = T.training_set(cond)
        train_ctx |= {dec[i][2] for i in ts['idx']}
    check('256 training bases come from exactly 32 contexts × all 8 rule patterns',
          len(train_ctx) == 32 and all(len({(dec[i][0], dec[i][2]) for i in T.training_set(c)['idx']}) == 256 for c in COND_K)
          and {(r, c) for r in product((0, 1), repeat=3) for c in train_ctx} == {(dec[i][0], dec[i][2]) for i in T.training_set('rho1')['idx']})
    held_ctx = {dec[i][2] for i in range(1024) if held[i]}
    check('held-out = every input whose context is not a training context (512 inputs)',
          held.sum() == 512 and all(held[i] == (dec[i][2] not in train_ctx) for i in range(1024)) and not (held_ctx & train_ctx))
    train_bases_any = {(dec[i][0], dec[i][2]) for c in (*COND_K, 'shortcut_only') for i in T.training_set(c)['idx']}
    check('no base crosses the split under any condition or s value',
          not (train_bases_any & {(dec[i][0], dec[i][2]) for i in range(1024) if held[i]}))
    closed_s = all(held[_flip_row(X, i, 3)] for i in range(1024) if held[i])
    closed_r = all(held[_flip_row(X, i, j)] for i in range(1024) if held[i] for j in range(3))
    check('held-out set closed under flipping s and under flipping any rule bit', closed_s and closed_r)
    check('reliable / reversed environments are s = y / s != y on held-out bases, 256 each',
          np.array_equal(ev['reliable'], held & (d['s'] == d['y'])) and np.array_equal(ev['reversed'], held & (d['s'] != d['y']))
          and ev['reliable'].sum() == ev['reversed'].sum() == 256)

    exc_sets = {}
    for cond, k in COND_K.items():
        ts = T.training_set(cond); idx = np.asarray(ts['idx']); tgt = np.asarray(ts['target']); ex = np.asarray(ts['exception'], bool)
        ys = np.array([dec[i][3] for i in idx]); ss = np.array([dec[i][1] for i in idx])
        ok = (np.array_equal(tgt, ys) and np.array_equal(ex, ss != ys) and abs((ss == ys).mean() - (1 - k/32)) < 1e-12
              and ys.sum() == 128 and ss.sum() == 128
              and all(sum(ex[j] for j in range(256) if dec[idx[j]][0] == r) == k for r in product((0, 1), repeat=3)))
        check(f'{cond}: target = parity, realized rho = 1-{k}/32 exactly, labels and s 128/128, {k} exceptions per rule pattern', ok)
        exc_sets[k] = {(dec[i][0], dec[i][2]) for i, e in zip(idx, ex) if e}
        conf = {(dec[i][0], dec[i][2]): dec[i][1] for i in idx}
        exc_sets.setdefault('s', {})[k] = conf
    ks = sorted(COND_K.values())
    check('exceptions nested across rho', all(exc_sets[a] <= exc_sets[b] for a, b in zip(ks, ks[1:])))
    base_s = exc_sets['s']
    check('conditions differ only in s on exception bases',
          all(base_s[k][b] == base_s[0][b] for k in ks for b in base_s[0] if b not in exc_sets[k]))
    so = T.training_set('shortcut_only'); idx = np.asarray(so['idx'])
    r16 = T.training_set('rho1of2')
    ss = np.array([dec[i][1] for i in idx]); ys = np.array([dec[i][3] for i in idx])
    rule_ind = all(abs(ss[np.array([dec[i][0][j] == 1 for i in idx])].mean() - .5) < 1e-12 for j in range(3))
    check('shortcut_only: rho1of2 inputs, target = s, s independent of y and of each rule bit',
          np.array_equal(idx, r16['idx']) and np.array_equal(so['target'], ss) and abs(np.mean(ss == ys) - .5) < 1e-12 and rule_ind)
    # Irrelevant bits exactly independent of y on training (each context has all 8 patterns).
    trn = np.asarray(T.training_set('rho1')['idx'])
    indep = all(abs(np.mean([dec[i][3] for i in trn if dec[i][2][j] == 1]) - .5) < 1e-12 for j in range(6))
    check('each irrelevant bit is exactly independent of y on the training set', indep)
    # Is exception status predictable from context alone? (it should vary within context)
    by_ctx = {}
    for b in exc_sets[16]:
        by_ctx[b[1]] = by_ctx.get(b[1], 0) + 1
    print(f'INFO rho1of2: exceptions per training context range {min(by_ctx.get(c, 0) for c in train_ctx)}-'
          f'{max(by_ctx.get(c, 0) for c in train_ctx)} of 8 (all-or-none would make exception status a context feature)')

    # Metrics: separate calculation of every metric the lane logs, on random margins.
    r = np.random.default_rng(55400)
    for cond in ('rho7of8', 'shortcut_only'):
        ts = T.training_set(cond)
        margin = r.normal(0, 1, 1024)
        lane = T.metrics(margin, ts, ev, d['y'])
        mine = my_metrics(margin, ts, held, X, dec, cond)
        common = sorted(set(lane) & set(mine))
        bad = [k for k in common if lane[k] is not None and abs(lane[k] - mine[k]) > 1e-12]
        check(f'{cond}: lane metrics equal a separate calculation on random margins ({len(common)} metrics)',
              not bad and len(common) >= 8, f'mismatch {bad}' if bad else f'unmatched: lane {sorted(set(lane) - set(mine))}, mine {sorted(set(mine) - set(lane))}')


def _flip_row(X, i, j):
    row = X[i].copy(); row[j] = -row[j]
    return int(np.flatnonzero((X == row).all(1))[0])


def my_metrics(margin, ts, held, X, dec, cond):
    pred = (margin > 0).astype(int)
    idx = np.asarray(ts['idx']); tgt = np.asarray(ts['target']); ex = np.asarray(ts['exception'], bool)
    y = np.array([q[3] for q in dec]); s = np.array([q[1] for q in dec])
    out = dict(train_acc=np.mean(pred[idx] == tgt))
    out['train_acc_exceptions'] = np.mean(pred[idx][ex] == tgt[ex]) if ex.any() else None
    out['train_acc_conforming'] = np.mean(pred[idx][~ex] == tgt[~ex])
    rel = held & (s == y); rev = held & (s != y)
    out['acc_reliable'] = np.mean(pred[rel] == y[rel]); out['acc_reversed'] = np.mean(pred[rev] == y[rev])
    out['acc_independent'] = np.mean(pred[held] == y[held])
    row_of = {tuple(r): i for i, r in enumerate(X)}

    def pairs(j):
        out_ = []
        for i in np.flatnonzero(held):
            if X[i, j] < 0:
                row = X[i].copy(); row[j] = 1.; k = row_of[tuple(row)]
                if held[k]:
                    out_.append((i, k))   # (bit j = 0 member, bit j = 1 member)
        return out_
    sp = pairs(3)
    out['flip_s'] = np.mean([pred[a] != pred[b] for a, b in sp])
    out['delta_s'] = np.mean([margin[b] - margin[a] for a, b in sp])
    out['absdelta_s'] = np.mean([abs(margin[b] - margin[a]) for a, b in sp])
    rp = [pairs(j) for j in range(3)]
    out['flip_rule'] = np.mean([np.mean([pred[a] != pred[b] for a, b in p]) for p in rp])
    # signed: margin of the y = 1 member minus the y = 0 member
    out['delta_rule'] = np.mean([np.mean([(margin[b] - margin[a]) if y[b] == 1 else (margin[a] - margin[b]) for a, b in p]) for p in rp])
    out['absdelta_rule'] = np.mean([np.mean([abs(margin[b] - margin[a]) for a, b in p]) for p in rp])
    ip = [pairs(j) for j in range(4, 10)]
    out['flip_irr'] = np.mean([np.mean([pred[a] != pred[b] for a, b in p]) for p in ip])
    out['absdelta_irr'] = np.mean([np.mean([abs(margin[b] - margin[a]) for a, b in p]) for p in ip])
    return {k: (None if v is None else float(v)) for k, v in out.items()}


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)
