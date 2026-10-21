#!/usr/bin/env python3
"""Checks of the restricted-rank definition (research/audit/heldout_rank.py; DISPATCH
item 2). Independent code only (forward pass from research/audit/independent.py).

definition.jsonl, one row per published run: split disjoint and covering, train
  transpose-closed and equal to the frozen split, mask from training pairs only, a+b
  absent for every held-out pair of an addition run, position-specific against
  position-ignored mask, absent-set sizes, chance for the top fraction, and at the
  final checkpoint ties and the agreement of the two orders (a, b) and (b, a).
nulls.jsonl, final checkpoints with the mask held fixed: (i) a random other absent
  label as the probe, ranked with a+b left out (1,000 draws); (ii) step 0 and the
  no-decay and shuffled-label models of the same seed, scored under the addition
  run's own mask, pairs and probe.
shares.jsonl, width 128, 30% against 50% share, same init seeds, every checkpoint: the
  483 pairs that neither share trains on, ranked among the labels absent in the 50%
  training set (absent in the 30% set too), next to the unrestricted rank of a+b among
  all 30 other labels, accuracy and cross-entropy on the same pairs.

uv run --locked python research/e2/audit/rank_checks.py OUT_DIR
"""
import json
from pathlib import Path
import re
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root/'research/audit'))
import independent as ind  # noqa: E402

P = 31
RUNS = sorted(f.parent for d in ('coordinator', 'size-time') for f in (root/'research'/d/'runs').glob('*/data.npz'))
FROZEN = {299: 'research/data/e1-split.npz', 478: 'research/data/e1f2-split-tf50.npz'}


def load(run):
    with np.load(run/'data.npz') as d:
        return {k: d[k] for k in d.files} | dict(held=np.concatenate([d['cal'], d['audit']]))


def steps(run):
    return sorted(int(f.stem.split('_')[1]) for f in run.glob('checkpoint_*.npz'))


def logits(run, step, d):
    with np.load(run/f'checkpoint_{step:06d}.npz') as z:
        return ind.forward((z['U'], z['V'], z['W']), *ind.one_hot_inputs(d['a'], d['b'], P))[0]


def absent(d, pairs, positional=False):
    """absent[i, c]: label c never a training target together with either operand of pair i."""
    first, second = np.zeros((P, P), bool), np.zeros((P, P), bool)
    tr = d['train']
    first[d['a'][tr], d['y'][tr]] = True; second[d['b'][tr], d['y'][tr]] = True
    if not positional:  # the definition: operand position ignored
        first = second = first | second
    return ~(first[d['a'][pairs]] | second[d['b'][pairs]])


def rank(z, probe, comp):
    """Normalized rank of probe among the labels in comp other than itself (strict >)."""
    rows = np.arange(len(probe))
    other = comp.copy(); other[rows, probe] = False
    t = z[rows, probe][:, None]
    above, ties, n = ((z > t) & other).sum(1), ((z == t) & other).sum(1), other.sum(1)
    return above/n, above == 0, n, ties


def definition(run):
    d = load(run); held = d['held']
    idx = np.full((P, P), -1); idx[d['a'], d['b']] = np.arange(P*P)
    train = set(d['train'].tolist())
    with np.load(root/FROZEN[len(d['train'])]) as f:
        frozen = all(set(f[k].tolist()) == set(d[k].tolist()) for k in ('train', 'cal', 'audit'))
    ab = absent(d, held); s = d['yadd'][held]; rows = np.arange(len(held))
    usable = ab[rows, s]
    ab_pos = absent(d, held, positional=True)
    n_abs = ab[usable].sum(1)
    final = steps(run)[-1]; z = logits(run, final, d)[held]
    r, top, n, ties = rank(z[usable], s[usable], ab[usable])
    above_ge = ((z[usable] >= z[usable][np.arange(usable.sum()), s[usable]][:, None]) & ab[usable]).sum(1) - 1
    # rank of (b, a) next to rank of (a, b), off-diagonal pairs whose transpose is also usable
    pos = {int(i): j for j, i in enumerate(held[usable])}
    tp = [(pos[i], pos[int(idx[d['b'][i], d['a'][i]])]) for i in held[usable].tolist()
          if d['a'][i] < d['b'][i] and int(idx[d['b'][i], d['a'][i]]) in pos]
    x, y = r[[i for i, _ in tp]], r[[j for _, j in tp]]
    return dict(run=run.name, train_pairs=len(train), targets='addition' if np.array_equal(d['y'], d['yadd']) else 'other',
                split_disjoint_and_covering=sorted(np.concatenate([d['train'], held]).tolist()) == list(range(P*P)),
                train_transpose_closed=all(int(idx[d['b'][i], d['a'][i]]) in train for i in train),
                equals_frozen_split=frozen, heldout_pairs=len(held), pairs_with_sum_absent=int(usable.sum()),
                positional_mask_entries_differing=int((ab != ab_pos).sum()),
                positional_mask_usability_differing=int((usable != ab_pos[rows, s]).sum()),
                absent_min_median_max=[int(n_abs.min()), float(np.median(n_abs)), int(n_abs.max())],
                singletons=int((n_abs == 1).sum()), chance_top=round(float(np.mean(1/n_abs)), 4),
                final_step=final, ties=int(ties.sum()), rank_mean=round(float(r.mean()), 4),
                rank_mean_ties_counted_above=round(float((above_ge/n).mean()), 4), top_frac=round(float(top.mean()), 4),
                transpose_pairs=len(tp), transpose_rank_corr=round(float(np.corrcoef(x, y)[0, 1]), 3) if x.std() and y.std() else None,  # None: constant
                transpose_rank_mean_abs_diff=round(float(np.abs(x - y).mean()), 4))


def random_probe_null(z, s, ab, rng, draws=1000):
    """Mean rank of a uniformly drawn absent label other than a+b, a+b left out of the comparison."""
    comp = ab.copy(); comp[np.arange(len(s)), s] = False
    keep = comp.sum(1) >= 2; comp = comp[keep]; z = z[keep]
    means = [rank(z, np.where(comp, rng.random(comp.shape), -1).argmax(1), comp)[0].mean() for _ in range(draws)]
    return dict(pairs=int(keep.sum()), mean=round(float(np.mean(means)), 4),
                p025_p975=[round(float(q), 4) for q in np.quantile(means, [.025, .975])])


def nulls():
    rng = np.random.default_rng(35100); out = []
    by = {r.name: r for r in RUNS}
    for seed in (30000, 30001, 30002):
        add = next(by[n] for n in by if re.fullmatch(rf'e1-\w+-add-w128-i{seed}-a1', n))
        d = load(add); held = d['held']; s = d['yadd'][held]
        ab = absent(d, held); u = ab[np.arange(len(held)), s]
        z = logits(add, 50000, d)[held]
        row = dict(mask_run=add.name, pairs=int(u.sum()), trained_rank=round(float(rank(z[u], s[u], ab[u])[0].mean()), 4),
                   random_absent_probe=random_probe_null(z[u], s[u], ab[u], rng))
        if 0 in steps(add):
            row['step0_rank'] = round(float(rank(logits(add, 0, d)[held][u], s[u], ab[u])[0].mean()), 4)
        for ctl in ('add0', 'shuf'):
            c = by[f'e1-st-{ctl}-w128-i{seed}-a1']; dc = load(c)
            same = all(np.array_equal(dc[k], d[k]) for k in ('a', 'b', 'yadd', 'train', 'cal', 'audit'))
            zc = logits(c, 50000, dc)[held]
            row[ctl] = dict(same_pairs_and_split=same, same_mask=bool(np.array_equal(absent(dc, held), ab)),
                            rank_under_addition_mask=round(float(rank(zc[u], s[u], ab[u])[0].mean()), 4),
                            top_under_addition_mask=round(float(rank(zc[u], s[u], ab[u])[1].mean()), 4))
        out.append(row)
    return out


def shares():
    by = {r.name: r for r in RUNS}
    d50 = load(by['e1f2-co-add-w128-i30000-tf50-a1']); pairs = d50['held']; s = d50['yadd'][pairs]
    ab50 = absent(d50, pairs); rows = np.arange(len(pairs)); every = np.ones_like(ab50)
    out = []
    for name in sorted(n for n in by if re.fullmatch(r'e1(f2)?-\w+-add-w128-i3000\d(-tf50)?-a1', n)):
        run = by[name]; d = load(run)
        assert all(np.array_equal(d[k], d50[k]) for k in ('a', 'b', 'yadd'))
        assert set(d['train'].tolist()) <= set(d50['train'].tolist()) and not set(pairs.tolist()) & set(d['train'].tolist())
        own = absent(d, pairs)
        assert not (ab50 & ~own).any() and ab50[rows, s].all()  # 50% absent labels are absent for this run too
        for step in steps(run):
            z = logits(run, step, d)[pairs]
            r, top, _, _ = rank(z, s, ab50); ro, to, _, _ = rank(z, s, own); ru, _, _, _ = rank(z, s, every)
            ev = ind.evaluate(z, s)
            out.append(dict(run=name, share='50%' if 'tf50' in name else '30%', step=step, pairs=len(pairs),
                            rank_50mask=round(float(r.mean()), 4), top_50mask=round(float(top.mean()), 4),
                            rank_own_mask=round(float(ro.mean()), 4), top_own_mask=round(float(to.mean()), 4),
                            rank_all_labels=round(float(ru.mean()), 4), acc=round(ev['acc'], 4),
                            cross_entropy=round(ev['cross_entropy'], 4)))
    chance = dict(chance_top_50mask=round(float(np.mean(1/ab50.sum(1))), 4),
                  absent_min_median_max=[int(ab50.sum(1).min()), float(np.median(ab50.sum(1))), int(ab50.sum(1).max())])
    return out, chance


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    write = lambda name, rows: (out/name).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write('definition.jsonl', map(definition, RUNS))
    write('nulls.jsonl', nulls())
    rows, chance = shares()
    write('shares.jsonl', [chance] + rows)
