#!/usr/bin/env python3
"""Per-group optimizer and decay diagnostics from saved checkpoints (QUESTION.md, secondary).

For each checkpoint (weights p, Adam moments m, v after t updates) and each parameter group
(start = Ws, bs; transition = A, B, bh; readout = V, bv; an untied group per copy k):
- param_norm      ||p||
- adam_step_norm  lr ||m_hat / (sqrt(v_hat) + eps)||, the Adam part of update t
- decay_step_norm lr wd ||p||, the decay part (nets.adam_step decays before the Adam step)
- cos_adam_decay  cosine between the two parts; -1 means the gradient step pushes straight
                  against decay
- frac_below_eps  fraction of coordinates with sqrt(v_hat) < eps
and, for untied groups, the cosine between every pair of copies (all start at 1).

Writes out/groups.csv; with --variant wd0 (the follow-up runs, decay step 0) out/groups_wd0.csv.
Step 0 has no Adam step; its step norms are blank.
uv run --locked python research/composition-c2/sharing/share_groups.py [--variant wd0]
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import share_runs as S  # noqa: E402
import c2rnn  # noqa: E402
import nets  # noqa: E402

LR, WD, B1, B2, EPS = 1e-3, 0.1, .9, .999, 1e-8
STEPS = (0, 1000, 2000, 4000, 7000, 10000)
GROUPS = dict(start=('Ws', 'bs'), transition=c2rnn.TRANSITION, readout=c2rnn.READOUT)


def flat(d, names):
    return np.concatenate([d[k].ravel() for k in names])


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))) if np.any(a) and np.any(b) else float('nan')


def rows_for(path, cell, seed, step, wd=WD):
    p, st, _ = nets.load(path)
    t = st['t']; assert t == step, (t, step)
    out = []
    copies = {}
    for group, names in GROUPS.items():
        ks = ('',) if group == 'start' or names[0] in p else ('1', '2', '3')
        for k in ks:
            keys = [n + k for n in names]
            w = flat(p, keys)
            row = dict(cell=cell, seed=seed, step=step, group=group, copy=k or 'shared', param_norm=float(np.linalg.norm(w)),
                       decay_step_norm=LR*wd*float(np.linalg.norm(w)))
            if t > 0:
                mh, vh = flat(st['m'], keys)/(1-B1**t), flat(st['v'], keys)/(1-B2**t)
                a = mh/(np.sqrt(vh)+EPS)
                row.update(adam_step_norm=LR*float(np.linalg.norm(a)), cos_adam_decay=cos(a, w),
                           frac_below_eps=float(np.mean(np.sqrt(vh) < EPS)))
            out.append(row)
            copies[(group, k)] = w
    for group in ('transition', 'readout'):
        if (group, '1') in copies:
            for a, b in (('1', '2'), ('1', '3'), ('2', '3')):
                out.append(dict(cell=cell, seed=seed, step=step, group=group, copy=f'{a}~{b}',
                                copy_cosine=cos(copies[(group, a)], copies[(group, b)])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--variant', choices=('main', 'wd0'), default='main')
    args = ap.parse_args()
    wd = WD if args.variant == 'main' else 0.
    rows = []
    for cell in S.CELLS:
        for seed in S.SEEDS:
            for step in STEPS:
                try:
                    path = S.checkpoint(cell, seed, step, args.variant)
                except FileNotFoundError:
                    if args.variant != 'main':
                        continue
                    # TT/TU originals kept only ckpt-0/1000/10000; their replays carry the rest.
                    try:
                        path = S.find(S.replay_id(cell, seed), f'ckpt-{step}.npz')
                    except FileNotFoundError:
                        continue
                rows += [dict(r, source=path.parent.name) for r in rows_for(path, cell, seed, step, wd)]
    fields = ['cell', 'seed', 'step', 'source', 'group', 'copy', 'param_norm', 'adam_step_norm', 'decay_step_norm',
              'cos_adam_decay', 'frac_below_eps', 'copy_cosine']
    out = Path(__file__).resolve().parent/'out'; out.mkdir(exist_ok=True)
    with (out/('groups.csv' if args.variant == 'main' else f'groups_{args.variant}.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(len(rows), 'rows')


if __name__ == '__main__':
    main()
