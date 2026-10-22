#!/usr/bin/env python3
"""Build and check the e3 nested training-share ladders (one per data seed).

For each data seed: the 30% / 40% / 45% / 50% training sets must nest, every split must
be transpose-closed and the three splits must partition the 961 pairs. The ladder's
evaluation pairs are those outside its 50% training set (that split's cal + audit), so
they are untrained at every share of the ladder. Label-set sizes per evaluation pair:
S = labels seen with an operand in the 30% training set, N = labels never seen with
either operand in the 50% training set, a+b excluded (a+b is asserted absent).

uv run --locked python research/e3/coordinator/ladders.py
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiment import array_sha256, data  # noqa: E402

P = 31
CONTROL_SEED = 32000
SHARES = (.3, .4, .45, .5)
DATA_SEEDS = (31000, 41001, 41002, 41003)  # 31000 is the existing e1/e2 ladder, for comparison


def seen(a, b, y, train):
    s = np.zeros((P, P), bool)
    s[a[train], y[train]] = True
    s[b[train], y[train]] = True
    return s


def ladder(seed):
    out, previous = dict(data_seed=seed, shares={}), None
    for share in SHARES:
        a, b, y, _, masks = data(P, seed, CONTROL_SEED, share)
        assert sorted(np.concatenate(masks).tolist()) == list(range(P*P)), 'splits must partition the pairs'
        for ix in masks:
            assert set((a[ix]*P + b[ix]).tolist()) == set((b[ix]*P + a[ix]).tolist()), 'transpose pair crosses a split'
        train = set(masks[0].tolist())
        assert previous is None or previous <= train, 'training sets must nest'
        previous = train
        out['shares'][str(share)] = dict(train=len(masks[0]), cal=len(masks[1]), audit=len(masks[2]),
                                         split_sha256=array_sha256(a, b, y, *masks))
    ev = np.array(sorted(set(range(P*P)) - previous))
    t30 = data(P, seed, CONTROL_SEED, .3)[4][0]
    s30, s50 = seen(a, b, y, t30), seen(a, b, y, np.array(sorted(previous)))
    assert not np.any((s50[a[ev]] | s50[b[ev]])[np.arange(len(ev)), y[ev]]), 'a+b seen with an operand'
    n_s = (s30[a[ev]] | s30[b[ev]]).sum(1)
    n_n = (~(s50[a[ev]] | s50[b[ev]])).sum(1) - 1
    counts = np.bincount(y[ev], minlength=P)
    np.savez_compressed(ROOT/f'research/e3/data/eval-d{seed}.npz', index=ev)
    out['evaluation'] = dict(pairs=len(ev), unordered_groups=len({(min(i, j), max(i, j)) for i, j in zip(a[ev], b[ev])}),
                             transpose_closed=set((a[ev]*P + b[ev]).tolist()) == set((b[ev]*P + a[ev]).tolist()),
                             per_class_min=int(counts.min()), per_class_max=int(counts.max()), sha256=array_sha256(ev),
                             S_size=dict(min=int(n_s.min()), mean=round(float(n_s.mean()), 2), max=int(n_s.max())),
                             N_size=dict(min=int(n_n.min()), mean=round(float(n_n.mean()), 2), max=int(n_n.max()),
                                         empty=int((n_n == 0).sum()), singleton=int((n_n == 1).sum())))
    return out


def main():
    report = [ladder(seed) for seed in DATA_SEEDS]
    (ROOT/'research/e3/coordinator/ladders.json').write_text(json.dumps(report, indent=1) + '\n')
    for r in report:
        print(r['data_seed'], [r['shares'][str(s)]['train'] for s in SHARES], r['evaluation']['pairs'],
              r['evaluation']['N_size'], r['evaluation']['sha256'][:12])


if __name__ == '__main__':
    main()
