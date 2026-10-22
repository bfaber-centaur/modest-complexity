"""Reviewer's oracle and split checks for R2 (REVIEW.md §2), written from the
brief and shared/QUESTION.md. Labels are rebuilt from Python integers (bit k of
i is x_{k+1}, as the lane's task.py docstring states); the lane module only
hands over its arrays. Also computes the baselines the primary metric is read
against on the actual test sets of the three data seeds.

uv run --locked python research/regimes/review/oracle_shared.py LANE_COMMIT
"""
from itertools import combinations, product
import sys
import numpy as np

from lanes import check, lane_module, FAIL

x = lambda i, k: (i >> (k-1)) & 1          # x_k of input i, k 1-indexed
MINE = dict(y1=lambda i: x(i, 1) ^ x(i, 2) ^ x(i, 3), y2=lambda i: (x(i, 1) ^ x(i, 2)) & x(i, 4),
            part=lambda i: x(i, 2) ^ x(i, 3) ^ x(i, 4), unrel=lambda i: x(i, 5) ^ x(i, 6) ^ x(i, 7))


def main(commit):
    T = lane_module(commit, 'shared')
    check('X row i is the binary expansion of i, x1 least significant',
          all(T.X[i, k-1] == x(i, k) for i in range(1024) for k in range(1, 11)))
    for t, f in MINE.items():
        check(f'lane label {t} equals independent rewrite; positives {sum(f(i) for i in range(1024))}/1024',
              all(int(T.LABELS[t][i]) == f(i) for i in range(1024)))
    check('on the x4 = 1 half of the domain y2 = z exactly and is balanced',
          all(MINE['y2'](i) == x(i, 1) ^ x(i, 2) for i in range(1024) if x(i, 4)) and
          sum(MINE['y2'](i) for i in range(1024) if x(i, 4)) == 256)
    neg = [i for i in range(1024) if MINE['y2'](i) == 0]; posr = [i for i in range(1024) if MINE['y2'](i) == 1]
    bal = .5*(np.mean([x(i, 4) == 1 for i in posr]) + np.mean([x(i, 4) == 0 for i in neg]))
    check('x4-only predictor: 0.75 accuracy, 0.833 balanced accuracy over the domain',
          np.isclose(np.mean([x(i, 4) == MINE['y2'](i) for i in range(1024)]), .75) and np.isclose(bal, 5/6), f'{bal:.4f}')
    for r in range(3):
        seed = 52100 + r
        pool, test = (np.asarray(a) for a in T.split(seed))
        perm = np.random.default_rng(seed).permutation(1024)   # what QUESTION.md says the split is
        check(f'seed {seed}: pool/test = the two halves of one permutation, disjoint, covering the domain',
              np.array_equal(np.concatenate([pool, test]), perm) and len(set(pool) | set(test)) == 1024 and not (set(pool) & set(test)))
        cov = {n: len({(x(i, 1), x(i, 2), x(i, 4)) for i in pool[:n]}) for n in (8, 16, 32, 64, 128, 256, 512)}
        lane_cov = {n: T.coverage(pool[:n], 'y2') for n in cov}
        h1 = [i for i in test if x(i, 4)]
        pos = sum(MINE['y2'](i) for i in h1)
        # Best z-blind rule on the x4 = 1 test rows: a lookup on any one or two bits other
        # than the pair (x1, x2), fitted on the pool's x4 = 1 rows, scored on test.
        p1 = [i for i in pool if x(i, 4)]
        best = (0, None)
        for S in [(k,) for k in range(1, 11)] + [c for c in combinations(range(1, 11), 2) if c != (1, 2)]:
            votes = {}
            for i in p1:
                key = tuple(x(i, k) for k in S); votes.setdefault(key, [0, 0])[MINE['y2'](i)] += 1
            acc = np.mean([(1 if votes.get(tuple(x(i, k) for k in S), [1, 0])[1] > votes.get(tuple(x(i, k) for k in S), [1, 0])[0]
                            else 0) == MINE['y2'](i) for i in h1])
            best = max(best, (acc, S))
        check(f'seed {seed}: lane coverage of (x1,x2,x4) per n equals recount', cov == lane_cov, str(cov))
        print(f'INFO seed {seed}: test x4=1 rows {len(h1)}, y2 positives {pos} (majority {max(pos, len(h1)-pos)/len(h1):.3f}); '
              f'best z-blind 1-2 bit lookup fitted on pool scores {best[0]:.3f} with bits {best[1]}; '
              f'D_n x4=1 rows: ' + ', '.join(f'n={n}:{sum(x(i, 4) for i in pool[:n])}' for n in (8, 16, 32, 64)))
    # Metric arithmetic: head_metrics against a per-row loop on random logits.
    Tr = lane_module(commit, 'shared', 'train.py')
    r = np.random.default_rng(55401); z = r.normal(0, 2, (1024, 2)); y = T.LABELS['y2']; idx = r.permutation(1024)[:300]
    m = Tr.head_metrics(z, y, idx)
    ce = np.mean([np.log(np.exp(z[i]).sum()) - z[i][y[i]] for i in idx])
    rec = [np.mean([z[i].argmax() == c for i in idx if y[i] == c]) for c in (0, 1)]
    check('head_metrics: accuracy, balanced accuracy and cross-entropy equal a per-row loop',
          np.isclose(m['acc'], np.mean([z[i].argmax() == y[i] for i in idx]), atol=1e-15)
          and np.isclose(m['bal_acc'], np.mean(rec), atol=1e-15) and np.isclose(m['loss'], ce, atol=1e-12))
    # Phase-1 pretraining data: the pool only.
    check('no test input is in any pool (phase 1 trains on the pool only, per train.py)',
          all(not (set(T.split(52100+r)[1]) & set(T.split(52100+r)[0])) for r in range(3)))


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
