"""Reviewer's oracle and split checks for R3 (REVIEW.md §3), written from the
brief and history/QUESTION.md. Labels come from Python integers and format
strings; the lane module only hands over its arrays. The plan check confirms
every branch run names the A checkpoint by hash, so a branch cannot silently
start from a different file.

uv run --locked python research/regimes/review/oracle_history.py LANE_COMMIT
"""
from itertools import combinations
import json
import subprocess
import sys
import numpy as np

from lanes import check, lane_module, FAIL, ROOT

BITS = dict(A=(0, 1, 2), Brel=(0, 1, 3), Bunrel=(4, 5, 6))


def bit(i, k):
    return int(format(i, '010b')[::-1][k])     # x_k = (i >> k) & 1, via the string


def lab(i, task):
    if task == 'notA':
        return 1 - lab(i, 'A')
    return sum(bit(i, k) for k in BITS[task]) % 2


def main(commit):
    H = lane_module(commit, 'history', 'history.py')
    X = H.domain()
    check('domain row i has x_k = (i >> k) & 1 for all 1024 x 10 entries',
          all(int(X[i, k]) == bit(i, k) for i in range(1024) for k in range(10)))
    for t in ('A', 'Brel', 'Bunrel', 'notA'):
        y = H.labels(X, t)
        check(f'lane labels {t} equal the string-bit rewrite; balance {int(np.sum(y))}/1024',
              all(int(y[i]) == lab(i, t) for i in range(1024)) and np.sum(y) == 512)
    for a, b in combinations(('A', 'Brel', 'Bunrel'), 2):
        agree = np.mean([lab(i, a) == lab(i, b) for i in range(1024)])
        check(f'{a} vs {b}: label agreement over the domain is exactly 1/2', agree == .5)
    train, test = H.split()
    mine = np.sort(np.random.default_rng(53100).choice(1024, 256, replace=False))
    check('split: 256/768, disjoint, covers the domain, equals choice(1024, 256) at seed 53100',
          np.array_equal(train, mine) and len(test) == 768 and not (set(train) & set(test)) and len(set(train) | set(test)) == 1024)
    for t in ('A', 'Brel', 'Bunrel'):
        pos = sum(lab(i, t) for i in train)
        pats = [sum(1 for i in train if tuple(bit(i, k) for k in BITS[t]) == p) for p in
                [tuple((v >> j) & 1 for j in range(3)) for v in range(8)]]
        print(f'INFO train {t}: positives {pos}/256, per-pattern counts of its 3 bits {pats}')
        check(f'train {t}: all 8 patterns of its relevant bits present', min(pats) > 0)
    r = np.random.default_rng(55402); z = r.normal(0, 2, (500, 2)); y = r.integers(0, 2, 500)
    acc, ce = H.ce_acc(z, y)
    check('ce_acc equals a per-row loop', np.isclose(acc, np.mean([z[i].argmax() == y[i] for i in range(500)]))
          and np.isclose(ce, np.mean([np.log(np.exp(z[i]).sum()) - z[i][y[i]] for i in range(500)]), atol=1e-12))
    # Plan: every run that starts from a checkpoint names its SHA-256.
    plan = json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:research/regimes/history/plan.json']))
    branch = [r for r in plan['runs'] if '--from' in r['argv']]
    named = [r for r in branch if '--from-sha256' in r['argv']]
    srcs = {}
    for r in named:
        a = r['argv']; srcs.setdefault(a[a.index('--from')+1], set()).add(a[a.index('--from-sha256')+1])
    check(f'every checkpoint-started plan entry passes --from-sha256 ({len(named)}/{len(branch)})',
          len(branch) > 0 and len(named) == len(branch))
    check('each source checkpoint path is named with exactly one hash across all its branches',
          all(len(v) == 1 for v in srcs.values()), f'{len(srcs)} source files')
    return srcs


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
