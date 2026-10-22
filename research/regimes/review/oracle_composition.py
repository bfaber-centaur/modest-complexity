"""Reviewer's oracle and split checks for R1 (REVIEW.md §1), written from the
brief and composition/QUESTION.md before reading the lane's code.

The permutation tables and Latin square are copied from QUESTION.md as data.
Every property and baseline QUESTION.md reports is recomputed here in plain
Python/NumPy. Then the lane's task.py is used only as a black box that hands
over its arrays, which must agree. It also answers a question QUESTION.md left
open: do the 48 training programs' final tables alone determine the four ops?

uv run --locked python research/regimes/review/oracle_composition.py LANE_COMMIT
"""
from collections import Counter
from itertools import combinations, permutations, product
import sys
import numpy as np

from lanes import check, lane_module, FAIL

P = np.array([[7, 0, 1, 2, 4, 6, 5, 3], [5, 0, 7, 2, 6, 1, 3, 4],
              [6, 3, 0, 5, 1, 2, 7, 4], [6, 1, 4, 3, 0, 5, 7, 2]])
L = [[0, 1, 3, 2], [3, 2, 0, 1], [2, 0, 1, 3], [1, 3, 2, 0]]
PROGS = list(product(range(4), repeat=3))
HELD = {(a, b, L[a][b]) for a in range(4) for b in range(4)}
TRAIN = [q for q in PROGS if q not in HELD]


def run(prog, s, perms=P):
    for k in prog:          # op 1 first: P[c][P[b][P[a][s]]]
        s = int(perms[k][s])
    return s


def table(prog, perms=P):
    return tuple(run(prog, s, perms) for s in range(8))


def cycles(p):
    seen, out = set(), []
    for i in range(8):
        if i not in seen:
            n, j = 0, i
            while j not in seen:
                seen.add(j); j = int(p[j]); n += 1
            out.append(n)
    return sorted(out, reverse=True)


def order(p):
    q, k = list(range(8)), 0
    while True:
        q = [int(p[x]) for x in q]; k += 1
        if q == list(range(8)):
            return k


def baselines():
    rows_tr = [(s, *q) for q in TRAIN for s in range(8)]
    rows_ho = [(s, *q) for q in sorted(HELD) for s in range(8)]
    ans = {r: run(r[1:], r[0]) for r in rows_tr + rows_ho}
    acc = lambda f: float(np.mean([f(r) for r in rows_ho]))
    out = dict(identity=acc(lambda r: r[0] == ans[r]))
    for i, nm in enumerate('abc'):
        out[f'single_{nm}'] = acc(lambda r, i=i: int(P[r[1+i]][r[0]]) == ans[r])
        out[f'drop_{i+1}'] = acc(lambda r, i=i: run(tuple(x for j, x in enumerate(r[1:]) if j != i), r[0]) == ans[r])
    best = 0.
    for k in (1, 2, 3):
        for S in combinations(range(4), k):
            votes = {}
            for r in rows_tr:
                votes.setdefault(tuple(r[j] for j in S), Counter())[ans[r]] += 1

            def expected(r, S=S):
                c = votes.get(tuple(r[j] for j in S))
                if not c:
                    return 1/8
                top = max(c.values()); win = [y for y, v in c.items() if v == top]
                return (ans[r] in win) / len(win)
            best = max(best, acc(expected))
    out['best_subset_lookup'] = best
    tr = set(rows_tr)

    def vote(r):
        nb = [x for x in tr if sum(u != v for u, v in zip(x, r)) == 1]
        assert len(nb) == 9, len(nb)
        c = Counter(ans[x] for x in nb); top = max(c.values()); win = [y for y, v in c.items() if v == top]
        return (ans[r] in win) / len(win)
    out['neighbour_vote'] = acc(vote)
    out['nearest_program_oracle'] = float(np.mean([max(np.mean(np.array(table(h)) == np.array(table(t))) for t in TRAIN)
                                                   for h in HELD]))
    return out


def identifiable():
    """All 4-tuples of permutations reproducing the 48 training tables. The first
    slot gives P_k = P_0 o D_k with D_k known; then search all 8! choices of P_0."""
    T = {q: np.array(table(q)) for q in TRAIN}
    D = [np.arange(8)]
    for k in (1, 2, 3):
        b, c = next((b, c) for b in range(4) for c in range(4) if (0, b, c) in T and (k, b, c) in T)
        D.append(np.argsort(T[(0, b, c)])[T[(k, b, c)]])   # P_0^-1 o P_k
    C = np.array(list(permutations(range(8))))           # candidate P_0, 40320 x 8
    Pk = [C[:, d] for d in D]                             # P_k = P_0 o D_k
    ok = np.ones(len(C), bool)
    for (a, b, c), t in T.items():
        x = np.take_along_axis(Pk[b], Pk[a], 1); x = np.take_along_axis(Pk[c], x, 1)
        ok &= (x == t).all(1)
    sols = np.flatnonzero(ok)
    return len(sols), all(np.array_equal(Pk[k][sols[0]], P[k]) for k in range(4)) if len(sols) else False


def main(commit):
    check('L is a Latin square', all(sorted(r) == [0, 1, 2, 3] for r in L) and all(sorted(c) == [0, 1, 2, 3] for c in zip(*L)))
    check('no op is the identity, an involution, or equal to another',
          all(order(p) > 2 for p in P) and len({tuple(p) for p in P}) == 4)
    print('INFO cycle types', [cycles(p) for p in P], 'fixed points', [[i for i in range(8) if p[i] == i] for p in P],
          'orders', [order(p) for p in P])
    check('cycle types / fixed points / orders as QUESTION.md table',
          [cycles(p) for p in P] == [[5, 2, 1], [5, 3], [8], [5, 1, 1, 1]] and [order(p) for p in P] == [10, 15, 8, 5]
          and [[i for i in range(8) if p[i] == i] for p in P] == [[4], [], [], [1, 3, 5]])
    check('no pair of ops commutes', all(table((i, j)) != table((j, i)) for i, j in combinations(range(4), 2)))
    check('16 length-2 and 64 length-3 products are all distinct',
          len({table(q) for q in product(range(4), repeat=2)}) == 16 and len({table(q) for q in PROGS}) == 64)
    check('no held-out program equals any shorter program (lengths 1-2) either',
          not ({table(h) for h in HELD} & ({table((k,)) for k in range(4)} | {table(q) for q in product(range(4), repeat=2)})))
    pos = [Counter(q[i] for q in TRAIN) for i in range(3)]
    pairs = {ij: Counter((q[ij[0]], q[ij[1]]) for q in TRAIN) for ij in ((0, 1), (1, 2), (0, 2))}
    check('every op 12x at every position; every ordered pair 3x at slot pairs (1,2), (2,3), (1,3)',
          all(set(c.values()) == {12} and len(c) == 4 for c in pos) and all(set(c.values()) == {3} and len(c) == 16 for c in pairs.values()))
    trans = [{(run(q[:i], s), q[i]) for q in TRAIN for s in range(8)} for i in range(3)]
    check('all 32 (state, op) transitions at each position; all 128 (s,a,b) prefixes and (step1,b,c) contexts',
          all(len(t) == 32 for t in trans) and len({(s, q[0], q[1]) for q in TRAIN for s in range(8)}) == 128
          and len({(run(q[:1], s), q[1], q[2]) for q in TRAIN for s in range(8)}) == 128)
    print('INFO held-out repetition pattern', Counter(('aaa' if a == b == c else 'a=b' if a == b else 'b=c' if b == c
                                                       else 'a=c' if a == c else 'distinct') for a, b, c in HELD))
    base = baselines()
    print('INFO baselines', {k: round(v, 3) for k, v in base.items()})
    want = dict(identity=.102, single_a=.063, single_b=.102, single_c=.063, drop_1=.125, drop_2=.125, drop_3=.125,
                best_subset_lookup=.125, neighbour_vote=.077, nearest_program_oracle=.453)
    check('every baseline in QUESTION.md reproduced to 3 decimals', all(abs(base[k] - v) < 6e-4 for k, v in want.items()))
    n, same = identifiable()
    print(f'INFO tuples of permutations reproducing all 48 training final tables: {n} (equal to P: {same})')

    T = lane_module(commit, 'composition')
    Pl, used = T.draw_ops()
    check('lane draw_ops() equals the QUESTION.md tables, first seed accepted', np.array_equal(Pl, P) and used == 51000)
    Ll, tr, ho = T.split()
    check('lane split: same square, same 48 train / 16 held-out programs',
          [list(map(int, r)) for r in Ll] == L and {tuple(map(int, q)) for q in tr} == set(TRAIN)
          and {tuple(map(int, q)) for q in ho} == HELD)
    sq = T.latin_squares()
    check('lane enumerates 576 distinct 4x4 Latin squares', len(sq) == 576 and len({str(np.asarray(x).tolist()) for x in sq}) == 576)
    for name, progs in (('train', tr), ('held', ho)):
        d = T.rows(Pl, progs); X = np.asarray(T.encode(d))
        n = len(d['s']); ok = n == 8*len(progs)
        for i in range(n):
            s0, a, b, c = (int(d[k][i]) for k in 'sabc')
            ok &= (int(d['step1'][i]), int(d['step2'][i]), int(d['final'][i])) == (run((a,), s0), run((a, b), s0), run((a, b, c), s0))
            ok &= X[i].tolist() == [float(j in (s0, 8+a, 12+b, 16+c)) for j in range(20)]
        ok &= len({(int(d['s'][i]), int(d['a'][i]), int(d['b'][i]), int(d['c'][i])) for i in range(n)}) == n
        check(f'lane {name} rows: step1/step2/final labels and slot one-hot encoding match ({n} rows)', bool(ok))
    return T

if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
