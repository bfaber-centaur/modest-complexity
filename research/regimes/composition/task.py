"""Composing small programs: the task, the held-out split, and pre-training checks.

Eight symbols, four fixed random permutations ("ops"). A program is an ordered
triple of ops (a, b, c) applied left to right to a start symbol s:
    step1 = P[a][s], step2 = P[b][step1], final = P[c][step2].
Inputs are slot one-hots: start (8) + op slot 1 (4) + op slot 2 (4) + op slot 3 (4).

Held-out split: a 4x4 Latin square L, drawn uniformly from all 576 with the
split seed, holds out the 16 programs (a, b, L[a][b]) at every start symbol.
The other 48 programs contain every op 12 times at every position and every
ordered op pair exactly 3 times at slot pairs (1,2), (2,3) and (1,3).

Rule for the permutations, fixed before the draw: draw four with the task seed
and reject the draw only if an op is the identity or two ops are equal (then
use task seed + 1, and so on, and record it). Everything else is reported.

uv run --locked python research/regimes/composition/task.py   # prints the checks as JSON
"""
from collections import Counter
from itertools import combinations, permutations as all_orders, product
import json
import numpy as np

N_SYM, N_OP, LENGTH = 8, 4, 3
TASK_SEED, SPLIT_SEED = 51000, 51100
PROGRAMS = list(product(range(N_OP), repeat=LENGTH))  # 64, lexicographic


def draw_ops(seed=TASK_SEED):
    """P[k][s] is op k applied to symbol s. Returns (P, seed actually used)."""
    while True:
        P = np.stack([np.random.default_rng([seed, k]).permutation(N_SYM) for k in range(N_OP)])
        if not any((P[k] == np.arange(N_SYM)).all() for k in range(N_OP)) and len({p.tobytes() for p in P}) == N_OP:
            return P, seed
        seed += 1


def execute(P, s, program):
    """States after each op, in order; the last one is the answer."""
    states = []
    for k in program:
        s = int(P[k][s]); states.append(s)
    return states


def latin_squares(n=N_OP):
    """All n x n Latin squares (rows are permutations, columns distinct), in a fixed order."""
    rows = list(all_orders(range(n)))
    out = []

    def extend(square):
        if len(square) == n:
            out.append(np.array(square)); return
        for r in rows:
            if all(r[j] != q[j] for q in square for j in range(n)):
                extend(square + [r])
    extend([])
    return out


def split(seed=SPLIT_SEED):
    """(L, train programs, held-out programs), programs in lexicographic order."""
    squares = latin_squares()
    L = squares[np.random.default_rng(seed).integers(len(squares))]
    held = sorted((a, b, int(L[a][b])) for a in range(N_OP) for b in range(N_OP))
    train = [q for q in PROGRAMS if q not in set(held)]
    return L, train, held


def rows(P, programs):
    """Every (start, program) pair: integer columns s, a, b, c and labels step1, step2, final."""
    R = [(s, *q, *execute(P, s, q)) for q in programs for s in range(N_SYM)]
    R = np.array(R, dtype=np.int64)
    return dict(s=R[:, 0], a=R[:, 1], b=R[:, 2], c=R[:, 3], step1=R[:, 4], step2=R[:, 5], final=R[:, 6])


def encode(d):
    X = np.zeros((len(d['s']), N_SYM + LENGTH*N_OP))
    n = np.arange(len(d['s']))
    X[n, d['s']] = 1
    for i, slot in enumerate('abc'):
        X[n, N_SYM + i*N_OP + d[slot]] = 1
    return X


def product_table(P, program):
    """The program's overall permutation as a tuple over start symbols."""
    return tuple(execute(P, s, program)[-1] for s in range(N_SYM))


def cycle_type(p):
    seen, lengths = set(), []
    for s in range(len(p)):
        n = 0
        while s not in seen:
            seen.add(s); s = int(p[s]); n += 1
        if n:
            lengths.append(n)
    return sorted(lengths, reverse=True)


def checks(P, train, held):
    """Equivalences, coverage counts and non-trained baselines on held-out rows."""
    I = np.arange(N_SYM)
    ops = []
    for k, p in enumerate(P):
        order = next(n for n in range(1, 841) if (np.linalg.matrix_power(np.eye(N_SYM)[p], n) == np.eye(N_SYM)).all())
        ops.append(dict(op=k, table=p.tolist(), cycle_type=cycle_type(p), fixed_points=np.flatnonzero(p == I).tolist(),
                        order=order, identity=bool((p == I).all()), involution=bool((p[p] == I).all())))
    commuting = [(i, j) for i in range(N_OP) for j in range(i+1, N_OP) if (P[i][P[j]] == P[j][P[i]]).all()]
    len2 = Counter(product_table(P, q) for q in product(range(N_OP), repeat=2))
    len3 = {q: product_table(P, q) for q in PROGRAMS}
    groups = Counter(len3.values())
    train_tables = {len3[q] for q in train}
    cov = dict(
        op_at_position={f'pos{i+1}': [sum(q[i] == k for q in train) for k in range(N_OP)] for i in range(LENGTH)},
        ordered_pair_counts={f'{i+1}{j+1}': sorted(Counter((q[i], q[j]) for q in train).items())
                             for i, j in ((0, 1), (1, 2), (0, 2))})
    cov['ordered_pair_count_range'] = {k: [min(n for _, n in v), max(n for _, n in v), len(v)]
                                       for k, v in cov['ordered_pair_counts'].items()}
    del cov['ordered_pair_counts']
    tr, he = rows(P, train), rows(P, held)
    # Transitions (state before, op) -> state after, at each position, seen in training rows.
    cov['transitions_seen'] = {
        'pos1 (s,a)': len(set(zip(tr['s'], tr['a']))),
        'pos2 (step1,b)': len(set(zip(tr['step1'], tr['b']))),
        'pos3 (step2,c)': len(set(zip(tr['step2'], tr['c']))),
        'prefix (s,a,b)': len(set(zip(tr['s'], tr['a'], tr['b']))),
        'suffix-context (step1,b,c)': len(set(zip(tr['step1'], tr['b'], tr['c']))), 'of': [32, 32, 32, 128, 128]}
    s, a, b, c, y = he['s'], he['a'], he['b'], he['c'], he['final']
    heur = {'identity: s': s, 'first op only: P[a](s)': P[a, s], 'middle op only: P[b](s)': P[b, s],
            'last op only: P[c](s)': P[c, s], 'drop op 1: P[c](P[b](s))': P[c, P[b, s]],
            'drop op 2: P[c](P[a](s))': P[c, P[a, s]], 'drop op 3: P[b](P[a](s))': P[b, P[a, s]]}
    baselines = {'chance': 1/N_SYM, **{k: float(np.mean(v == y)) for k, v in heur.items()}}
    # Nearest seen program, chosen with the held-out table itself (an optimistic, oracle-selected bound).
    nearest = [max(sum(x == t for x, t in zip(len3[r], len3[q])) for r in train)/N_SYM for q in held]
    baselines['nearest seen program (oracle-selected)'] = float(np.mean(nearest))
    # Plurality vote of the training rows that share 3 of the 4 input slots (a lazy, kernel-like interpolation).
    lookup = {(int(t[0]), *map(int, t[1:4])): int(t[4]) for t in zip(tr['s'], tr['a'], tr['b'], tr['c'], tr['final'])}
    vote = []
    for x in zip(s, a, b, c, y):
        votes = Counter()
        for slot in range(4):
            for v in range(N_SYM if slot == 0 else N_OP):
                key = list(map(int, x[:4])); key[slot] = v
                if tuple(key) != tuple(map(int, x[:4])) and tuple(key) in lookup:
                    votes[lookup[tuple(key)]] += 1
        top = max(votes.values()); winners = [k for k, n in votes.items() if n == top]
        vote.append((int(x[4]) in winners)/len(winners))  # expected accuracy under a uniform tie break
    baselines['3-of-4-slot neighbour vote'] = float(np.mean(vote))
    # Partial-input lookups fitted on training rows: plurality final label given a subset of slots
    # (expected accuracy under a uniform tie break). The best of these is the strongest slot-subset shortcut.
    cols = dict(s=tr['s'], a=tr['a'], b=tr['b'], c=tr['c']); hcols = dict(s=s, a=a, b=b, c=c)
    partial = {}
    for n in (1, 2, 3):
        for subset in combinations('sabc', n):
            table = {}
            for key, label in zip(zip(*(cols[k] for k in subset)), tr['final']):
                table.setdefault(tuple(map(int, key)), Counter())[int(label)] += 1
            acc = []
            for key, label in zip(zip(*(hcols[k] for k in subset)), y):
                votes = table.get(tuple(map(int, key)))
                if not votes:
                    acc.append(1/N_SYM); continue
                top = max(votes.values()); winners = [k for k, m in votes.items() if m == top]
                acc.append((int(label) in winners)/len(winners))
            partial[''.join(subset)] = float(np.mean(acc))
    baselines['best partial-slot lookup (fitted on train)'] = max(partial.values())
    return dict(
        ops=ops, commuting_pairs=commuting,
        length2_distinct_products=len(len2), length2_collisions=sum(n-1 for n in len2.values() if n > 1),
        length3_distinct_products=len(groups), length3_collision_groups=[
            [list(q) for q in PROGRAMS if len3[q] == t] for t, n in groups.items() if n > 1],
        held_out_programs=[list(q) for q in held],
        held_out_with_product_equal_to_a_training_program=[list(q) for q in held if len3[q] in train_tables],
        held_out_with_repeated_ops=[list(q) for q in held if len(set(q)) < LENGTH],
        row_overlap_train_heldout=len({(int(x), *q) for q in train for x in range(N_SYM)} &
                                      {(int(x), *q) for q in held for x in range(N_SYM)}),
        coverage=cov, heldout_baselines=baselines, partial_slot_lookups=partial, nearest_program_per_heldout=nearest,
        final_label_counts={'train': np.bincount(tr['final'], minlength=N_SYM).tolist(),
                            'heldout': np.bincount(y, minlength=N_SYM).tolist()})


if __name__ == '__main__':
    P, used = draw_ops()
    L, train, held = split()
    print(json.dumps(dict(task_seed_used=used, latin_square=L.tolist(), **checks(P, train, held)), indent=1, default=int))
