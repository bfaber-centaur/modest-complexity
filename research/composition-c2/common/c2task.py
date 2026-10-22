"""c2 task layer: r1's composition task (research/regimes/composition/task.py, imported
unchanged under the unique name `r1_comp_task`), plus task/split IDs, programs of any
length, and the one-op-slot neighbour classification of held-out rows.

A task is (permutation draw, split). Its ID is f'ops{task_seed}-ls{split_seed}', from the
seeds requested; `task_seed_used` records any rejection step of r1's draw rule. The original
r1 task is ORIGINAL = 'ops51000-ls51100'.

Rows of length-L programs: every (start, program) pair, programs in the order given, start
symbols 0..7 within each program (r1's order). `ops[:, t]` is the op applied at step t+1 and
`states[:, t]` the symbol after it, so `states[:, -1]` is the answer.
"""
import importlib.util
from itertools import product
from pathlib import Path
import numpy as np

R1_DIR = Path(__file__).resolve().parents[2]/'regimes'/'composition'
_spec = importlib.util.spec_from_file_location('r1_comp_task', R1_DIR/'task.py')
r1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(r1)

N_SYM, N_OP, TRAIN_LENGTH = r1.N_SYM, r1.N_OP, r1.LENGTH
ORIGINAL = f'ops{r1.TASK_SEED}-ls{r1.SPLIT_SEED}'


def task_id(task_seed, split_seed):
    return f'ops{task_seed}-ls{split_seed}'


def parse_id(tid):
    ops, ls = tid.split('-')
    assert ops.startswith('ops') and ls.startswith('ls'), tid
    return int(ops[3:]), int(ls[2:])


def load(tid=ORIGINAL):
    """dict(id, P, task_seed_used, L, train, held): r1's draw and Latin-square split for these seeds."""
    task_seed, split_seed = parse_id(tid)
    P, used = r1.draw_ops(task_seed)
    L, train, held = r1.split(split_seed)
    return dict(id=tid, P=P, task_seed_used=used, L=L, train=train, held=held)


def programs(length):
    return list(product(range(N_OP), repeat=length))


def rows(P, progs):
    """dict(s (n,), ops (n, L), states (n, L)) for every (start, program); all programs one length."""
    progs = [tuple(q) for q in progs]
    L = len(progs[0]); assert all(len(q) == L for q in progs)
    s = np.tile(np.arange(N_SYM), len(progs))
    ops = np.repeat(np.array(progs, dtype=np.int64).reshape(-1, L), N_SYM, axis=0)
    states, x = np.zeros_like(ops), s.copy()
    for t in range(L):
        x = P[ops[:, t], x]; states[:, t] = x
    return dict(s=s, ops=ops, states=states)


def from_r1(d):
    """r1's row dict (s, a, b, c, step1, step2, final) in this module's format."""
    return dict(s=d['s'], ops=np.stack([d[k] for k in 'abc'], 1), states=np.stack([d[k] for k in ('step1', 'step2', 'final')], 1))


def neighbour_answerable(P, train, held):
    """Boolean per held-out row (rows(P, held) order): the true answer equals the final label of a
    training row that differs from it in exactly one op slot (r1's N subset; ~N is 42 of 128 for
    ORIGINAL). Also returns whether every one-slot neighbour of every held-out row is a training row."""
    tr, he = rows(P, train), rows(P, held)
    lookup = {(int(s), *map(int, q)): int(y) for s, q, y in zip(tr['s'], tr['ops'], tr['states'][:, -1])}
    out, all_train = [], True
    for s, q, y in zip(he['s'], he['ops'], he['states'][:, -1]):
        labels, q = set(), list(map(int, q))
        for j in range(len(q)):
            for v in range(N_OP):
                if v != q[j]:
                    key = (int(s), *q[:j], v, *q[j+1:])
                    all_train &= key in lookup
                    if key in lookup:
                        labels.add(lookup[key])
        out.append(int(y) in labels)
    return np.array(out), all_train
