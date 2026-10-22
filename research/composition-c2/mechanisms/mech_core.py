"""c2 mechanisms lane: the recovered checkpoints, verified loading, and whole-state patching.

Depth i is the state h_i after i ops (h_0 = start embedding). A donor prefix (s, o_1..o_i)
contributes h_i; a recipient at depth j runs a suffix o_{j+1}..o_3 from it and reads the
answer with the recipient's final readout (step 3; V3 when readouts are per-step: the
declared recipient-frame map, QUESTION.md). Every recovered condition shares its transition,
so patched == running the recombined program prefix ++ suffix (checked in test_mech.py).

Enumeration orders (used to index states without search):
  prefixes(i): op tuples lexicographic, start symbols 0..7 inside; index = lex(ops)*8 + s
  suffixes(m): op tuples of length m, lexicographic; index = lex(ops)
  rows of length-3 programs (c2task.rows over c2task.programs(3)) are prefixes(3).
"""
from itertools import product
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
C2 = HERE.parent
REPO = C2.parents[1]
sys.path[:0] = [str(C2/'common'), str(REPO/'research'/'regimes'/'common')]
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import nets  # noqa: E402

N_SYM, N_OP, LENGTH = c2task.N_SYM, c2task.N_OP, c2task.TRAIN_LENGTH
RECIPIENT_READOUT = 3
SEEDS = (51200, 51201, 51202)
STEPS = (0, 1000, 10000)
# condition label: (run-id template, directory of the published r1 run records)
CONDITIONS = {
    'tied/inter': ('r1-comp-rnn-inter-s{seed}-a1', 'research/regimes/composition/runs'),
    'per-step readout/inter': ('r1-rev-comp-rnn-untied-s{seed}-a1', 'research/regimes/review/runs'),
    'tied/final': ('r1-comp-rnn-final-s{seed}-a1', 'research/regimes/composition/runs'),
}


def checkpoints(steps=STEPS):
    """One dict per recovered checkpoint, with the SHA-256 recorded in RECOVERY.json."""
    rec = {(f['run_id'], f['file']): f['sha256'] for f in json.loads((C2/'recovered'/'RECOVERY.json').read_text())['files']}
    out = []
    for cond, (tmpl, published) in CONDITIONS.items():
        for seed in SEEDS:
            run = tmpl.format(seed=seed)
            for step in steps:
                out.append(dict(condition=cond, seed=seed, run_id=run, step=step,
                                path=C2/'recovered'/run/f'ckpt-{step}.npz', sha256=rec[(run, f'ckpt-{step}.npz')],
                                metrics=REPO/published/run/'metrics.jsonl'))
    return out


def load_verified(ck):
    sha = nets.sha256_file(ck['path'])
    if sha != ck['sha256']:
        raise ValueError(f"{ck['path']}: SHA-256 {sha} is not the recovered {ck['sha256']}")
    p, st, _ = nets.load(ck['path'])
    assert st['t'] == ck['step'], (st['t'], ck['step'])
    return p


def lex(ops):
    """Lexicographic index of each row of an (n, m) op array (base N_OP); m may be 0."""
    ops = np.asarray(ops, dtype=np.int64)
    return (ops * N_OP ** np.arange(ops.shape[1]-1, -1, -1)).sum(1)


def prefixes(depth):
    """dict(s (n,), ops (n, depth)) for every prefix of this depth, in the documented order."""
    q = np.array(list(product(range(N_OP), repeat=depth)), dtype=np.int64).reshape(N_OP**depth, depth)
    return dict(s=np.tile(np.arange(N_SYM), len(q)), ops=np.repeat(q, N_SYM, axis=0))


def suffixes(length):
    return np.array(list(product(range(N_OP), repeat=length)), dtype=np.int64).reshape(N_OP**length, length)


def apply_ops(P, x, ops):
    """Oracle: symbols x (n,) after ops (n, m) applied left to right."""
    x = np.array(x, dtype=np.int64)
    for t in range(ops.shape[1]):
        x = P[ops[:, t], x]
    return x


def state(p, s, ops):
    """h_depth for prefixes (s, ops); depth = ops.shape[1] (0 = start embedding)."""
    if ops.shape[1] == 0:
        return c2rnn.start(p, s)
    return c2rnn.forward(p, ops, s=s)[0][-1]


def run_suffix(p, h, ops, j):
    """Continue hidden states h, placed at recipient depth j, through the suffix ops (n, 3-j)."""
    if ops.shape[1] == 0:
        return h
    return c2rnn.forward(p, ops, h0=h, step_map=range(j+1, LENGTH+1))[0][-1]


def readout(p, h, k=RECIPIENT_READOUT):
    V, bv = ('V', 'bv') if 'V' in p else (f'V{k}', f'bv{k}')
    return h @ p[V] + p[bv]


def recombined_state(p, s, prefix_ops, suffix_ops):
    """Final hidden state of the plain forward pass on prefix ++ suffix (the equivalence check)."""
    ops = np.concatenate([prefix_ops, suffix_ops], 1)
    if ops.shape[1] == 0:
        return c2rnn.start(p, s)
    smap = [min(t, LENGTH) for t in range(1, ops.shape[1]+1)]  # transition is shared: values unused
    return c2rnn.forward(p, ops, s=s, step_map=smap)[0][-1]


def agreement(pred, group, n_class=N_SYM):
    """Mean over groups of the fraction of pairs of distinct members with equal predictions.
    pred (n, m): prediction of member n under condition m; group (n,): group label. Each
    (group, condition) cell is one unit; groups with fewer than two members are skipped."""
    out = []
    for g in np.unique(group):
        sel = pred[group == g]
        n = len(sel)
        if n < 2:
            continue
        counts = np.stack([(sel == k).sum(0) for k in range(n_class)])
        out.append((counts * (counts - 1)).sum(0) / (n * (n - 1)))
    return float(np.mean(out)) if out else float('nan')


def cross_agreement(pred_a, group_a, pred_b, group_b, n_class=N_SYM):
    """Mean over shared groups and conditions of P(pred_a == pred_b) for a member of a and one of b."""
    out = []
    for g in np.intersect1d(group_a, group_b):
        a, b = pred_a[group_a == g], pred_b[group_b == g]
        ca = np.stack([(a == k).sum(0) for k in range(n_class)])
        cb = np.stack([(b == k).sum(0) for k in range(n_class)])
        out.append((ca * cb).sum(0) / (len(a) * len(b)))
    return float(np.mean(out)) if out else float('nan')
