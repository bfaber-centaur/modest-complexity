#!/usr/bin/env python3
"""Independent check of mechanisms packet 2 (research/e2/mechanisms/unitfilter.py) and the
controls of QUESTION-3.md. Written from the packet's definitions with cos/sin projections
(no FFT):

- live unit: nonzero input (U, V) and output (W row) weights; dominant frequency k_j: the
  k in 1..15 with the most non-DC energy in the unit's U and V columns;
- filter: each live unit's U and V columns reduced to their DC plus frequency k_j, and its
  W row to frequency k_j ("input DC removed": U and V to frequency k_j only);
- nulls (8 draws each, seed 36100): every unit filtered to a uniformly random other
  frequency; the dominant filter with each unit's kept U, V and W components rotated by
  independent uniform phases;
- scores: argmax accuracy against a+b on the 483 common pairs and the run's own cal +
  audit, train accuracy against the run's own labels; the fraction of (pair, live unit)
  gates relu(U[a] + V[b]) that the edit flips, over all 961 pairs.

Constructed control (QUESTION-3.md, item 3): for each init seed, the step-0 U and V with
W fitted in closed form (ridge, lambda = r * mean eigenvalue of the feature Gram matrix,
r in 1e-6, 1e-2) to the one-hot a+b labels (minus 1/p) of the 299 training pairs of the
30% split. No optimizer, no training run.

uv run --locked python research/e2/audit/unitfilter_check.py OUT.csv RUN_DIR:STEP,... [ridge:INIT_RUN_DIR ...] [ridge50:INIT_RUN_DIR ...]
"""
import csv
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
P, DRAWS = 31, 8
COMMON = np.load(root/'research/e2/data/common-never-trained-30-50.npz')['index']
n = np.arange(P)
TH = 2*np.pi*np.outer(np.arange(1, 16), n)/P  # (15, p) angles for k = 1..15
COS, SIN = np.cos(TH), np.sin(TH)


def coefficients(X, k, axis):
    """cos and sin coefficients of frequency k[j] along `axis` for every unit j: x_k = a cos + b sin."""
    Xu = X if axis == 0 else X.T  # (p, units)
    c, s = COS[k - 1].T, SIN[k - 1].T  # (p, units)
    return 2/P*(Xu*c).sum(0), 2/P*(Xu*s).sum(0), c, s


def component(X, k, axis, phase=None):
    a, b, c, s = coefficients(X, k, axis)
    if phase is not None:  # rotate a cos + b sin = Re[(a - ib) e^{i theta}] by phase
        z = (a - 1j*b)*np.exp(1j*phase)
        a, b = z.real, -z.imag
    out = a*c + b*s
    return out if axis == 0 else out.T


def dominant(U, V):
    e = sum((COS @ X)**2 + (SIN @ X)**2 for X in (U, V))
    return e.argmax(0) + 1, float((e.max(0)/e.sum(0)).mean())


def edit(w, k, live, dc=True, phases=None):
    U, V, W = (x.copy() for x in w)
    ph = phases if phases is not None else (None, None, None)
    for X, axis, p_ in ((U, 0, ph[0]), (V, 0, ph[1]), (W, 1, ph[2])):
        new = component(X[:, live] if axis == 0 else X[live], k[live], axis, p_)
        if axis == 0:
            X[:, live] = new + (X[:, live].mean(0) if dc else 0)
        else:
            X[live] = new
    return U, V, W


def score(w, d, gates0, live):
    U, V, W = w
    pre = U[d['a']] + V[d['b']]
    z = np.maximum(pre, 0) @ W
    acc = lambda ix, y: float(np.mean(z[ix].argmax(1) == y[ix]))
    held = np.concatenate([d['cal'], d['audit']])
    return dict(common_acc=acc(COMMON, d['yadd']), own_heldout_acc=acc(held, d['yadd']), train_acc=acc(d['train'], d['y']),
                gate_flip_frac=float(np.mean((pre[:, live] > 0) != gates0[:, live])))


def analyse(name, w, d, rng, step):
    U, V, W = w
    live = ((U**2).sum(0) + (V**2).sum(0) > 0) & ((W**2).sum(1) > 0)
    k, conc = dominant(U, V)
    k_out = ((COS @ W.T)**2 + (SIN @ W.T)**2).argmax(0) + 1
    gates0 = (U[d['a']] + V[d['b']]) > 0
    head = dict(run_id=name, step=step, live=int(live.sum()), input_concentration=round(conc, 4),
                in_out_agreement=round(float(np.mean(k[live] == k_out[live])), 4))
    rows = [dict(head, kind='none', draw=0, **score(w, d, gates0, live)),
            dict(head, kind='filter dominant', draw=0, **score(edit(w, k, live), d, gates0, live)),
            dict(head, kind='filter dominant, input DC removed', draw=0, **score(edit(w, k, live, dc=False), d, gates0, live))]
    for draw in range(DRAWS):
        other = (k - 1 + rng.integers(1, 15, len(k))) % 15 + 1
        rows.append(dict(head, kind='filter random non-dominant', draw=draw, **score(edit(w, other, live), d, gates0, live)))
    for draw in range(DRAWS):
        phases = tuple(2*np.pi*rng.random(len(k[live])) for _ in range(3))
        rows.append(dict(head, kind='filter dominant, phases scrambled', draw=draw,
                         **score(edit(w, k, live, phases=phases), d, gates0, live)))
    return rows


def load(run, step):
    with np.load(run/'data.npz') as z:
        d = {key: z[key] for key in z.files}
    assert not set(COMMON.tolist()) & set(d['train'].tolist())
    with np.load(run/f'checkpoint_{step:06d}.npz') as z:
        return (z['U'], z['V'], z['W']), d


def ridge(run, r, split='e1-split.npz'):
    """Step-0 U, V of `run` with W fitted in closed form to a+b on the training pairs of `split`."""
    (U, V, _), d = load(run, 0)
    with np.load(root/'research/data'/split) as z:
        tr = z['train']
        assert np.array_equal(z['a'], d['a']) and np.array_equal(z['b'], d['b'])
    H = np.maximum(U[d['a'][tr]] + V[d['b'][tr]], 0)
    Y = np.eye(P)[d['yadd'][tr]] - 1/P
    G = H.T @ H
    W = np.linalg.solve(G + r*np.trace(G)/len(G)*np.eye(len(G)), H.T @ Y)
    return (U, V, W), dict(d, y=d['yadd'], train=tr, cal=np.array([], int), audit=np.setdiff1d(np.arange(P*P), tr))


if __name__ == '__main__':
    rng = np.random.default_rng(36100)
    rows = []
    for spec in sys.argv[2:]:
        if spec.startswith('ridge'):  # ridge:RUN (30% split, planned) or ridge50:RUN (50% split, added post hoc)
            tag, run = spec.split(':'); run = root/run
            split = 'e1f2-split-tf50.npz' if tag == 'ridge50' else 'e1-split.npz'
            for r in (1e-6, 1e-2):
                w, d = ridge(run, r, split)
                rows += analyse(f'{tag}-{r:g}-step0-{run.name}', w, d, rng, 0)
            continue
        run, steps = spec.split(':')
        for step in map(int, steps.split(',')):
            w, d = load(root/run, step)
            rows += analyse(Path(run).name, w, d, rng, step)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(sys.argv[1], 'w', newline='') as f:
        out = csv.DictWriter(f, keys); out.writeheader()
        out.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
