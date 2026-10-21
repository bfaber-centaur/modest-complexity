#!/usr/bin/env python3
"""Independent check of mechanisms packet 1 (research/e2/mechanisms/groups.py) and the
controls of QUESTION-2.md. Written from the packet's definitions, not its code:

- live unit: nonzero input (U, V) and output (W row) weights;
- dominant frequency: the k in 1..15 with the most non-DC input energy, from cos/sin
  projections of the unit's U and V columns;
- group G_k: the live units with dominant frequency k, read from the analysed checkpoint;
- contribution: relu(U[a] + V[b]) W over the group, centred over labels for each pair;
- d-function on a set of pairs: g(d) = mean over the pairs of the contribution at label
  c = a + b - d; share = 1 - SS(contribution - g(d)) / SS(contribution);
- own-frequency term: amplitude A and phase f of A cos(2 pi k d / p + f) in g; f = 0
  peaks at the correct label;
- ablation: logits without the group's units (no other unit's gate can change). Common-set
  accuracy, and RMS change of the centred logits on all pairs and on the common pairs.
- random sets: 16 per group, uniform over live units, same count, total |u_j||w_j| within
  10% (seed 35200, so the draws differ from the packet's).
Common set: the 483 pairs that neither the 30% nor the 50% split trains on.

uv run --locked python research/e2/audit/groups_check.py OUT.csv RUN_DIR:STEP,STEP ...
"""
import csv
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
P = 31
COMMON = np.load(root/'research/e2/data/common-never-trained-30-50.npz')['index']
x = np.arange(P)
COS = np.cos(2*np.pi*np.outer(np.arange(1, 16), x)/P); SIN = np.sin(2*np.pi*np.outer(np.arange(1, 16), x)/P)


def dominant(U, V):
    e = sum((COS @ X)**2 + (SIN @ X)**2 for X in (U, V))  # (15, hidden)
    return e.argmax(0) + 1, e.max(0)/e.sum(0)


def contribution(U, V, W, a, b, units):
    """Raw logit contribution of the units (pairs x labels)."""
    return np.maximum(U[a][:, units] + V[b][:, units], 0) @ W[units]


def readout(c, a, b, rows, k):
    cs = c[rows]
    dd = (a[rows, None] + b[rows, None] - x) % P  # d at label c, and equally the label c at d
    g = cs[np.arange(len(rows))[:, None], dd].mean(0)  # g[d], d = 0..30
    share = 1 - np.sum((cs - g[dd])**2)/np.sum(cs**2)
    coef = np.sum(g*np.exp(-2j*np.pi*k*x/P))
    return share, float(np.angle(coef)), float(2*np.abs(coef)/P)


def analyse(run, step, rng, draws=16):
    with np.load(run/'data.npz') as d:
        a, b, yadd, tr = d['a'], d['b'], d['yadd'], d['train']
    assert not set(COMMON.tolist()) & set(tr.tolist())
    with np.load(run/f'checkpoint_{step:06d}.npz') as z:
        U, V, W = z['U'], z['V'], z['W']
    live = np.flatnonzero(((U**2).sum(0) + (V**2).sum(0) > 0) & ((W**2).sum(1) > 0))
    dom, conc = dominant(U, V)
    size = np.sqrt((U**2).sum(0) + (V**2).sum(0))*np.linalg.norm(W, axis=1)
    base = np.maximum(U[a] + V[b], 0) @ W
    head = dict(run_id=run.name, step=step, live=len(live), input_concentration=round(float(conc[live].mean()), 4))

    def measure(units, k):
        raw = contribution(U, V, W, a, b, units)
        c = raw - raw.mean(1, keepdims=True)  # softmax-invariant part; also the centred logit change
        share, phase, amp = readout(c, a, b, COMMON, k)
        z = base - raw  # the other units' activations are untouched: no gate can change
        return dict(n_units=len(units), d_share_common=share, k_phase_common=phase, k_amplitude_common=amp,
                    common_acc=float(np.mean(z[COMMON].argmax(1) == yadd[COMMON])),
                    rms_change_all=float(np.sqrt(np.mean(c**2))), rms_change_common=float(np.sqrt(np.mean(c[COMMON]**2))))
    centred = base - base.mean(1, keepdims=True)
    out = [dict(head, kind='none', k=0, draw=0, n_units=0,
                common_acc=float(np.mean(base[COMMON].argmax(1) == yadd[COMMON])),
                rms_logits_common=float(np.sqrt(np.mean(centred[COMMON]**2))))]  # scale for the amplitudes
    for k in range(1, 16):
        group = live[dom[live] == k]
        if len(group) == 0:
            continue
        out.append(dict(head, kind='group', k=k, draw=0, **measure(group, k)))
        target, found = size[group].sum(), 0
        for _ in range(200*draws):
            pick = rng.choice(live, len(group), replace=False)
            if abs(size[pick].sum()/target - 1) <= .1:
                out.append(dict(head, kind='random set', k=k, draw=found, **measure(pick, k)))
                found += 1
                if found == draws:
                    break
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(35200)
    rows = []
    for spec in sys.argv[2:]:
        run, steps = spec.split(':')
        for step in map(int, steps.split(',')):
            rows += analyse(root/run, step, rng)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(sys.argv[1], 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
