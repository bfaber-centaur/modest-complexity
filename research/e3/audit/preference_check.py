#!/usr/bin/env python3
"""First e3 audit investigation (QUESTION.md): scale-free pairwise preference of
operand-associated labels (S) over unrelated labels (N) on the 483 common pairs of the e2
nested ladder, from published checkpoints only.

Rows: every checkpoint of the 12 width-128 ladder runs (30/40/45/50%), the three
shuffled-label and three no-decay 30% runs (scored with the same a+b label sets), and the
closed-form memorizer of e2 audit cycle 4 at the four shares. Each row also records whether
the per-pair values survive scaling and offsets unchanged, and the number of ties.

uv run --locked python research/e3/audit/preference_check.py OUT.csv
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(root/'research/e2/audit'))
sys.path.insert(0, str(root/'research/audit'))
from pairwise import preference, ladder_sets  # noqa: E402
import margins_check as e2  # noqa: E402  (run lists, checkpoint lookup and cycle-4 label sets)
import independent as ind  # noqa: E402

P = 31
a, b, COMMON = e2.a, e2.b, e2.COMMON
S, D, N, Y = ladder_sets(a, b, e2.tr30, e2.tr50, COMMON, P)
assert all(np.array_equal(x, y) for x, y in ((S, e2.S), (D, e2.D), (N, e2.N)))  # same sets as cycle 4
C = np.eye(P, dtype=bool)[Y]
ca, cb = a[COMMON], b[COMMON]
GROUP = np.unique(np.minimum(ca, cb)*P + np.maximum(ca, cb), return_inverse=True)[1]  # transpose groups
WEIGHT = (S.sum(1)*N.sum(1)).astype(float)
ST = e2.ST
CONTROLS = {'shuffled 30%': [ST/f'e1-st-shuf-w128-i{s}-a1' for s in (30000, 30001, 30002)],
            'no decay 30%': [ST/f'e1-st-add0-w128-i{s}-a1' for s in (30000, 30001, 30002)]}
OFFSET = np.random.default_rng(37100).normal(0, 10, (len(COMMON), 1))


def measures(z):
    pSN, tSN = preference(z, S, N)
    out = dict(pairs_defined=int(np.isfinite(pSN).sum()), P_SN=float(np.nanmean(pSN)),
               P_SN_pooled=float(np.nansum(pSN*WEIGHT)/WEIGHT[np.isfinite(pSN)].sum()),
               P_SN_transpose_groups=float(np.mean(np.bincount(GROUP, pSN)/np.bincount(GROUP))),
               P_SN_median=float(np.nanmedian(pSN)), ties_SN=tSN)
    for name, (A, B) in dict(P_DN=(D, N), P_SD=(S, D), P_cN=(C, N), P_cS=(C, S), P_cD=(C, D)).items():
        v, t = preference(z, A, B)
        out[name] = float(np.nanmean(v)); out['ties_' + name[2:]] = t
    out['acc'] = float(np.mean(z.argmax(1) == Y))
    # Invariance: per-pair values must be identical under positive scaling and common offsets.
    same = True
    for zt in (0.37*z, 1000*z, z + 5, z + OFFSET):
        v, _ = preference(zt, S, N)
        same &= bool(np.array_equal(v, pSN))
    out['invariant'] = same
    return out


def logits(U, V, W):
    return np.maximum(U[ca] + V[cb], 0) @ W


def ce_table(run):
    return {r['step']: r['objective']['train']['cross_entropy']
            for r in map(json.loads, (run/'metrics.jsonl').read_text().splitlines())}


def rows_for(run, kind, share):
    ce = ce_table(run)
    out = []
    for step, f in e2.checkpoints(run).items():
        with np.load(f) as w:
            assert int(w['step']) == step
            out.append(dict(kind=kind, run_id=run.name, share=share, init_seed=int(run.name.split('-i')[1][:5]),
                            step=step, train_ce=ce[step], **measures(logits(w['U'], w['V'], w['W']))))
    return out


def memorizer_rows():
    """e2 audit cycle 4's closed-form memorizer: step-0 U, V; W = ridge fit to one-hot a+b."""
    out = []
    for frac in (.3, .4, .45, .5):
        tr = ind.regenerate_split(P, 31000, 32000, frac)[4][0]
        for seed in (30000, 30001, 30002):
            U, V, _ = ind.initial_weights(P, 128, seed)
            H = np.maximum(U[a[tr]] + V[b[tr]], 0)
            G = H.T @ H
            for r in (1e-6, 1e-2):
                W = np.linalg.solve(G + r*np.trace(G)/len(G)*np.eye(len(G)), H.T @ (np.eye(P)[(a[tr] + b[tr]) % P] - 1/P))
                out.append(dict(kind=f'memorizer lambda {r:g}', run_id=f'ridge-{r:g}-i{seed}', share=f'{round(100*frac)}%',
                                init_seed=seed, step='', train_ce='', **measures(logits(U, V, W))))
    return out


if __name__ == '__main__':
    rows = []
    for share, runs in e2.RUNS.items():
        for run in runs:
            rows += rows_for(run, 'a+b, decay 1', share)
    for kind, runs in CONTROLS.items():
        for run in runs:
            rows += rows_for(run, kind, '30%')
    rows += memorizer_rows()
    with open(sys.argv[1], 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader()
        w.writerows({k: (float(f'{v:.10g}') if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
