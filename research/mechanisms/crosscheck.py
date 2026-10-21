#!/usr/bin/env python3
"""Recompute selected rows of a freq.py output with separately written code: a
real cosine/sine projection instead of rfft, its own forward pass and metrics,
and its own rescaling. Imports nothing from freq.py or experiment.py.

uv run --locked python research/mechanisms/crosscheck.py RUN_DIR STEP research/mechanisms/figures/NAME
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np


def projector(p, freqs):
    """Orthogonal projector onto the token-space span of DC plus cos/sin at `freqs`."""
    t = np.arange(p)
    cols = [np.ones(p)] + [f(2*np.pi*k*t/p) for k in freqs for f in (np.cos, np.sin)]
    q, _ = np.linalg.qr(np.column_stack(cols))
    return q @ q.T


def logits(U, V, W, a, b):
    return np.maximum(U[a] + V[b], 0) @ W


def metrics(z, y):
    correct = z[np.arange(len(y)), y]
    others = np.where(np.eye(z.shape[1], dtype=bool)[y], -np.inf, z)
    lse = np.log(np.sum(np.exp(z - z.max(1, keepdims=True)), 1)) + z.max(1)
    return dict(accuracy=np.mean(z.argmax(1) == y), cross_entropy=np.mean(lse - correct),
                margin_median=np.median(correct - others.max(1)))


def main(run_dir, step, out_dir):
    ck = np.load(run_dir/f'checkpoint_{step:06d}.npz')
    d = np.load(run_dir/'data.npz')
    U, V, W, a, b, y = ck['U'], ck['V'], ck['W'], d['a'], d['b'], d['y']
    p = U.shape[0]
    rows = {(r['mode'], int(r['K'])): r for r in csv.DictReader(open(out_dir/'retention.csv'))}
    worst = 0.
    for (mode, k), row in rows.items():
        freqs = [int(f) for f in row['frequencies'].split()]
        P = projector(p, freqs) if mode == 'retain' else np.eye(p) - projector(p, freqs) + projector(p, [])
        z = logits(P @ U, P @ V, W, a, b)
        for split in ('train', 'audit'):
            ix = d[split]
            for name, value in metrics(z[ix], y[ix]).items():
                worst = max(worst, abs(value - float(row[f'{split}_{name}'])))
    summary = json.loads((out_dir/'summary.json').read_text())
    t = summary['targeted_rescaling']
    freqs = range(1, (p + 1)//2)
    band = [projector(p, [k]) - projector(p, []) for k in freqs]  # projector onto frequency k alone
    per_unit = np.array([np.sum((B @ U)**2, 0) + np.sum((B @ V)**2, 0) for B in band])
    c = np.where(per_unit.argmax(0) + 1 == t['target'], t['gain'], 1.)
    z0, z1 = logits(U, V, W, a, b), logits(U*c, V*c, W/c[:, None], a, b)
    energy = (per_unit*c**2).sum(1)
    top = sorted(np.argsort(-energy, kind='stable')[:4] + 1)
    units = c != 1
    out = dict(retention_rows=len(rows), max_abs_diff_vs_csv=worst,
               targeted_units=int(units.sum()), targeted_max_abs_logit_change=float(np.abs(z1 - z0).max()),
               targeted_top4=[int(k) for k in top],
               freq_py_targeted_top4=sorted(int(k) for k in t['ranking'].split()[:4]))
    if (out_dir/'gauge_summary.json').exists():
        # Balanced gauge without rescaling anything: each unit's input energy becomes
        # sqrt(input norm² x output norm²), split across frequencies in its own proportions.
        g = json.loads((out_dir/'gauge_summary.json').read_text())
        a_in, b_out = np.sum(U**2, 0) + np.sum(V**2, 0), np.sum(W**2, 1)
        balanced = (per_unit*np.sqrt(b_out/a_in)).sum(1)
        out['gauge_max_abs_share_diff'] = float(np.abs(balanced/balanced.sum() - g['balanced_energy_share']).max())
        out['gauge_ranking_equal'] = (np.argsort(-balanced, kind='stable') + 1).tolist() == g['ranking']['balanced']
    if (out_dir/'relational.csv').exists():
        # Held-out true-label rank among labels absent from both operands' training pairs,
        # via a sort instead of pairwise comparisons.
        tr, held = d['train'], np.concatenate([d['cal'], d['audit']])
        seen = set(zip(a[tr], y[tr])) | set(zip(b[tr], y[tr]))  # (operand, label) pairs in training
        worst_rel = worst_energy = 0.
        rel = list(csv.DictReader(open(out_dir/'relational.csv')))
        nondc = sum(np.sum((x - x.mean(0))**2) for x in (U, V))
        for row in rel:
            if row['kind'] != 'top-K':
                continue
            freqs = [int(f) for f in row['frequencies'].split()]
            P = projector(p, freqs) if row['mode'] == 'retain' else np.eye(p) - projector(p, freqs) + projector(p, [])
            z = logits(P @ U, P @ V, W, a, b)
            if 'delta_energy_fraction' in row:
                changed = sum(np.sum((P @ x - x)**2) for x in (U, V))/nondc
                worst_energy = max(worst_energy, abs(changed - float(row['delta_energy_fraction'])))
            ranks, tops = [], []
            for i in held:
                absent = [c for c in range(p) if (a[i], c) not in seen and (b[i], c) not in seen and c != y[i]]
                # Ties go below the true label, matching a strict 'greater than' count.
                order = sorted(absent + [y[i]], key=lambda c: (-z[i, c], c != y[i]))
                above = order.index(y[i])
                ranks.append(above/len(absent)); tops.append(above == 0)
            worst_rel = max(worst_rel, abs(np.mean(ranks) - float(row['true_rank_among_absent'])),
                            abs(np.mean(tops) - float(row['true_top_among_absent'])))
        out['relational_topK_rows_max_abs_diff'] = worst_rel
        out['relational_topK_energy_fraction_max_abs_diff'] = worst_energy
        sub = [r for r in rel if r['kind'] == 'random-subspace null']
        if sub:
            # A uniformly random 2K-dim subspace of the 30-dim zero-mean space holds 2K/30 of
            # any fixed energy on average; at K = 15 it is the whole space, leaving DC only.
            ks = sorted({int(r['K']) for r in sub})
            mean = {k: np.mean([float(r['delta_energy_fraction']) for r in sub if int(r['K']) == k]) for k in ks}
            out['subspace_null_mean_energy_fraction_minus_2K_over_30'] = {k: round(float(mean[k] - 2*k/(p - 1)), 4) for k in ks}
            dc = next(r for r in rel if r['kind'] == 'top-K' and r['mode'] == 'remove' and int(r['K']) == (p - 1)//2)
            keys = ('train_accuracy', 'true_rank_among_absent', 'true_top_among_absent')
            out['subspace_null_full_rank_max_abs_diff_vs_dc_only'] = max(
                abs(float(r[k]) - float(dc[k])) for r in sub if int(r['K']) == (p - 1)//2 for k in keys)
    print(json.dumps(out))


if __name__ == '__main__':
    main(Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
