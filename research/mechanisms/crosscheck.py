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
    print(json.dumps(dict(retention_rows=len(rows), max_abs_diff_vs_csv=worst,
                          targeted_units=int(units.sum()), targeted_max_abs_logit_change=float(np.abs(z1 - z0).max()),
                          targeted_top4=[int(k) for k in top],
                          freq_py_targeted_top4=sorted(int(k) for k in t['ranking'].split()[:4]))))


if __name__ == '__main__':
    main(Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
