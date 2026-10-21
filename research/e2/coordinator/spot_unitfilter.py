#!/usr/bin/env python3
"""Coordinator spot check of the mechanisms lane's per-unit frequency filter (packet 2),
written without unitfilter.py: explicit cos/sin projections instead of rfft.

Each unit keeps the mean of its U and V columns plus their projection onto its dominant
input frequency k (largest summed cos/sin energy of U and V, k = 1..15); its W row keeps
only the frequency-k projection. Accuracy is plain argmax against a+b. Adds the e1
no-decay width-128 runs, which fit the same a+b training labels, as a further comparison.

uv run --locked python research/e2/coordinator/spot_unitfilter.py > research/e2/coordinator/spot_unitfilter.jsonl
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
P = 31
CO, ST = 'research/coordinator/runs', 'research/size-time/runs'
RUNS = ([(f'{CO}/e1-co-add-w128-i30000-a1', s) for s in (0, 50000)] + [(f'{ST}/e1-st-add-w128-i30001-a1', 50000)] +
        [(f'{CO}/e1f2-co-add-w128-i30000-tf50-a1', 50000)] + [(f'{ST}/e1-st-add0-w128-i{i}-a1', 50000) for i in (30000, 30001, 30002)])


def basis(k):
    t = 2*np.pi*k*np.arange(P)/P
    b = np.stack([np.cos(t), np.sin(t)], 1)
    return b/np.linalg.norm(b, axis=0)


def filtered(U, V, W):
    U2, V2, W2, conc = np.zeros_like(U), np.zeros_like(V), np.zeros_like(W), []
    for j in range(U.shape[1]):
        e = np.array([np.sum((basis(k).T @ U[:, j])**2) + np.sum((basis(k).T @ V[:, j])**2) for k in range(1, 16)])
        B = basis(1 + int(e.argmax()))
        conc.append(e.max()/e.sum())
        U2[:, j] = U[:, j].mean() + B @ (B.T @ U[:, j])
        V2[:, j] = V[:, j].mean() + B @ (B.T @ V[:, j])
        W2[j] = B @ (B.T @ W[j])
    return U2, V2, W2, float(np.mean(conc))


def main():
    common = np.load(ROOT/'research/e2/data/common-never-trained-30-50.npz')['index']
    for run, step in RUNS:
        ck, d = np.load(ROOT/run/f'checkpoint_{step:06d}.npz'), np.load(ROOT/run/'data.npz')
        a, b = d['a'], d['b']
        y = (a + b) % P
        acc = lambda w, ix: float(np.mean((np.maximum(w[0][a[ix]] + w[1][b[ix]], 0) @ w[2]).argmax(1) == y[ix]))
        w = [ck['U'], ck['V'], ck['W']]
        *f, conc = filtered(*w)
        print(json.dumps(dict(run_id=Path(run).name, step=step, input_concentration=round(conc, 3),
                              unedited_common=acc(w, common), filtered_common=acc(f, common),
                              filtered_train_vs_a_plus_b=acc(f, d['train']))))


if __name__ == '__main__':
    main()
