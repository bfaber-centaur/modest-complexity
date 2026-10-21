#!/usr/bin/env python3
"""Where do held-out predictions go? A check on below-chance audit accuracy.

For a held-out pair (a, b), let L(a, b) be the labels of training pairs that share
exactly one operand position with it: (a, b') or (a', b). For addition the correct
label a+b is never in L(a, b) (that would need b' = b or a' = a), and transposes
share a split, so a model that answers with labels it has seen alongside either
operand would be wrong on every held-out pair. This script measures, from
checkpoints with its own forward pass (no imports from experiment.py):

- premise: fraction of audit pairs whose correct label is in L(a, b);
- pred_in_L: fraction of audit argmax predictions that fall in L(a, b);
- base_rate: the same fraction expected if each wrong prediction were uniform over
  the other p-1 labels;
- mass_L / mass_correct: mean softmax probability on L(a, b) and on the correct label.

uv run --locked python research/size-time/operand_labels.py RUN_DIR ... [--steps 1000,50000]
"""
import argparse
import json
from pathlib import Path
import numpy as np


def operand_label_sets(a, b, y, train, p):
    """For every input pair, a boolean (p,) row: labels of train pairs sharing a or b (same position)."""
    seen_a = np.zeros((p, p), bool); seen_b = np.zeros((p, p), bool)
    seen_a[a[train], y[train]] = True  # operand value in first position -> labels seen with it
    seen_b[b[train], y[train]] = True
    return seen_a[a] | seen_b[b]


def check(run_dir, step):
    d = np.load(run_dir/'data.npz'); ck = np.load(run_dir/f'checkpoint_{step:06d}.npz')
    a, b, y, train, audit = d['a'], d['b'], d['y'], d['train'], d['audit']
    p = ck['W'].shape[1]
    assert int(ck['step']) == step
    logits = np.maximum(ck['U'][a]+ck['V'][b], 0) @ ck['W']
    prob = np.exp(logits-logits.max(1, keepdims=True)); prob /= prob.sum(1, keepdims=True)
    L = operand_label_sets(a, b, y, train, p)[audit]
    ya, pred, pr = y[audit], logits[audit].argmax(1), prob[audit]
    rows = np.arange(len(audit))
    wrong_L = L.copy(); wrong_L[rows, ya] = False
    return dict(run=run_dir.name, step=step, audit_accuracy=float(np.mean(pred == ya)),
                premise=float(np.mean(L[rows, ya])), pred_in_L=float(np.mean(L[rows, pred])),
                base_rate=float(np.mean(wrong_L.sum(1)/(p-1))), mean_L_size=float(L.sum(1).mean()),
                mass_L=float(np.mean((pr*L).sum(1))), mass_correct=float(np.mean(pr[rows, ya])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--steps', default='1000,5000,20000,50000')
    ap.add_argument('--out', type=Path, help='Optional JSON lines output')
    args = ap.parse_args()
    results = []
    for run in args.runs:
        for step in map(int, args.steps.split(',')):
            if (run/f'checkpoint_{step:06d}.npz').exists():
                results.append(check(run, step))
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in results[-1].items()}))
    if args.out:
        args.out.write_text(''.join(json.dumps(r)+'\n' for r in results))


if __name__ == '__main__':
    main()
