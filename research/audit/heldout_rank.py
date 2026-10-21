#!/usr/bin/env python3
"""Where does the true label of a held-out pair rank, relative to the labels that
never co-occur with either operand in training ("absent" labels; the true label
is always one of them in modular addition)?

For each published checkpoint: fraction of held-out argmax predictions that are
"present" labels, and the true label's mean normalized rank among the absent
labels (0 = above every other absent label, 1 = below all; 0.5 = indistinguishable
from them on average). Independent code only.

uv run --locked python research/audit/heldout_rank.py RUN_DIR ...
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402

for run_dir in map(Path, sys.argv[1:]):
    with np.load(run_dir/'data.npz') as d:
        A, B, Y, tr = d['a'], d['b'], d['yadd'], d['train']; held = np.concatenate([d['cal'], d['audit']])
    p = int(Y.max()) + 1
    present = np.zeros((p, p), bool)  # present[x, c]: label c co-occurs with operand x in training
    present[A[tr], Y[tr]] = True; present[B[tr], Y[tr]] = True
    pres = present[A[held]] | present[B[held]]
    assert not pres[np.arange(len(held)), Y[held]].any()
    Xa, Xb = ind.one_hot_inputs(A, B, p)
    out = []
    for path in sorted(run_dir.glob('checkpoint_*.npz')):
        with np.load(path) as z:
            logits = ind.forward((z['U'], z['V'], z['W']), Xa, Xb)[0][held]
            step = int(z['step'])
        true = logits[np.arange(len(held)), Y[held]]
        absent_other = ~pres; absent_other[np.arange(len(held)), Y[held]] = False
        above = ((logits > true[:, None]) & absent_other).sum(1)
        norm_rank = above / absent_other.sum(1)
        pred_present = pres[np.arange(len(held)), logits.argmax(1)]
        out.append(dict(step=step, pred_is_present_label=round(float(pred_present.mean()), 4),
                        true_rank_among_absent_mean=round(float(norm_rank.mean()), 4),
                        true_top_among_absent=round(float((above == 0).mean()), 4)))
    chance_top = float(np.mean(1 / (~pres).sum(1)))  # no preference among absent labels
    print(json.dumps(dict(run=run_dir.name, heldout_pairs=len(held), chance_top_among_absent=round(chance_top, 4), rows=out)))
