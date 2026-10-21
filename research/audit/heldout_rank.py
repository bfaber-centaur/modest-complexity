#!/usr/bin/env python3
"""Where does the addition label a+b of a held-out pair rank, relative to the labels
that never co-occur with either operand among the run's *training targets*
("absent" labels)?

For an addition run the training targets are the addition labels, and a+b is always
absent. For a shuffled-label control the training targets carry no addition
structure; pairs whose a+b happens to co-occur with an operand are left out, and the
statistic asks whether training alone (without addition targets) lifts a+b. Under no
preference the mean normalized rank is 0.5 and the top-of-absent fraction is the mean
of 1/#absent ("chance_top_among_absent").

For each checkpoint: fraction of held-out argmax predictions that are present labels,
the probe label's mean normalized rank among the absent labels (0 = above all) and the
fraction of pairs where it tops them. Independent code only.

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
        A, B, Y, probe, tr = d['a'], d['b'], d['y'], d['yadd'], d['train']
        held = np.concatenate([d['cal'], d['audit']])
    p = int(probe.max()) + 1
    present = np.zeros((p, p), bool)  # present[x, c]: target c co-occurs with operand x in training
    present[A[tr], Y[tr]] = True; present[B[tr], Y[tr]] = True
    pres = present[A[held]] | present[B[held]]
    usable = ~pres[np.arange(len(held)), probe[held]]
    held, pres = held[usable], pres[usable]
    rows_ix = np.arange(len(held))
    Xa, Xb = ind.one_hot_inputs(A, B, p)
    out = []
    for path in sorted(run_dir.glob('checkpoint_*.npz')):
        with np.load(path) as z:
            logits = ind.forward((z['U'], z['V'], z['W']), Xa, Xb)[0][held]
            step = int(z['step'])
        true = logits[rows_ix, probe[held]]
        absent_other = ~pres; absent_other[rows_ix, probe[held]] = False
        above = ((logits > true[:, None]) & absent_other).sum(1)
        pred_present = pres[rows_ix, logits.argmax(1)]
        out.append(dict(step=step, pred_is_present_label=round(float(pred_present.mean()), 4),
                        true_rank_among_absent_mean=round(float((above / absent_other.sum(1)).mean()), 4),
                        true_top_among_absent=round(float((above == 0).mean()), 4)))
    chance_top = float(np.mean(1 / (~pres).sum(1)))
    trained_on = 'addition' if np.array_equal(Y, probe) else 'other targets (probe = addition label)'
    print(json.dumps(dict(run=run_dir.name, trained_on=trained_on, heldout_pairs=int(usable.size),
                          pairs_with_probe_absent=int(usable.sum()), chance_top_among_absent=round(chance_top, 4), rows=out)))
