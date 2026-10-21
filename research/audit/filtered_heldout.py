#!/usr/bin/env python3
"""Held-out accuracy after keeping only a few input frequencies (DC plus 1-3), on
a late checkpoint and on the same run's step-0 checkpoint (no learned structure).
Tests the mechanism lane's reading that above-chance held-out accuracy at small K
is within the spread of arbitrary filtering of a memorizing model. Independent code.

uv run --locked python research/audit/filtered_heldout.py RUN_DIR STEP [STEP ...]
"""
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402

run_dir = Path(sys.argv[1])
with np.load(run_dir/'data.npz') as z:
    # Scored against the addition label a+b, whatever the run was trained on.
    a, b, y = z['a'], z['b'], z['yadd']; held = np.concatenate([z['cal'], z['audit']])
Xa, Xb = ind.one_hot_inputs(a, b, 31)
for step in map(int, sys.argv[2:]):
    with np.load(run_dir/f'checkpoint_{step:06d}.npz') as z:
        w = [z['U'], z['V'], z['W']]
    row = dict(run=run_dir.name, step=step, heldout_pairs=len(held), chance=1/31)
    for K in (1, 2, 3):
        accs = []
        for freqs in combinations(range(1, 16), K):  # every subset of size K
            logits = ind.forward(ind.keep_frequencies(w, list(freqs)), Xa, Xb)[0]
            accs.append(float(np.mean(logits[held].argmax(1) == y[held])))
        accs = np.array(accs)
        row[f'K{K}'] = dict(subsets=len(accs), mean=round(float(accs.mean()), 4), min=round(float(accs.min()), 4),
                            max=round(float(accs.max()), 4), frac_above_chance=round(float(np.mean(accs > 1/31)), 3))
    print(json.dumps(row))
