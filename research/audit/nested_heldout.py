#!/usr/bin/env python3
"""Like-for-like held-out comparison for follow-up 2. With the same data seed the 50%
split is nested in the e1 split: its audit pairs are a subset of the e1 audit pairs.
Scores e1 and follow-up 2 checkpoints on the same pairs (the follow-up 2 audit split),
with independent code.

uv run --locked python research/audit/nested_heldout.py STEP RUN_DIR ...
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402

root = Path(__file__).resolve().parents[2]
with np.load(root/'research/data/e1f2-split-tf50.npz') as z:
    pairs = z['audit']
with np.load(root/'research/data/e1-split.npz') as z:
    assert set(pairs.tolist()) <= set(z['audit'].tolist())
    a, b, y = z['a'], z['b'], z['yadd']
Xa, Xb = ind.one_hot_inputs(a, b, 31)
step = int(sys.argv[1])
for run in map(Path, sys.argv[2:]):
    with np.load(run/f'checkpoint_{step:06d}.npz') as z:
        logits = ind.forward((z['U'], z['V'], z['W']), Xa, Xb)[0][pairs]
    print(json.dumps(dict(run=run.name, step=step, pairs=len(pairs), **{k: round(float(v), 4) for k, v in
                                                                        ind.evaluate(logits, y[pairs]).items()})))
