#!/usr/bin/env python3
"""Checks that call the shared code as the object under test, on real e1 data.

1. Central finite differences of the independent loss against the independent
   analytic gradient (checks research/audit/independent.py itself).
2. experiment.lossgrad against the independent gradient on every element.
3. Positive control through the shared evaluation path: an exact modular-addition
   network in the trained architecture must score 1 on every split with
   experiment.forward/accuracy/objective_metrics.

uv run --locked python research/audit/shared_checks.py RUN_DIR STEP
"""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402
from audit_preview import perfect_lookup  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment import accuracy, forward, lossgrad, objective_metrics  # noqa: E402  (objects under test)

run_dir, step = Path(sys.argv[1]), int(sys.argv[2])
with np.load(run_dir/'data.npz') as d:
    A, B, Y = d['a'], d['b'], d['y']; masks = [d['train'], d['cal'], d['audit']]
tr = masks[0]; a, b, y = A[tr], B[tr], Y[tr]
with np.load(run_dir/f'checkpoint_{step:06d}.npz') as z:
    params = [z['U'].copy(), z['V'].copy(), z['W'].copy()]
p = params[0].shape[0]
Xa, Xb = ind.one_hot_inputs(a, b, p)
loss, grads = ind.loss_and_grad(params, Xa, Xb, y)
rng = np.random.default_rng(4242)
fd = []
for k in range(3):
    flat = params[k].reshape(-1); g = grads[k].reshape(-1)
    for i in rng.choice(flat.size, 40, replace=False):
        old = flat[i]; eps = 1e-6
        flat[i] = old + eps; lp = ind.loss_and_grad(params, Xa, Xb, y)[0]
        flat[i] = old - eps; lm = ind.loss_and_grad(params, Xa, Xb, y)[0]
        flat[i] = old
        fd.append(abs((lp - lm)/(2*eps) - g[i]))
scale = max(float(np.max(np.abs(np.concatenate([x.ravel() for x in grads])))), 1e-300)
shared_loss, shared = lossgrad(params, a, b, y)
lookup = forward(list(perfect_lookup(p)), A, B)[0]
print(json.dumps(dict(
    run=run_dir.name, step=step, train_loss=loss, grad_max_abs=scale,
    fd_coordinates=len(fd), fd_max_abs_error=float(max(fd)), fd_max_error_over_grad_max=float(max(fd)/scale),
    shared_loss_rel_diff=abs(shared_loss - loss)/loss,
    shared_grad_max_abs_diff_over_grad_max=max(float(np.max(np.abs(s - g)))/scale for s, g in zip(shared, grads)),
    shared_eval_lookup_accuracy=[accuracy(lookup, Y, ix) for ix in masks],
    shared_eval_lookup_margin_min=[objective_metrics(lookup, Y, ix)['margin_min'] for ix in masks])))
