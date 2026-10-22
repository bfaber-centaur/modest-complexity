#!/usr/bin/env python3
"""Train one R4 condition: full-batch Adam on 256 training inputs, scoring the
whole 1,024-input domain at roughly log-spaced steps (task.metrics).

Writes config.json (every setting, seeds by role, data hashes, provenance),
metrics.jsonl, final.npz (weights + Adam state), stepN.npz for each --save-at
step, and runtime.json. The numerics diagnostics in metrics.jsonl were added for
the follow-up; they are read-only, so training arithmetic is unchanged.

uv run --locked python research/regimes/routes/train.py --out DIR --run-id ID \
    --condition rho7of8 --init-seed 54200 --updates 20000
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent/'common'))
import task  # noqa: E402
import nets  # noqa: E402
from experiment import array_sha256  # noqa: E402
from instrumentation import Timings  # noqa: E402


def log_steps(updates, n=100):
    return sorted({0, *np.unique(np.round(np.geomspace(1, updates, n)).astype(int)).tolist()})


def diagnostics(p, state, z_train, target, step, prev):
    """Read-only numerics checks for long training (added for the follow-up;
    nothing here touches p or state). A training row is 'saturated' when float64
    rounds its target probability to exactly 1, which zeroes the gradient on the
    target logit (margin above about 36.7). sqrt(v_hat) is compared with Adam's
    eps = 1e-8: below it, steps shrink in proportion."""
    margin = (z_train[:, 1] - z_train[:, 0]) * (2*target - 1)
    prob = nets.softmax(z_train)[np.arange(len(target)), target]
    t = state['t']
    out = dict(param_norm=float(np.sqrt(sum((v**2).sum() for v in p.values()))),
               train_margin_min=float(margin.min()), train_margin_median=float(np.median(margin)),
               train_frac_saturated=float((prob == 1.0).mean()),
               mean_update_norm=None, adam_sqrt_vhat_median=None)
    if step > prev[0]:
        out['mean_update_norm'] = float(np.sqrt(sum(((p[k] - prev[1][k])**2).sum() for k in p))) / (step - prev[0])
    if t > 0:
        out['adam_sqrt_vhat_median'] = float(np.median(np.concatenate(
            [np.sqrt(v / (1 - .999**t)).ravel() for v in state['v'].values()])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--condition', choices=task.CONDITIONS, required=True)
    ap.add_argument('--init-seed', type=int, required=True)
    ap.add_argument('--updates', type=int, required=True)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--purpose', default='exploratory', choices=('exploratory', 'calibration', 'follow-up'))
    ap.add_argument('--log-points', type=int, default=100)
    ap.add_argument('--save-at', type=int, nargs='*', default=[], help='extra checkpoint steps (stepN.npz)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    timing = Timings()

    d = task.domain(); ev = task.evaluation_sets(); train = task.training_set(args.condition)
    X, target = d['X'][train['idx']], train['target']
    p = nets.init(10, (args.width,), {'y': 2}, args.init_seed); state = nets.adam_state(p)
    config = dict(
        run_id=args.run_id, purpose=args.purpose, condition=args.condition,
        exceptions_per_pattern=task.EXCEPTIONS.get(args.condition, task.EXCEPTIONS['rho1of2']),
        realized_reliability=float((d['s'][train['idx']] == d['y'][train['idx']]).mean()),
        target='s' if args.condition == 'shortcut_only' else 'parity of inputs 0-2',
        seeds=dict(task=None, split=task.SPLIT_SEED, exceptions=task.EXCEPTION_SEED, init=args.init_seed,
                   note='task is fixed parity; full batch, so no minibatch-order seed'),
        model=dict(inputs=10, hidden=[args.width], heads={'y': 2}, encoding='+-1',
                   parameters=int(sum(v.size for v in p.values()))),
        optimizer=dict(name='adam', lr=args.lr, wd=0., betas=[.9, .999], eps=1e-8, batch='full (256)'),
        updates=args.updates, log_steps=log_steps(args.updates, args.log_points), save_at=args.save_at,
        data=dict(train_rows=len(target), train_label_mean=float(target.mean()),
                  train_s_mean=float(d['s'][train['idx']].mean()), exceptions=int(train['exception'].sum()),
                  train_sha256=array_sha256(X, target), domain_sha256=array_sha256(d['X'], d['y']),
                  heldout_mask_sha256=array_sha256(ev['held'])),
        provenance=nets.provenance(args.out, __file__, HERE/'task.py'))
    (args.out/'config.json').write_text(json.dumps(config, indent=1))

    steps = set(config['log_steps'])
    prev = (0, {k: v.copy() for k, v in p.items()})
    with (args.out/'metrics.jsonl').open('w') as stream:
        for step in range(args.updates + 1):
            if step in steps:
                with timing.measure('evaluate'):
                    z = nets.forward(p, d['X'])[1]['y']
                    loss = nets.loss_grad(p, X, {'y': target})[0]
                    row = dict(step=step, train_loss=loss, **task.metrics(z[:, 1] - z[:, 0], train, ev, d['y']),
                               **diagnostics(p, state, z[train['idx']], target, step, prev))
                    prev = (step, {k: v.copy() for k, v in p.items()})
                stream.write(json.dumps(row) + '\n'); stream.flush()
            if step in args.save_at:
                nets.save(args.out/f'step{step}.npz', p, state, run_id=args.run_id, step=step)
            if step == args.updates:
                break
            with timing.measure('train'):
                nets.adam_step(p, nets.loss_grad(p, X, {'y': target})[2], state, args.lr)
    nets.save(args.out/'final.npz', p, state, run_id=args.run_id, step=args.updates)
    (args.out/'runtime.json').write_text(json.dumps(dict(status='complete', **timing.snapshot()), indent=1))


if __name__ == '__main__':
    main()
