#!/usr/bin/env python3
"""Train one R1 composition run (QUESTION.md) and write its run record.

Conditions:
  final   final-answer head only
  inter   final + step1 + step2 heads, true intermediate states
  shuf    final + step1 + step2 heads, (step1, step2) label pairs permuted across training rows
  full    positive control P1: inter's heads trained on all 512 rows (held-out rows are trained)
  interp  positive control P2: a (state, op) -> next-state net trained on the transitions the
          training rows expose, then applied three times with argmax feedback

Every run is evaluated on the 48 training programs and the 16 held-out programs
(all starts), against the true labels of every head, every --eval-every updates.

uv run --locked python research/regimes/composition/train.py --out DIR --condition inter --init-seed 51200
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'common'), str(HERE)]
import nets  # noqa: E402  (also puts the repository root on sys.path)
import task  # noqa: E402
from experiment import array_sha256  # noqa: E402
from instrumentation import Timings  # noqa: E402

STEPS = ('step1', 'step2', 'final')


def ce_acc(logits, y):
    z = logits - logits.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return dict(acc=float(np.mean(z.argmax(1) == y)), ce=float(np.mean(lse - z[np.arange(len(y)), y])))


def interp_rows(states, ops):
    X = np.zeros((len(states), task.N_SYM + task.N_OP))
    X[np.arange(len(states)), states] = 1; X[np.arange(len(states)), task.N_SYM + ops] = 1
    return X


def evaluate(p, cond, data):
    """{split: {head: {acc, ce}}} against true labels, plus held-out final predictions."""
    out, preds = {}, None
    for name, d in data.items():
        if cond == 'interp':  # iterate the one-step net; there are no logits for whole programs
            state, row = d['s'], {}
            for slot, head in zip('abc', STEPS):
                state = nets.forward(p, interp_rows(state, d[slot]))[1]['next'].argmax(1)
                row[head] = dict(acc=float(np.mean(state == d[head])))
            final_pred = state
        else:
            logits = nets.forward(p, task.encode(d))[1]
            row = {h: ce_acc(logits[h], d[h]) for h in STEPS if h in logits}
            final_pred = logits['final'].argmax(1)
        out[name] = row
        if name == 'held':
            preds = ''.join(map(str, final_pred))
    return out, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--condition', choices=('final', 'inter', 'shuf', 'full', 'interp'), required=True)
    ap.add_argument('--init-seed', type=int, required=True)
    ap.add_argument('--task-seed', type=int, default=task.TASK_SEED)
    ap.add_argument('--split-seed', type=int, default=task.SPLIT_SEED)
    ap.add_argument('--shuffle-seed', type=int, default=51300, help='Used by shuf only')
    ap.add_argument('--hidden', type=int, nargs='+', default=[128, 128])
    ap.add_argument('--updates', type=int, default=10000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--checkpoint-steps', type=int, nargs='*', default=[0], help='The final update is always saved')
    ap.add_argument('--run-id'); ap.add_argument('--purpose', default='research')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    timing = Timings()
    cond = args.condition

    P, task_seed_used = task.draw_ops(args.task_seed)
    L, train_programs, held_programs = task.split(args.split_seed)
    data = dict(train=task.rows(P, train_programs), held=task.rows(P, held_programs))
    fit = task.rows(P, task.PROGRAMS if cond == 'full' else train_programs)
    if cond == 'interp':
        trans = sorted({(int(x), int(o), int(y)) for before, slot, after in (('s', 'a', 'step1'), ('step1', 'b', 'step2'),
                        ('step2', 'c', 'final')) for x, o, y in zip(fit[before], fit[slot], fit[after])})
        trans = np.array(trans)
        X = interp_rows(trans[:, 0], trans[:, 1]); targets = dict(next=trans[:, 2]); heads = dict(next=task.N_SYM)
    else:
        X = task.encode(fit)
        heads = dict(final=task.N_SYM) if cond == 'final' else {h: task.N_SYM for h in STEPS}
        targets = {h: fit[h] for h in heads}
        if cond == 'shuf':
            perm = np.random.default_rng(args.shuffle_seed).permutation(len(X))
            targets['step1'], targets['step2'] = fit['step1'][perm], fit['step2'][perm]
    p = nets.init(X.shape[1], args.hidden, heads, args.init_seed); st = nets.adam_state(p)

    config = dict(
        vars(args), out=str(args.out), task_seed_used=task_seed_used,
        seeds_by_role=dict(task=task_seed_used, split=args.split_seed, init=args.init_seed,
                           label_shuffle=args.shuffle_seed if cond == 'shuf' else None),
        ops=P.tolist(), latin_square=L.tolist(), held_out_programs=held_programs,
        heldout_is_trained=cond == 'full',
        exposure=dict(fit_rows=len(X), updates=args.updates, row_presentations=len(X)*args.updates,
                      labels_per_row=len(targets), heads=heads,
                      parameters={k: int(v.size) for k, v in p.items()}, total_parameters=int(sum(v.size for v in p.values()))),
        sha256=dict(ops=array_sha256(P), fit_inputs=array_sha256(X), fit_targets=array_sha256(*targets.values()),
                    **{f'{k}_rows': array_sha256(*v.values()) for k, v in data.items()}),
        provenance=nets.provenance(args.out, HERE/'task.py', __file__))
    (args.out/'config.json').write_text(json.dumps(config, indent=1, default=int))

    with (args.out/'metrics.jsonl').open('w') as log:
        for t in range(args.updates+1):
            if t % args.eval_every == 0 or t == args.updates:
                with timing.measure('evaluate'):
                    row, preds = evaluate(p, cond, data)
                    if cond == 'shuf':  # the labels actually optimized, which evaluate() does not score
                        logits = nets.forward(p, X)[1]
                        row['fit_targets'] = {h: ce_acc(logits[h], targets[h]) for h in heads}
                log.write(json.dumps(dict(step=t, **row, held_final_pred=preds))+'\n')
            if t in args.checkpoint_steps or t == args.updates:
                nets.save(args.out/f'ckpt-{t}.npz', p, st, step=t, run_id=args.run_id, condition=cond)
            if t == args.updates:
                break
            with timing.measure('update'):
                nets.adam_step(p, nets.loss_grad(p, X, targets)[2], st, args.lr, args.wd)
    (args.out/'runtime.json').write_text(json.dumps(dict(status='complete', **timing.snapshot()), indent=1))


if __name__ == '__main__':
    main()
