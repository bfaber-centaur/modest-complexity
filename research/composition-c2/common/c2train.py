#!/usr/bin/env python3
"""Train or continue one c2 RNN run on length-3 programs and write its run record.

r1's RNN recipe by default: width 128, full-batch Adam (lr 1e-3, b1 .9, b2 .999, eps 1e-8)
with decoupled decay wd 0.1 on every parameter (biases, start and op embeddings included),
r1's init, sum of per-step mean cross-entropies with weights --step-weights (1 1 1 is r1's
`inter`, 0 0 1 its `final`). --transition/--readout choose the tying scheme (c2rnn.py).

--resume CKPT continues a run from a checkpoint with its weights and full Adam state
(moments and step count), to --updates total updates. The checkpoint's SHA-256 must equal
--resume-sha256 and its scheme must match. Continuation is checked bitwise against
uninterrupted training in test_c2.py; a weight-only restart is not this.

metrics.jsonl, one line per --eval-every updates (and the last): r1's keys (step, train, held,
held_final_pred) with identical values in the tied/tied scheme, then held_subset (final accuracy
on held-out rows whose answer is / is not a one-op-slot training neighbour's label, N / notN)
and, with --diag, optimization diagnostics of the update taken at that step: grad_norm and
update_norm (global L2), adam_sqrt_vhat (median, max, fraction below eps over all coordinates,
after the update) and the train final-step margin (min, median).

uv run --locked python research/composition-c2/common/c2train.py --out DIR --init-seed 51200
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'regimes'/'common')]
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import nets  # noqa: E402  (also puts the repository root on sys.path)
from experiment import array_sha256  # noqa: E402
from instrumentation import Timings  # noqa: E402

R1_HEADS = ('step1', 'step2', 'final')
B1, B2, EPS = .9, .999, 1e-8


def evaluate(p, data, nbr):
    """r1's metrics row, then the N / notN held-out final accuracy, and the train logits."""
    out, extra = {}, {}
    for name, d in data.items():
        logits = c2rnn.forward(p, d['ops'], s=d['s'])[1]
        out[name] = {h: c2rnn.ce_acc(logits[t], d['states'][:, t]) for t, h in enumerate(R1_HEADS)}
        extra[name] = logits
    pred = extra['held'][-1].argmax(1); ok = pred == data['held']['states'][:, -1]
    out['held_final_pred'] = ''.join(map(str, pred))
    out['held_subset'] = dict(N=float(ok[nbr].mean()), notN=float(ok[~nbr].mean()), n_N=int(nbr.sum()), n_notN=int((~nbr).sum()))
    return out, extra['train']


def diagnostics(p_before, p, g, st, train_logits, y):
    norm = lambda d: float(np.sqrt(sum(float(np.sum(v*v)) for v in d.values())))
    vhat = np.sqrt(np.concatenate([v.ravel() for v in st['v'].values()]) / (1 - B2**st['t']))
    m = c2rnn.margin(train_logits[-1], y)
    return dict(grad_norm=norm(g), update_norm=norm({k: p[k] - p_before[k] for k in p}),
                adam_sqrt_vhat=dict(median=float(np.median(vhat)), max=float(vhat.max()), frac_below_eps=float(np.mean(vhat < EPS))),
                train_final_margin=dict(min=float(m.min()), median=float(np.median(m))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--task-id', default=c2task.ORIGINAL)
    ap.add_argument('--transition', choices=('tied', 'untied'), default='tied')
    ap.add_argument('--readout', choices=('tied', 'untied'), default='tied')
    ap.add_argument('--step-weights', type=float, nargs=3, default=[1., 1., 1.])
    ap.add_argument('--init-seed', type=int, required=True)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--updates', type=int, default=10000, help='Total updates, counted from init (also on --resume)')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--eval-every', type=int, default=50)
    ap.add_argument('--diag', action='store_true', help='Log optimization diagnostics at every evaluation')
    ap.add_argument('--checkpoint-steps', type=int, nargs='*', default=[0], help='The final update is always saved')
    ap.add_argument('--resume', type=Path); ap.add_argument('--resume-sha256')
    ap.add_argument('--run-id'); ap.add_argument('--purpose', default='research')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    timing = Timings()

    T = c2task.load(args.task_id)
    data = dict(train=c2task.rows(T['P'], T['train']), held=c2task.rows(T['P'], T['held']))
    nbr, nbr_all_train = c2task.neighbour_answerable(T['P'], T['train'], T['held'])
    fit = data['train']
    w = args.step_weights
    targets = [fit['states'][:, t] for t in range(3)]
    tie = dict(transition=args.transition == 'tied', readout=args.readout == 'tied')
    if args.resume:
        sha = nets.sha256_file(args.resume)
        if sha != args.resume_sha256:
            ap.error(f'--resume {args.resume}: SHA-256 {sha} does not equal --resume-sha256 {args.resume_sha256}')
        p, st, src_meta = nets.load(args.resume)
        assert c2rnn.scheme(p) == tie, (c2rnn.scheme(p), tie)
        assert p['bs'].shape == (args.hidden,), 'width differs from the checkpoint'
        start, resumed = st['t'], dict(path=str(args.resume), sha256=sha, step=st['t'], meta=src_meta)
        assert start < args.updates
    else:
        p = c2rnn.init(args.hidden, args.init_seed, tie['transition'], tie['readout'])
        st, start, resumed = nets.adam_state(p), 0, None

    config = dict(
        vars(args), out=str(args.out), resume=str(args.resume) if args.resume else None,
        model='rnn', scheme=c2rnn.scheme(p), resumed_from=resumed,
        task=dict(id=T['id'], task_seed_used=T['task_seed_used'], ops=T['P'].tolist(), latin_square=T['L'].tolist(),
                  held_out_programs=T['held'], n_notN=int((~nbr).sum()), all_neighbours_are_training_rows=bool(nbr_all_train)),
        seeds_by_role=dict(task=c2task.parse_id(T['id'])[0], split=c2task.parse_id(T['id'])[1], init=args.init_seed),
        recipe=dict(optimizer='Adam, decoupled decay on every parameter', b1=B1, b2=B2, eps=EPS, full_batch=True,
                    loss='sum over steps of weight * mean cross-entropy', step_weights=w),
        exposure=dict(fit_rows=len(fit['s']), updates=args.updates, updates_this_run=args.updates - start,
                      row_presentations=len(fit['s'])*args.updates, labels_per_row=int(sum(x != 0 for x in w)),
                      parameters={k: int(v.size) for k, v in p.items()}, total_parameters=int(sum(v.size for v in p.values()))),
        sha256=dict(ops=array_sha256(T['P']), **{f'{k}_rows': array_sha256(d['s'], d['ops'], d['states']) for k, d in data.items()}),
        provenance=nets.provenance(args.out, HERE/'c2task.py', HERE/'c2rnn.py', __file__))
    (args.out/'config.json').write_text(json.dumps(config, indent=1, default=int))

    row = None
    with (args.out/'metrics.jsonl').open('w') as log:
        for t in range(start, args.updates+1):
            logged = t % args.eval_every == 0 or t == args.updates
            if logged:
                with timing.measure('evaluate'):
                    row, train_logits = evaluate(p, data, nbr)
                    row = dict(step=t, **row)
            if t in args.checkpoint_steps or t == args.updates:
                nets.save(args.out/f'ckpt-{t}.npz', p, st, step=t, run_id=args.run_id, scheme=c2rnn.scheme(p))
            if t == args.updates:
                if logged:
                    log.write(json.dumps(row)+'\n')
                break
            with timing.measure('update'):
                g = c2rnn.loss_grad(p, fit['s'], fit['ops'], targets, w)[2]
                before = {k: v.copy() for k, v in p.items()} if logged and args.diag else None
                nets.adam_step(p, g, st, args.lr, args.wd)
            if logged:
                if args.diag:
                    row['diag'] = diagnostics(before, p, g, st, train_logits, targets[2])
                log.write(json.dumps(row)+'\n')
    (args.out/'runtime.json').write_text(json.dumps(dict(status='complete', **timing.snapshot()), indent=1))


if __name__ == '__main__':
    main()
