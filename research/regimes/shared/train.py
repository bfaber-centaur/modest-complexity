#!/usr/bin/env python3
"""One R2 run: an arm (pretraining task or joint auxiliary) x one paired replicate.

transfer mode: optionally pretrain head 'pre' + trunk on the whole 512-input pool
  (--pre-updates full-batch Adam updates), then for each n start from that state,
  reset Adam, and train head 'new' + trunk on D_n = pool[:n] (--updates).
  From the same start, a frozen-feature readout trains only head 'new' on the
  last-hidden-layer features of D_n with the same updates and learning rate.
joint mode: for each n, from initialization, train heads 'pre' (--aux task)
  and 'new' with the trunk on D_n, both labeled on the same n inputs.
Every arm of a replicate shares the initialization (init seed) and split (data
seed); head names do not depend on the task, so their initial weights match too.

uv run --locked python research/regimes/shared/train.py --out OUT --pre y1 --new y2 \
    --init-seed 52200 --data-seed 52100
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'common')); sys.path.insert(0, str(HERE))
import nets  # noqa: E402
import task  # noqa: E402
from experiment import array_sha256  # noqa: E402
from instrumentation import Timings  # noqa: E402


def steps_grid(T, k=30):
    return sorted({0, T, *np.unique(np.round(np.logspace(0, np.log10(T), k)).astype(int)).tolist()})


def head_metrics(logits, y, idx):
    """Accuracy, balanced accuracy and mean cross-entropy (nats) on rows idx."""
    z = logits[idx]; yy = y[idx]
    zs = z - z.max(1, keepdims=True); lse = np.log(np.exp(zs).sum(1))
    pred = z.argmax(1)
    rec = [np.mean(pred[yy == c] == c) for c in (0, 1) if np.any(yy == c)]
    return dict(acc=float(np.mean(pred == yy)), bal_acc=float(np.mean(rec)),
                loss=float(np.mean(lse - zs[np.arange(len(yy)), yy])))


def evaluate(p, heads, train_idx, test_idx):
    """heads: {head name: task name}. y2 also gets test accuracy on each x4 half."""
    _, logits = nets.forward(p, task.X)
    out = {}
    for h, t in heads.items():
        y = task.LABELS[t]
        for part, idx in (('train', train_idx), ('test', test_idx)):
            for k, v in head_metrics(logits[h], y, idx).items():
                out[f'{h}_{part}_{k}'] = v
        if t == 'y2':
            x4 = task.B[test_idx, 3]
            out[f'{h}_test_acc_x4is1'] = head_metrics(logits[h], y, test_idx[x4 == 1])['acc']
            out[f'{h}_test_acc_x4is0'] = head_metrics(logits[h], y, test_idx[x4 == 0])['acc']
    return out


def flip_sensitivity(p, head):
    """For each input bit, the fraction of all 1,024 inputs whose predicted class
    flips when that bit alone flips."""
    pred = nets.forward(p, task.X)[1][head].argmax(1)
    idx = np.arange(len(task.X))
    return [float(np.mean(nets.forward(p, task.X[idx ^ (1 << k)])[1][head].argmax(1) != pred))
            for k in range(task.N_BITS)]


def train(p, heads, watch, idx, test_idx, updates, lr, trainable, log):
    """Full-batch Adam on rows idx from a fresh optimizer state (the phase-boundary
    reset), training `heads` and logging metrics of every head in `watch`."""
    st = nets.adam_state(p)
    targets = {h: task.LABELS[t][idx] for h, t in heads.items()}
    X = task.X[idx]; grid = set(steps_grid(updates))
    for step in range(updates+1):
        if step in grid:
            log(dict(step=step, **evaluate(p, watch, idx, test_idx)))
        if step < updates:
            nets.adam_step(p, nets.loss_grad(p, X, targets)[2], st, lr, trainable=trainable)
    return st


def readout(p, new, n_idx, test_idx, updates, lr, log):
    """Frozen-feature readout: train only head 'new' on fixed last-layer features,
    from the same initial head weights as the fine-tuning branch."""
    H = nets.forward(p, task.X)[0][-1]
    q = {'W_new': p['W_new'].copy(), 'b_new': p['b_new'].copy()}
    st = nets.adam_state(q); y = task.LABELS[new]
    grid = set(steps_grid(updates))
    for step in range(updates+1):
        if step in grid:
            logits = {'new': H @ q['W_new'] + q['b_new']}
            row = dict(step=step)
            for part, idx in (('train', n_idx), ('test', test_idx)):
                row.update({f'new_{part}_{k}': v for k, v in head_metrics(logits['new'], y, idx).items()})
            if new == 'y2':
                x4 = task.B[test_idx, 3]
                row['new_test_acc_x4is1'] = head_metrics(logits['new'], y, test_idx[x4 == 1])['acc']
            log(row)
        if step < updates:
            nets.adam_step(q, nets.loss_grad(q, H[n_idx], {'new': y[n_idx]})[2], st, lr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=('transfer', 'joint'), default='transfer')
    ap.add_argument('--pre', choices=('none', *task.LABELS), default='none', help='transfer: pretraining task')
    ap.add_argument('--aux', choices=tuple(task.LABELS), help='joint: auxiliary task on the same D_n')
    ap.add_argument('--new', choices=tuple(task.LABELS), default='y2')
    ap.add_argument('--init-seed', type=int, required=True)
    ap.add_argument('--data-seed', type=int, required=True)
    ap.add_argument('--hidden', type=int, nargs='+', default=[64, 64])
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--pre-updates', type=int, default=2000)
    ap.add_argument('--updates', type=int, default=2000)
    ap.add_argument('--ns', type=int, nargs='+', default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument('--save-ns', type=int, nargs='*', default=[64, 512])
    args = ap.parse_args()
    if (args.mode == 'joint') != (args.aux is not None) or (args.mode == 'joint' and args.pre != 'none'):
        ap.error('joint mode takes --aux and no --pre; transfer mode takes no --aux')
    args.out.mkdir(parents=True, exist_ok=False)
    timing = Timings()
    pool, test = task.split(args.data_seed)
    p0 = nets.init(task.N_BITS, args.hidden, {'pre': 2, 'new': 2}, args.init_seed)
    trunk = [k for k in p0 if k[0] in 'Wb' and k[1:].isdigit()]
    params = dict(trunk=sum(p0[k].size for k in trunk), head=p0['W_new'].size + p0['b_new'].size)
    config = dict(vars(args), out=str(args.out), seeds=dict(init=args.init_seed, data=args.data_seed),
                  optimizer='Adam b1=.9 b2=.999 eps=1e-8, wd=0, full batch, state reset at every phase start',
                  parameters=params,
                  split=dict(pool=len(pool), test=len(test), pool_sha256=array_sha256(pool),
                             test_sha256=array_sha256(test)),
                  coverage={str(n): {t: task.coverage(pool[:n], t) for t in task.RELEVANT} for n in args.ns},
                  positives={str(n): int(task.LABELS[args.new][pool[:n]].sum()) for n in args.ns},
                  labels_sha256={t: array_sha256(y) for t, y in task.LABELS.items()},
                  provenance=nets.provenance(args.out, __file__, HERE/'task.py'))
    (args.out/'config.json').write_text(json.dumps(config, indent=1))
    stream = (args.out/'metrics.jsonl').open('w')

    def log(row):
        stream.write(json.dumps(row)+'\n'); stream.flush()

    summary = dict(per_n={})
    start = p0
    if args.pre != 'none':
        start = {k: v.copy() for k, v in p0.items()}
        with timing.measure('pretrain'):
            st = train(start, {'pre': args.pre}, {'pre': args.pre}, pool, test, args.pre_updates, args.lr,
                       trunk+['W_pre', 'b_pre'], lambda r: log(dict(r, tag='pre')))
        nets.save(args.out/'pretrained.npz', start, st, task=args.pre, updates=args.pre_updates)
        summary['pretrained'] = dict(sha256=nets.sha256_file(args.out/'pretrained.npz'),
                                     pre_flip=flip_sensitivity(start, 'pre'),
                                     **evaluate(start, {'pre': args.pre}, pool, test))
    for n in args.ns:
        idx = pool[:n]
        p = {k: v.copy() for k, v in start.items()}
        heads = {'new': args.new} if args.mode == 'transfer' else {'pre': args.aux, 'new': args.new}
        watch = dict(heads, **({'pre': args.pre} if args.pre != 'none' else {}))
        trainable = trunk + [f'{x}_{h}' for h in heads for x in 'Wb']
        with timing.measure('train'):
            # In transfer mode the untrained 'pre' head is watched to track forgetting.
            train(p, heads, watch, idx, test, args.updates, args.lr, trainable,
                  lambda r: log(dict(r, tag='train', n=n)))
        row = dict(new_flip=flip_sensitivity(p, 'new'), **evaluate(p, watch, idx, test))
        if 'pre' in watch:
            row['pre_flip'] = flip_sensitivity(p, 'pre')
        if n in args.save_ns:
            np.savez_compressed(args.out/f'final_n{n}.npz', **p)
        if args.mode == 'transfer':
            with timing.measure('readout'):
                readout(start, args.new, idx, test, args.updates, args.lr, lambda r: log(dict(r, tag='readout', n=n)))
        summary['per_n'][str(n)] = row
    stream.close()
    (args.out/'summary.json').write_text(json.dumps(summary, indent=1))
    (args.out/'runtime.json').write_text(json.dumps(dict(status='complete', **timing.snapshot()), indent=1))


if __name__ == '__main__':
    main()
