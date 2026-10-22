#!/usr/bin/env python3
"""R3 training history and recovery. One run is a sequence of phases on the
shared network (../common/nets.py), started from an init seed or from a saved
checkpoint. Every phase starts with a fresh Adam state and trains only the trunk
and the active task's head. Design, arms and criteria: QUESTION.md.

uv run --locked python research/regimes/history/history.py --out DIR --init-seed 53200 --phases A:2000
uv run --locked python research/regimes/history/history.py --out DIR --from CKPT --from-sha256 HEX --phases Brel:1000,A:4000
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'common'))
import nets  # noqa: E402  (puts the repository root on sys.path)
from experiment import array_sha256  # noqa: E402
from instrumentation import Timings  # noqa: E402

N_BITS, HIDDEN, N_TRAIN = 10, 128, 256
DATA_SEED, READOUT_SEED, NULL_SEED = 53100, 53300, 53301
LR, WD = 1e-3, 0.
READOUT_LR, READOUT_CHECKS = 1e-2, (200, 2000)
TRUNK = ('W1', 'b1')
# task: (head, input bits whose parity is the label, flip)
TASKS = {'A': ('A', (0, 1, 2), 0), 'Brel': ('B', (0, 1, 3), 0),
         'Bunrel': ('B', (4, 5, 6), 0), 'notA': ('A', (0, 1, 2), 1)}
# Logged (head, task) pairs. notA accuracy through head A is 1 - A accuracy.
EVALS = (('A', 'A'), ('B', 'Brel'), ('B', 'Bunrel'))


def domain():
    """All 2^10 inputs; row i has bit k = (i >> k) & 1."""
    i = np.arange(2**N_BITS)
    return ((i[:, None] >> np.arange(N_BITS)) & 1).astype(float)


def labels(X, task):
    _, bits, flip = TASKS[task]
    return (X[:, list(bits)].sum(1).astype(int) % 2) ^ flip


def split():
    rng = np.random.default_rng(DATA_SEED)
    train = np.sort(rng.choice(2**N_BITS, N_TRAIN, replace=False))
    return train, np.setdiff1d(np.arange(2**N_BITS), train)


def split_stats(X, train, test):
    """Label balance and 3-bit pattern coverage of each task in each part."""
    out = {}
    for task, (_, bits, _) in TASKS.items():
        y = labels(X, task); code = X[:, list(bits)] @ (1 << np.arange(3))
        out[task] = {part: dict(n=len(ix), positive=int(y[ix].sum()),
                                pattern_counts=np.bincount(code[ix].astype(int), minlength=8).tolist())
                     for part, ix in (('train', train), ('test', test))}
    return out


def ce_acc(z, y):
    zs = z - z.max(1, keepdims=True)
    ce = np.log(np.exp(zs).sum(1)) - zs[np.arange(len(y)), y]
    return float((z.argmax(1) == y).mean()), float(ce.mean())


def trunk_vector(p):
    return np.concatenate([p[k].ravel() for k in TRUNK])


def evaluate(p, X, Y, train, test, p0):
    _, logits = nets.forward(p, X)
    row = {}
    for head, task in EVALS:
        for part, ix in (('train', train), ('test', test)):
            row[f'{task}_{part}_acc'], row[f'{task}_{part}_loss'] = ce_acc(logits[head][ix], Y[task][ix])
    t0 = trunk_vector(p0)
    row['trunk_rel_drift'] = float(np.linalg.norm(trunk_vector(p)-t0)/np.linalg.norm(t0))
    return row


def readout(H, y, train, test, permute=False):
    """Fresh linear softmax readout on frozen, train-standardized features.
    permute=True is the row-shuffled null: training labels permuted across rows;
    its train accuracy is against the permuted labels (memorization), its test
    accuracy against the true labels."""
    mu, sd = H[train].mean(0), H[train].std(0)
    live = sd > 1e-12
    F = np.where(live, (H-mu)/np.where(live, sd, 1.), 0.)
    ytr = np.random.default_rng(NULL_SEED).permutation(y[train]) if permute else y[train]
    r = nets.init(H.shape[1], (), {'R': 2}, READOUT_SEED); st = nets.adam_state(r)
    Ftr, out = F[train], dict(dead_units=int((~live).sum()))
    for u in range(1, max(READOUT_CHECKS)+1):
        nets.adam_step(r, nets.loss_grad(r, Ftr, {'R': ytr})[2], st, READOUT_LR)
        if u in READOUT_CHECKS:
            pred = nets.forward(r, F)[1]['R'].argmax(1)
            out[str(u)] = dict(train_acc=float((pred[train] == ytr).mean()),
                               test_acc=float((pred[test] == y[test]).mean()))
    return out


def linear_cka(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    denom = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    return float(np.linalg.norm(b.T @ a)**2 / denom) if denom > 0 else None


def change(p0, p, X):
    """Functional change over the full domain from p0 to p."""
    a0, l0 = nets.forward(p0, X); a1, l1 = nets.forward(p, X)
    out = dict(hidden_linear_cka=linear_cka(a0[1], a1[1]),
               trunk_rel_drift=float(np.linalg.norm(trunk_vector(p)-trunk_vector(p0))/np.linalg.norm(trunk_vector(p0))))
    for h in ('A', 'B'):
        m0 = l0[h][:, 1]-l0[h][:, 0]; m1 = l1[h][:, 1]-l1[h][:, 0]
        out[h] = dict(pred_changed=float((l0[h].argmax(1) != l1[h].argmax(1)).mean()),
                      margin_rms_change=float(np.sqrt(np.mean((m1-m0)**2))),
                      margin_rms_before=float(np.sqrt(np.mean(m0**2))))
    return out


def on_grid(s):
    return s <= 100 or (s <= 1000 and s % 10 == 0) or s % 100 == 0


def parse_phases(text):
    phases = []
    for item in text.split(','):
        task, steps = item.split(':')
        if task not in TASKS or int(steps) < 1:
            raise ValueError(f'bad phase {item!r}')
        phases.append((task, int(steps)))
    return phases


def run(out, phases, init_seed=None, source=None, source_sha256=None, snapshot_at=()):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    timing = Timings()
    X = domain(); train, test = split()
    Y = {task: labels(X, task) for task in TASKS}
    if source is None:
        p = nets.init(N_BITS, (HIDDEN,), {'A': 2, 'B': 2}, init_seed); source_meta = None
    else:
        found = nets.sha256_file(source)
        if source_sha256 and found != source_sha256:
            raise ValueError(f'{source} has SHA-256 {found}, expected {source_sha256}')
        p, _, source_meta = nets.load(source)  # weights only; the optimizer is reset below
        if array_sha256(*(p[k] for k in sorted(p))) != source_meta['params_sha256']:
            raise ValueError('loaded weights do not match the hash recorded at save time')
    config = dict(phases=phases, init_seed=init_seed, source=str(source) if source else None,
                  source_file_sha256=nets.sha256_file(source) if source else None, source_meta=source_meta,
                  seeds=dict(task=None, data_split=DATA_SEED, init=init_seed, readout_init=READOUT_SEED,
                             readout_null_permutation=NULL_SEED),
                  n_bits=N_BITS, hidden=HIDDEN, n_train=N_TRAIN, lr=LR, wd=WD, optimizer='adam, reset at every phase start',
                  trainable='W1, b1 and the active task head', tasks={k: list(v[1]) + [v[2]] for k, v in TASKS.items()},
                  task_columns='bits..., flip', readout=dict(lr=READOUT_LR, checks=READOUT_CHECKS),
                  criterion='held-out A accuracy >= 0.95 (analysis)', snapshot_at=sorted(snapshot_at),
                  data_sha256=dict(X=array_sha256(X), train=array_sha256(train), test=array_sha256(test),
                                   **{f'y_{t}': array_sha256(y) for t, y in Y.items()}),
                  split_stats=split_stats(X, train, test), provenance=nets.provenance(out, __file__))
    (out/'config.json').write_text(json.dumps(config, indent=1))
    p_start = {k: v.copy() for k, v in p.items()}
    snapshots, last_end = [], p_start
    metrics = (out/'metrics.jsonl').open('w')

    def snapshot(label, k, s, t, state):
        # The saved optimizer state is the phase's actual state (fresh at the start).
        name = f'ckpt_t{t}.npz'
        nets.save(out/name, p, state, params_sha256=array_sha256(*(p[q] for q in sorted(p))),
                  run=out.name, t=t, phase=k, step=s)
        H = nets.forward(p, X)[0][1]
        snapshots.append(dict(label=label, t=t, phase=k, step=s, ckpt=name, ckpt_sha256=nets.sha256_file(out/name),
                              metrics=evaluate(p, X, Y, train, test, p_start),
                              readout=readout(H, Y['A'], train, test),
                              readout_null=readout(H, Y['A'], train, test, permute=True),
                              change_vs_start=change(p_start, p, X), change_vs_prev_phase_end=change(last_end, p, X)))

    t, updates = 0, 0
    with timing.measure('snapshots'):
        snapshot('start', 0, 0, 0, nets.adam_state(p))
    for k, (task, steps) in enumerate(phases, 1):
        head = TASKS[task][0]
        trainable = [*TRUNK, f'W_{head}', f'b_{head}']
        frozen = {q: v.copy() for q, v in p.items() if q not in trainable}
        state = nets.adam_state(p)  # reset at every phase boundary, in every arm
        Xtr, targets = X[train], {head: Y[task][train]}
        metrics.write(json.dumps(dict(phase=k, task=task, step=0, t=t, **evaluate(p, X, Y, train, test, p_start)))+'\n')
        for s in range(1, steps+1):
            with timing.measure('updates'):
                nets.adam_step(p, nets.loss_grad(p, Xtr, targets)[2], state, LR, WD, trainable)
            t += 1; updates += 1
            if on_grid(s) or s == steps:
                with timing.measure('evaluation'):
                    metrics.write(json.dumps(dict(phase=k, task=task, step=s, t=t,
                                                  **evaluate(p, X, Y, train, test, p_start)))+'\n')
            if t in snapshot_at and s != steps:
                with timing.measure('snapshots'):
                    snapshot('snapshot_at', k, s, t, state)
        # With wd = 0 the inactive head and anything else outside `trainable` must be untouched.
        assert all(np.array_equal(p[q], v) for q, v in frozen.items()), 'a frozen parameter changed'
        with timing.measure('snapshots'):
            snapshot('phase_end', k, steps, t, state)
        last_end = {q: v.copy() for q, v in p.items()}
    metrics.close()
    (out/'summary.json').write_text(json.dumps(dict(snapshots=snapshots), indent=1))
    snap = timing.snapshot()
    upd = snap['phases'].get('updates', dict(wall_seconds=0.))
    (out/'runtime.json').write_text(json.dumps(dict(
        status='complete', updates=updates, train_rows=N_TRAIN,
        updates_per_second=round(updates/upd['wall_seconds'], 1) if upd['wall_seconds'] else None,
        finished=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **snap), indent=1))
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phases', type=parse_phases, required=True, help='task:steps[,task:steps...]')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--init-seed', type=int)
    src.add_argument('--from', dest='source', type=Path)
    ap.add_argument('--from-sha256', help='Expected SHA-256 of the --from file')
    ap.add_argument('--snapshot-at', type=lambda s: {int(v) for v in s.split(',')}, default=set(),
                    help='Extra run steps that get a checkpoint, readout and functional change')
    args = ap.parse_args()
    run(args.out, args.phases, args.init_seed, args.source, args.from_sha256, args.snapshot_at)


if __name__ == '__main__':
    main()
