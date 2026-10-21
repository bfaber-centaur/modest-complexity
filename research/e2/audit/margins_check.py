#!/usr/bin/env python3
"""Independent check of dynamics packet 2 (research/e2/dynamics/README.md) and the scale and
set-size questions of QUESTION-4.md, written from the README's definitions.

On the 483 common pairs, for pair (a, b) with correct label c = a+b:
- L30, L50: labels of training pairs of the 30% / 50% split that contain a or b, in either
  position. S = L30, D = L50 - L30, N = every other label except c (c is in none: asserted);
  a run's own set: the same with its own training pairs.
- margin over X = logit(c) - max over X; gap S - N = max over S - max over N; "beats X" =
  margin over X > 0; partial = beats every label outside the own set but is not correct.
- sigma = standard deviation of the pair's 30 wrong-label logits; "/sigma" columns are
  medians of the per-pair ratios.
- set-size null for the gap: the pair's 30 wrong-label logits assigned at random to sets
  of sizes |S|, |D|, |N| (200 draws per pair, seed 36200); expected max S - max N.
- post hoc (added after the sigma columns were seen, because sigma includes the S - N
  offset itself): "within_sigma" = pooled standard deviation of the wrong-label logits
  around their own set's mean (S, D and N each centred).

Rows: every published checkpoint of the 12 width-128 runs (run directories and the extra
e1 checkpoints of D9), or the ridge memorizer (QUESTION-4.md, D): step-0 U, V of each init
seed with W fitted in closed form to one-hot a+b minus 1/p on each share's training pairs.

uv run --locked python research/e2/audit/margins_check.py OUT.csv TRANSITIONS.csv
uv run --locked python research/e2/audit/margins_check.py --ridge OUT.csv
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
P, DRAWS = 31, 200
COMMON = np.load(root/'research/e2/data/common-never-trained-30-50.npz')['index']
C, ST, E2 = root/'research/coordinator/runs', root/'research/size-time/runs', root/'research/e2'
RUNS = {'30%': [C/'e1-co-add-w128-i30000-a1', ST/'e1-st-add-w128-i30001-a1', ST/'e1-st-add-w128-i30002-a1'],
        '40%': [E2/f'dynamics/runs/e2-dy-add-w128-i{s}-tf40-a1' for s in (30000, 30001, 30002)],
        '45%': [E2/f'dynamics/runs/e2-dy-add-w128-i{s}-tf45-a1' for s in (30000, 30001, 30002)],
        '50%': [C/f'e1f2-co-add-w128-i{s}-tf50-a1' for s in (30000, 30001, 30002)]}
EXTRA = [E2/'coordinator/e1-checkpoints', E2/'dynamics/e1-checkpoints']


def operand_labels(train, a, b):
    """(p, p) boolean: label l occurs on a training pair containing operand x, either position."""
    m = np.zeros((P, P), bool)
    m[a[train], (a[train] + b[train]) % P] = True
    m[b[train], (a[train] + b[train]) % P] = True
    return m


def frozen(name):
    with np.load(root/'research/data'/name) as z:
        return z['a'], z['b'], z['train']


a, b, tr30 = frozen('e1-split.npz')
_, _, tr50 = frozen('e1f2-split-tf50.npz')
ca, cb = a[COMMON], b[COMMON]
y = (ca + cb) % P
op30, op50 = operand_labels(tr30, a, b), operand_labels(tr50, a, b)
S = op30[ca] | op30[cb]
D = (op50[ca] | op50[cb]) & ~S
N = ~(S | D)
N[np.arange(len(y)), y] = False
assert not (S | D)[np.arange(len(y)), y].any()  # a+b never co-occurs with a or b in training
WRONG = ~np.eye(P, dtype=bool)[y]
assert np.array_equal(S | D | N, WRONG)


def set_size_null(wrong, rng):
    """Expected max over |S| minus max over |N| when the wrong-label logits are relabelled at random."""
    nS, nD = S.sum(1), D.sum(1)
    perm = rng.permuted(np.broadcast_to(wrong, (DRAWS,) + wrong.shape), axis=2)  # (draws, pairs, 30)
    pos = np.arange(P - 1)
    inS = pos < nS[:, None]; inN = pos >= (nS + nD)[:, None]
    mx = lambda m: np.where(m, perm, -np.inf).max(2)
    return (mx(inS) - mx(inN)).mean(0)


def measures(U, V, W, own):
    z = np.maximum(U[ca] + V[cb], 0) @ W
    zc = z[np.arange(len(y)), y]
    mx = lambda m: np.where(m, z, -np.inf).max(1)
    wrong = z[WRONG].reshape(len(y), P - 1)  # row order: labels ascending, c removed
    sigma = wrong.std(1)
    cen = sum(np.where(m, (z - np.where(m, z, 0).sum(1, keepdims=True)/m.sum(1, keepdims=True))**2, 0).sum(1) for m in (S, D, N))
    within = np.sqrt(cen/(P - 1 - 3))  # every pair has nonempty S, D and N
    mS, mD, mN, mOwn, mOut = (zc - mx(m) for m in (S, D, N, own, WRONG & ~own))
    correct = z.argmax(1) == y
    null = set_size_null(wrong, np.random.default_rng(36200))
    gSN, gDN = mx(S) - mx(N), mx(D) - mx(N)
    med = lambda v: float(np.median(v))
    row = dict(common_acc=float(correct.mean()), beats_S=float((mS > 0).mean()), beats_D=float((mD > 0).mean()),
               beats_N=float((mN > 0).mean()), beats_own_out=float((mOut > 0).mean()),
               partial=float(((mOut > 0) & ~correct).mean()),
               median_margin_S=med(mS), median_margin_D=med(mD), median_margin_N=med(mN),
               median_gap_S_over_N=med(gSN), median_gap_D_over_N=med(gDN),
               identity_max_abs=float(np.abs(mS - (mN - gSN)).max()),
               median_sigma=med(sigma), sigma_margin_N=med(mN/sigma), sigma_margin_S=med(mS/sigma),
               sigma_gap_S_over_N=med(gSN/sigma), sigma_null_gap=med(null/sigma), median_null_gap=med(null),
               sigma_excess_gap=med((gSN - null)/sigma), median_within_sigma=med(within),
               within_sigma_margin_N=med(mN/within), within_sigma_gap_S_over_N=med(gSN/within))
    return row, correct, mOut > 0


def checkpoints(run):
    found = {int(f.stem[11:]): f for f in run.glob('checkpoint_*.npz')}
    for d in EXTRA:
        found.update({int(f.stem[11:]): f for f in (d/run.name).glob('checkpoint_*.npz')})
    return dict(sorted(found.items()))


def trained(out_csv, trans_csv):
    rows, trans = [], []
    for share, runs in RUNS.items():
        for run in runs:
            with np.load(run/'data.npz') as z:
                own_op = operand_labels(z['train'], z['a'], z['b'])
                assert not set(COMMON.tolist()) & set(z['train'].tolist())
            own = own_op[ca] | own_op[cb]
            assert not (S & ~own).any() and not (own & ~(S | D)).any()  # S <= own <= S u D
            ce = {r['step']: r['objective']['train']['cross_entropy']
                  for r in map(json.loads, (run/'metrics.jsonl').read_text().splitlines())}
            prev = None
            for step, f in checkpoints(run).items():
                with np.load(f) as w:
                    assert int(w['step']) == step
                    row, correct, out = measures(w['U'], w['V'], w['W'], own)
                rows.append(dict(run_id=run.name, share=share, step=step, train_ce=ce[step], own_set_mean=float(own.sum(1).mean()), **row))
                if prev is not None:
                    pstep, pc, pout = prev; new = ~pc & correct
                    trans.append(dict(run_id=run.name, share=share, from_step=pstep, to_step=step, wrong_at_from=int((~pc).sum()),
                                      wrong_and_beats_out=int((~pc & pout).sum()), newly_correct=int(new.sum()),
                                      newly_correct_from_beats_out=int((new & pout).sum())))
                prev = step, correct, out
    write(out_csv, rows); write(trans_csv, trans)


def ridge(out_csv):
    """QUESTION-4.md, D: a training-free memorizer at each nested share."""
    sys.path.insert(0, str(root/'research/audit'))
    import independent as ind
    rows = []
    for frac in (.3, .4, .45, .5):
        _, _, _, _, (tr, _, _) = ind.regenerate_split(P, 31000, 32000, frac)
        own_op = operand_labels(tr, a, b); own = own_op[ca] | own_op[cb]
        for seed, run in zip((30000, 30001, 30002), RUNS['50%']):
            U, V, _ = ind.initial_weights(P, 128, seed)
            with np.load(run/'checkpoint_000000.npz') as w:  # the published step 0 of this init seed
                assert np.array_equal(U, w['U']) and np.array_equal(V, w['V'])
            H = np.maximum(U[a[tr]] + V[b[tr]], 0)
            Y = np.eye(P)[(a[tr] + b[tr]) % P] - 1/P
            G = H.T @ H
            for r in (1e-6, 1e-2):
                W = np.linalg.solve(G + r*np.trace(G)/len(G)*np.eye(len(G)), H.T @ Y)
                train_acc = float(np.mean((H @ W).argmax(1) == (a[tr] + b[tr]) % P))
                row, _, _ = measures(U, V, W, own)
                rows.append(dict(run_id=f'ridge-{r:g}-i{seed}', share=f'{round(100*frac)}%', ridge=r, init_seed=seed,
                                 train_pairs=len(tr), train_acc=train_acc, **row))
    write(out_csv, rows)


def write(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader()
        w.writerows({k: (float(f'{v:.10g}') if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


if __name__ == '__main__':
    if sys.argv[1] == '--ridge':
        ridge(sys.argv[2])
    else:
        trained(sys.argv[1], sys.argv[2])
