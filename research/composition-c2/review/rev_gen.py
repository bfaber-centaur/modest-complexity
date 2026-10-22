"""Generalization lane checks (QUESTION.md 2.2 and Revisions 04:14; REVIEW.md section 3).
Part A: task facts and the frozen length 1-6 evaluation of the six recovered r1 tied/tied
checkpoints, recomputed with the oracle and forward pass of rev_pin.py (not the lane's code),
against the lane's published CSVs at a lane commit; then counterexample CX-G1.

uv run --locked python research/composition-c2/review/rev_gen.py LANE_COMMIT
"""
import csv
import io
from itertools import product
import json
import subprocess
import sys
import numpy as np

from rev_pin import C2, FAIL, HELD, P, ROOT, TRAIN, check, my_forward, oracle_rows
import nets

RUNS = [f'r1-comp-rnn-{c}-s{s}-a1' for c in ('final', 'inter') for s in (51200, 51201, 51202)]
LANE = 'research/composition-c2/generalization'


def lane_csv(commit, path):
    text = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{LANE}/{path}'], text=True)
    return list(csv.DictReader(io.StringIO(text)))


def compose(q):
    f = tuple(range(8))
    for k in q:
        f = tuple(P[k][x] for x in f)
    return f


def notn_mask(he):
    """True where no one-op-slot training neighbour (same start) has the row's final label."""
    lab = {(s, *q): y for q in TRAIN for s in range(8) for y in [compose(q)[s]]}
    out = []
    for s, q, y in zip(he['s'], he['ops'], he['states'][:, -1]):
        nb = {lab[k] for j in range(3) for v in range(4) if v != q[j]
              for k in [(int(s), *[int(x) for x in q[:j]], v, *[int(x) for x in q[j+1:]])] if k in lab}
        out.append(int(y) not in nb)
    return np.array(out)


def main(commit):
    # 1. Task facts (lane QUESTION.md table), with permutation tuples.
    seen, frontier = {tuple(range(8))}, [tuple(range(8))]
    while frontier:
        frontier = [g2 for g in frontier for k in range(4) for g2 in [tuple(P[k][x] for x in g)] if g2 not in seen and not seen.add(g2)]
    short = {compose(q) for L in (1, 2) for q in product(range(4), repeat=L)}
    f3tr, f3he = {compose(q) for q in TRAIN}, {compose(q) for q in HELD}
    distinct84 = len(short | f3tr | f3he) == 84
    mine = {}
    for L in (4, 5, 6):
        fs = [compose(q) for q in product(range(4), repeat=L)]
        cnt = {}
        for f in fs:
            cnt[f] = cnt.get(f, 0) + 1
        mine[L] = (len(fs), len(cnt), sum(cnt[f] > 1 for f in fs), sum(f in short for f in fs), sum(f in f3tr for f in fs),
                   sum(f in f3he for f in fs), sum(f not in short | f3tr | f3he for f in fs))
    lane_table = {4: (256, 256, 0, 0, 0, 0, 256), 5: (1024, 1015, 18, 0, 0, 0, 1024), 6: (4096, 3907, 372, 8, 1, 1, 4086)}
    check(f'task facts: group order {len(seen)} (S8 = 40320); 84 products of lengths 1-3 distinct; length 4-6 function-class table equals the lane\'s',
          len(seen) == 40320 and distinct84 and mine == lane_table, str(mine))

    # 2. Part A, every row at lengths 1-6, my forward and oracle, against the lane's length.csv and steps.csv.
    rows_a = lane_csv(commit, 'runs/c2-gen-frozen-orig-a2/length.csv'); rows_s = lane_csv(commit, 'runs/c2-gen-frozen-orig-a2/steps.csv')
    he = oracle_rows(HELD); notn = notn_mask(he)
    worst, n_cmp, acc, steps_ok, windows = 0., 0, {}, True, {}
    for run in RUNS:
        p = nets.load(C2/'recovered'/run/'ckpt-10000.npz')[0]
        lane = {(int(r['length']), r['subset']): r for r in rows_a if r['run_id'] == run}
        for L in range(1, 7):
            progs = list(product(range(4), repeat=L)); d = oracle_rows(progs)
            pred = my_forward(p, d['s'], d['ops'], [1]*L)[1][-1].argmax(1); ok = pred == d['states'][:, -1]
            acc[run, L] = ok.mean()
            sub = {'all': np.ones(len(ok), bool)}
            q3 = [tuple(map(int, q[:3])) for q in d['ops']]
            if L == 3:
                held = np.array([q in set(HELD) for q in q3]); sub.update({'train programs': ~held, 'held-out programs': held})
                order = {q: i for i, q in enumerate(HELD)}  # held rows in oracle_rows(HELD) order
                idx = np.array([order[q]*8 + int(s) for q, s in zip(q3, d['s']) if q in order])
                hn = np.zeros(len(ok), bool); hn[held] = notn[idx]
                sub.update({'held-out N': held & ~hn, 'held-out notN': hn})
            if L >= 4:
                held = np.array([q in set(HELD) for q in q3])
                sub.update({'length-3 prefix is a training program': ~held, 'length-3 prefix is a held-out program': held})
                fs = [compose(q) for q in d['ops'][::8]]  # oracle_rows: programs in order, 8 starts each
                cls = np.repeat([('F12' if f in short else 'F3train' if f in f3tr else 'F3held' if f in f3he else 'novel') for f in fs], 8)
                sub.update({f'function class {c}': cls == c for c in ('F12', 'F3train', 'F3held', 'novel')})
                # CX-G1: any 3-op window a held-out program
                anyw = np.array([any(tuple(map(int, q[i:i+3])) in set(HELD) for i in range(L-2)) for q in d['ops']])
                windows[run, L] = (ok[anyw].mean(), ok[~anyw].mean(), int(anyw.sum()), int((~anyw).sum()))
            for name, m in sub.items():
                r = lane[L, name]; n_cmp += 1
                if int(r['n_rows']) != int(m.sum()):
                    worst = np.inf
                elif m.any():
                    worst = max(worst, abs(float(r['accuracy']) - ok[m].mean()))
        # steps.csv: per-step accuracy, first error, and the step after it, on all length-6 rows
        d = oracle_rows(list(product(range(4), repeat=6))); Z = my_forward(p, d['s'], d['ops'], [1]*6)[1]
        yhat = np.stack([z.argmax(1) for z in Z], 1); wrong = yhat != d['states']
        first = np.where(wrong.any(1), wrong.argmax(1), -1)
        for r in (r for r in rows_s if r['run_id'] == run and r['step'] != '0'):
            if not r['step'].isdigit():
                steps_ok &= r['step'] == 'never wrong in 6 steps' and int(r['first_error_here']) == (first == -1).sum(); continue
            t = int(r['step']) - 1; at = first == t
            steps_ok &= float(r['accuracy']) == (~wrong[:, t]).mean() and int(r['first_error_here']) == at.sum()
            if t < 5:
                nxt, rows_ = yhat[at, t+1], np.where(at)[0]
                follows = nxt == np.array([P[d['ops'][i, t+1]][yhat[i, t]] for i in rows_])
                rec = nxt == d['states'][at, t+1]
                steps_ok &= (int(r['after_first_error_n']), int(r['after_recovered']), int(r['after_follows_own_error']), int(r['after_other'])) == \
                    (int(at.sum()), int(rec.sum()), int(follows.sum()), int((~rec & ~follows).sum()))
    check(f'Part A: {n_cmp} (run, length, subset) accuracies and row counts in length.csv equal my forward and oracle on every row', worst < 1e-12, f'max |diff| {worst:.1e}')
    check('Part A: steps.csv accuracy, first-error counts and the after-first-error split (recovered / follows own error / other) equal mine', steps_ok)
    for run in RUNS:
        print(f'INFO {run}: accuracy at L = 1..6:', [round(float(acc[run, L]), 3) for L in range(1, 7)])
    # references.csv: oracle, and each checkpoint's own one-step map re-encoded (argmax readout of one step from start(s), iterated)
    refs = {(r['system'], int(r['length'])): float(r['accuracy']) for r in lane_csv(commit, 'runs/c2-gen-frozen-orig-a2/references.csv')}
    ref_ok = all(refs['oracle (per-row loop)', L] == 1. for L in range(1, 7))
    for run in RUNS:
        p = nets.load(C2/'recovered'/run/'ckpt-10000.npz')[0]
        step1 = np.array([[my_forward(p, np.array([s]), np.array([[k]]), [1])[1][0].argmax() for k in range(4)] for s in range(8)])
        for L in range(1, 7):
            d = oracle_rows(list(product(range(4), repeat=L))); y = d['s'].copy()
            for t in range(L):
                y = step1[y, d['ops'][:, t]]
            ref_ok &= refs[f'own one-step map re-encoded ({run})', L] == (y == d['states'][:, -1]).mean()
    check('references.csv: oracle 1.0 at every length; each checkpoint\'s own one-step map re-encoded equals mine at L = 1-6 (P2 line NOT CHECKED)', ref_ok)
    fig = lane_csv(commit, 'figures/length.csv')
    in_runs = lambda f: any(all(r[k] == f[k] for k in ('run_id', 'length', 'subset', 'n_rows', 'accuracy')) for r in rows_a)
    in_refs = lambda f: f['source'].endswith('references.csv') and f['subset'] == 'all' and float(f['accuracy']) == refs[f['run_id'], int(f['length'])]
    check('figures/length.csv: every row equals its source row (run length.csv or references.csv)', all(in_runs(f) or in_refs(f) for f in fig), f'{len(fig)} rows')

    # 3. CX-G1 (success branch: tied/inter L = 4 accuracy >= 0.5 in a seed).
    print('CX-G1 accuracy with any 3-op window a held-out program | with none (rows): ')
    held_ok = True
    for run in RUNS:
        if 'inter' in run and acc[run, 4] >= .5:
            for L in (4, 5, 6):
                a, b, na, nb = windows[run, L]
                held_ok &= abs(a - b) < .05
                print(f'  {run} L={L}: {a:.3f} ({na}) | {b:.3f} ({nb}); diff {a - b:+.3f}')
    check('CX-G1 prediction: tied/inter accuracy differs by < 0.05 between rows with and without a held-out window, at every L = 4-6 and seed', held_ok)


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
