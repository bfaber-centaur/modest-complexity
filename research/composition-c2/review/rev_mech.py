"""Mechanisms lane checks (QUESTION.md 2.3 and Revisions 04:21; REVIEW.md section 4). Analysis
only, on the recovered r1 checkpoints. Natural interchange is done by actual patching: the donor's
h_i replaces the recipient's h_j, the recipient's remaining ops run through the shared transition,
and the recipient's final readout (V, or V3 for per-step readouts) reads the answer. Compared with
the lane's cells.csv at a lane commit; then counterexample CX-M1.

uv run --locked python research/composition-c2/review/rev_mech.py LANE_COMMIT [FOLLOWUP_COMMIT [NOTE_COMMIT]]
"""
import csv
import io
from itertools import product
import json
import subprocess
import sys
import numpy as np

from rev_pin import C2, FAIL, ROOT, TRAIN, check, my_forward, oracle_rows
import nets

RUNS = [(c, s, f'r1-comp-rnn-{c}-s{s}-a1' if c != 'per-step' else f'r1-rev-comp-rnn-untied-s{s}-a1')
        for c in ('inter', 'per-step', 'final') for s in (51200, 51201, 51202)]
COND = {'inter': 'tied/inter', 'per-step': 'per-step readout/inter', 'final': 'tied/final'}
LANE = 'research/composition-c2/mechanisms'
ALL = list(product(range(4), repeat=3))


def lane_csv(commit, path):
    return list(csv.DictReader(io.StringIO(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{LANE}/{path}'], text=True))))


def run_suffix(p, h, suffix, readout):
    for k in suffix:
        h = np.maximum(h @ p['A'] + p['B'][k] + p['bh'], 0)
    V, bv = (p['V'], p['bv']) if 'V' in p else (p[f'V{readout}'], p[f'bv{readout}'])
    return (h @ V + bv).argmax(1)


def apply(suffix, x):
    from rev_pin import P
    for k in suffix:
        x = np.array([P[k][v] for v in x])
    return x


def main(commit):
    # M1 at ckpt-0 and ckpt-1000: my forward against r1's published held-out prediction strings
    he = oracle_rows(sorted(set(ALL) - set(TRAIN))); ok = True
    for cond, seed, run in RUNS:
        lane = 'review' if cond == 'per-step' else 'composition'
        rows = {r['step']: r for r in map(json.loads, (ROOT/'research'/'regimes'/lane/'runs'/run/'metrics.jsonl').read_text().splitlines())}
        for step in (0, 1000):
            p = nets.load(C2/'recovered'/run/f'ckpt-{step}.npz')[0]
            pred = my_forward(p, he['s'], he['ops'], [1, 2, 3])[1][-1].argmax(1)
            ok &= ''.join(map(str, pred)) == rows[step]['held_final_pred']
    check('M1 at ckpt-0 and ckpt-1000 (9 runs each): my forward gives r1\'s published 128-character held-out prediction string', ok)

    cells = {(r['run_id'], int(r['i']), int(r['j'])): r for r in lane_csv(commit, 'runs/c2-mech-interchange-a1/cells.csv') if r['step'] == '10000'}
    pairs = lane_csv(commit, 'runs/c2-mech-interchange-a1/pairs.csv')
    d = oracle_rows(ALL); tr = oracle_rows(TRAIN)
    sym = lambda dd, i: dd['s'] if i == 0 else dd['states'][:, i-1]
    worst, worst_c, worst_v, diag_ok, self_ok, cx = 0., 0., 0., True, True, []
    for cond, seed, run in RUNS:
        p = nets.load(C2/'recovered'/run/'ckpt-10000.npz')[0]
        H = my_forward(p, d['s'], d['ops'], [1, 2, 3])[0]; Ht = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[0]
        cent = {i: np.stack([Ht[i][sym(tr, i) == x].mean(0) for x in range(8)]) for i in range(4)}
        S = {}
        for i, j in product(range(4), repeat=2):
            sufs = list(product(range(4), repeat=3-j)); x = sym(d, i)
            nat = np.array([run_suffix(p, H[i], q, 3) == apply(q, x) for q in sufs])               # (suffixes, 512 donors)
            cen = np.array([run_suffix(p, cent[i], q, 3) == apply(q, np.arange(8)) for q in sufs])  # (suffixes, 8 symbols)
            S[i, j] = nat.mean(); r = cells[run, i, j]
            worst = max(worst, abs(S[i, j] - float(r['S']))); worst_c = max(worst_c, abs(cen.mean() - float(r['S_cent'])))
            if cond == 'per-step':
                for v in (1, 2):
                    sv = np.mean([run_suffix(p, H[i], q, v) == apply(q, x) for q in sufs])
                    worst_v = max(worst_v, abs(sv - float(r[f'S_V{v}'])))
            if i == j:
                own = run_suffix(p, H[0], d['ops'].T, 3) == d['states'][:, -1]   # unpatched final correctness
                lane_self = next(float(q['success']) for q in pairs if q['run_id'] == run and q['step'] == '10000' and q['i'] == q['j'] == str(i)
                                 and q['subset'] == 'all' and q['category'] == 'self')
                self_ok &= lane_self == own.mean()
            gap = cen.mean() - S[i, j]
            if gap >= .05 - 1e-12:
                # CX-M1 (a): donors whose h_i is nearest (Euclidean) to the training centroid of their own symbol
                near = np.linalg.norm(H[i][:, None, :] - cent[i][None], axis=2).argmin(1) == x
                # (b): each donor rescaled to the norm of its symbol's centroid
                hs = H[i] * (np.linalg.norm(cent[i][x], axis=1) / np.maximum(np.linalg.norm(H[i], axis=1), 1e-300))[:, None]
                resc = np.mean([run_suffix(p, hs, q, 3) == apply(q, x) for q in sufs])
                cx.append((cond, seed, i, j, S[i, j], cen.mean(), near.mean(), nat[:, near].mean(), resc))
        diag_ok &= all(S[i, j] == S[i+1, j+1] for i in range(3) for j in range(3))
    check(f'natural interchange by actual patching: S(i, j) in all 16 cells x 9 runs at 10k equals cells.csv', worst < 1e-12, f'max |diff| {worst:.1e}')
    check('centroid patches (centroids fit here on the 384 training rows): S_cent equals cells.csv in all 144 cells', worst_c < 1e-12, f'max |diff| {worst_c:.1e}')
    check('per-step readout: the V1 and V2 columns equal mine', worst_v < 1e-12, f'max |diff| {worst_v:.1e}')
    check('S is constant along every diagonal (i - j fixed) in all 9 runs; self-swap success equals unpatched accuracy', diag_ok and self_ok)

    print('CX-M1 cells with S_cent - S >= 0.05: S, S_cent, share of donors symbol-present, S on those donors, S with donors norm-matched; '
          'fraction of the gap closed by (a) and (b)')
    ok = True
    for cond, seed, i, j, s, sc, share, s_near, resc in cx:
        fa, fb = (s_near - s) / (sc - s), (resc - s) / (sc - s)
        if cond == 'inter':
            ok &= fa < .5 and fb < .25
        print(f'  {COND[cond]:23s} {seed} i{i} j{j} L{3+i-j}: S {s:.3f} S_cent {sc:.3f}; symbol-present {share:.3f}, S on them {s_near:.3f}; '
              f'norm-matched {resc:.3f}; closed (a) {fa:+.2f} (b) {fb:+.2f}')
    check('CX-M1 prediction (tied/inter): in every such cell (a) closes < 1/2 and (b) < 1/4 of the gap', ok)



def followup(commit):
    """Follow-up B (c2-mech-residual-a1): residual dose and the inside/outside-U split, recomputed."""
    res = {(r['run_id'], r['i'], r['L'], r['kind'], r['param'], r['subset']): float(r['accuracy'])
           for r in lane_csv(commit, 'runs/c2-mech-residual-a1/residual.csv')}
    d = oracle_rows(ALL); tr = oracle_rows(TRAIN); trs = set(TRAIN)
    held = np.array([tuple(map(int, q)) not in trs for q in d['ops']])
    worst, n = 0., 0
    for cond, seed, run in RUNS:
        p = nets.load(C2/'recovered'/run/'ckpt-10000.npz')[0]
        H = my_forward(p, d['s'], d['ops'], [1, 2, 3])[0]; Ht = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[0]
        for i in (2, 3):
            x, xt = d['states'][:, i-1], tr['states'][:, i-1]
            c = np.stack([Ht[i][xt == v].mean(0) for v in range(8)])
            U = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][:7].T
            r = H[i] - c[x]; inside = r @ U @ U.T
            states = {('dose', str(lam)): c[x] + lam * r for lam in (0., .25, .5, .75, 1.)}
            states.update({('inside', 'U'): c[x] + inside, ('outside', 'U'): c[x] + r - inside})
            for (kind, param), h in states.items():
                if i == 2:
                    ok = np.concatenate([run_suffix(p, h[d['ops'][:, 2] == k], [k], 3) == d['states'][d['ops'][:, 2] == k, 2] for k in range(4)])
                    m = np.concatenate([held[d['ops'][:, 2] == k] for k in range(4)])
                    for name, sel in (('held', m), ('train', ~m), ('all', np.ones_like(m))):
                        worst = max(worst, abs(ok[sel].mean() - res[run, '2', '3', kind, param, name])); n += 1
                else:
                    for j in (2, 1, 0):
                        ok = np.array([run_suffix(p, h, q, 3) == apply(q, x) for q in product(range(4), repeat=3-j)])
                        for name, sel in (('held_history', held), ('train_history', ~held), ('all', np.ones_like(held))):
                            worst = max(worst, abs(ok[:, sel].mean() - res[run, '3', str(6-j), kind, param, name])); n += 1
    check(f'follow-up B: residual dose (5 values) and inside/outside-U accuracies, recomputed with centroids and U fit here on training rows, '
          f'equal residual.csv ({n} entries)', worst < 1e-12, f'max |diff| {worst:.1e}')


def o6_scale():
    """NOTE 0fe8624 O6: share of each depth-2 training centroid's squared norm outside U (a pure scale residual r = a c(x))."""
    tr = oracle_rows(TRAIN); lo = 1.
    for cond, seed, run in RUNS:
        p = nets.load(C2/'recovered'/run/'ckpt-10000.npz')[0]
        Ht = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[0]
        c = np.stack([Ht[2][tr['states'][:, 1] == v].mean(0) for v in range(8)])
        U = np.linalg.svd(c - c.mean(0), full_matrices=False)[2][:7].T
        out = 1 - np.linalg.norm(c @ U, axis=1)**2 / np.linalg.norm(c, axis=1)**2
        lo = min(lo, out.min())
        print(f'INFO O6 {COND[cond]:23s} {seed}: share of |c(x)|^2 outside U, over the 8 symbols: {out.min():.3f}-{out.max():.3f}')
    check('NOTE O6 "a pure scale residual lies largely outside U": share outside U > 0.5 for every depth-2 centroid in all 9 runs', lo > .5, f'minimum {lo:.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
    if len(sys.argv) > 2:
        followup(sys.argv[2])
    if len(sys.argv) > 3:
        o6_scale()
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
