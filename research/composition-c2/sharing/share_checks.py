#!/usr/bin/env python3
"""Oracle checks for the sharing lane, no training (M1). Writes out/checks.txt.

1. Init identity: at init seeds 51200-51202 the four tying schemes give identical hidden
   states and logits on all 512 rows, and untied copies are equal but independent arrays.
2. Gradient aggregation: at init, untied copies' gradients sum to the tied gradient (and the
   start embedding's gradient is the same in every scheme).
3. The recovered TT / TU ckpt-0 weights equal c2rnn.init for their scheme and seed.
4. N / notN from r1's logged held_final_pred reproduce the reviewer's 10k numbers, and the
   N subset has 86 rows.
5. Once the follow-up has run: every wd 0 run starts from its wd 0.1 counterpart's weights.

uv run --locked python research/composition-c2/sharing/share_checks.py
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import share_runs as S  # noqa: E402
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import nets  # noqa: E402

lines = []


def say(*a):
    lines.append(' '.join(map(str, a))); print(*a)


def check(cond, msg):
    say(('PASS ' if cond else 'FAIL ') + msg)
    assert cond, msg


T, nbr = S.task()
tr, he = c2task.rows(T['P'], T['train']), c2task.rows(T['P'], T['held'])
check(int(nbr.sum()) == 86 and int((~nbr).sum()) == 42, f'N subset: {int(nbr.sum())} N, {int((~nbr).sum())} notN rows')
targets = [tr['states'][:, t] for t in range(3)]

for seed in S.SEEDS:
    P = {c: c2rnn.init(128, seed, S.TIE[c][0] == 'tied', S.TIE[c][1] == 'tied') for c in S.CELLS}
    ref = {name: c2rnn.forward(P['TT'], d['ops'], s=d['s']) for name, d in (('train', tr), ('held', he))}
    for c in S.CELLS[1:]:
        same = all(np.array_equal(a, b) for name, d in (('train', tr), ('held', he))
                   for a, b in zip(sum(map(list, c2rnn.forward(P[c], d['ops'], s=d['s'])), []), sum(map(list, ref[name]), [])))
        check(same, f'seed {seed}: {c} hidden states and logits at init equal TT bitwise on 384 + 128 rows')
        copies = [k for k in P[c] if k[-1] in '123']
        indep = all(not np.shares_memory(P[c][a], P[c][b]) for a in copies for b in copies if a != b)
        check(indep and all(np.array_equal(P[c][k], P['TT'][k[:-1]]) for k in copies),
              f'seed {seed}: {c} copies {sorted(copies)} equal the tied draw and share no memory')
    g = {c: c2rnn.loss_grad(P[c], tr['s'], tr['ops'], targets, [1., 1., 1.])[2] for c in S.CELLS}
    worst = 0.
    for c in S.CELLS[1:]:
        for name, v in g['TT'].items():
            summed = g[c][name] if name in g[c] else sum(g[c][f'{name}{k}'] for k in (1, 2, 3))
            worst = max(worst, float(np.max(np.abs(summed - v)) / (np.max(np.abs(v)) + 1e-300)))
    check(worst < 1e-12, f'seed {seed}: untied copies\' gradients sum to the tied gradient (worst relative error {worst:.1e})')
    for c in ('TT', 'TU'):
        p0 = nets.load(S.checkpoint(c, seed, 0))[0]
        check(set(p0) == set(P[c]) and all(np.array_equal(p0[k], P[c][k]) for k in p0),
              f'seed {seed}: recovered {S.run_id(c, seed)} ckpt-0 weights equal c2rnn.init')

expected = dict(TT=(1.0, 0.881, 1.0), TU=(0.24, 0.07, 0.05))  # REVIEW.md section 1, counterexamples 1 and 2
for c in ('TT', 'TU'):
    got = [S.trajectory(c, s, T, nbr)[-1] for s in S.SEEDS]
    check(all(r['step'] == 10000 for r in got) and all(abs(r['notN'] - e) < 0.005 for r, e in zip(got, expected[c])),
          f'{c} notN at 10k from logged predictions: {[round(r["notN"], 3) for r in got]} (reviewer {list(expected[c])})')
    say(f'   {c} held-out final / N at 10k: {[round(r["held_final"], 3) for r in got]} / {[round(r["N"], 3) for r in got]}')

# Follow-up pairing: each wd 0 run starts from the same weights as its wd 0.1 counterpart, and
# their step-0 metric rows (taken before any update) are identical.
for c in S.CELLS:
    for seed in S.SEEDS:
        try:
            p_wd0 = nets.load(S.checkpoint(c, seed, 0, 'wd0'))[0]
        except FileNotFoundError:
            continue
        p_main = nets.load(S.checkpoint(c, seed, 0))[0]
        r_wd0, r_main = S.read_metrics(S.metrics_path(c, seed, 'wd0'))[0], S.read_metrics(S.metrics_path(c, seed))[0]
        same_rows = all(r_wd0[k] == r_main[k] for k in ('step', 'train', 'held', 'held_final_pred'))
        check(set(p_wd0) == set(p_main) and all(np.array_equal(p_wd0[k], p_main[k]) for k in p_main) and same_rows,
              f'seed {seed}: {S.run_id(c, seed, "wd0")} ckpt-0 weights and step-0 metrics equal {S.run_id(c, seed)}')

out = Path(__file__).resolve().parent/'out'; out.mkdir(exist_ok=True)
(out/'checks.txt').write_text('\n'.join(lines)+'\n')
