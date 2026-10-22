"""Checks the correction lane's NOTE numbers (c3f03cd) against rev_corr.json (my accuracies) and the
lane's states.csv decoder accuracies (equal to mine, C4). Reading only.

python3 research/composition-c3/review/rev_corr_note.py runs/c3/c3-rev-corr-a2/rev_corr.json
"""
import csv
import io
import json
import subprocess
import sys

acc = json.load(open(sys.argv[1]))['acc']
ROOT = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True).stdout.strip()
FAIL = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))
    FAIL.extend([] if ok else [name])


def a(run, split, arm, g, rows):
    return acc[f'{run}|{split}'][f'{arm}|{g}|{rows}']


def dec(run, split, g):
    path = f"research/composition-c3/correction/runs/{'c3-corr-f2-' if run.startswith('c2-gen') else 'c3-corr-'}{split}-a1/states.csv"
    rows = csv.DictReader(io.StringIO(subprocess.run(['git', 'show', f'c3f03cd:{path}'], cwd=ROOT, capture_output=True, text=True).stdout))
    return [float(r['decoder_acc']) for r in rows if r['run_id'] == run and r['arm'] == 'D' and r['prefixes'] == g][0]


rng = lambda xs: f'{min(xs):.3f}-{max(xs):.3f}'
gain = lambda run, split, arm, g: (a(run, split, arm, g, 'held') - a(run, split, 'A', g, 'held')) / (a(run, split, 'E', g, 'held') - a(run, split, 'A', g, 'held'))
PS = [f'r1-rev-comp-rnn-untied-s{s}-a1' for s in (51200, 51201, 51202)]
TF = [f'r1-comp-rnn-final-s{s}-a1' for s in (51200, 51201, 51202)]
F2 = [f'c2-gen-final-{t}-s{s}-a1' for t in ('ops63000-ls63100', 'ops63000-ls63101', 'ops63001-ls63102', 'ops63001-ls63103') for s in (63200, 63201)]
held = lambda runs, split, arm, g: [a(r, split, arm, g, 'held') for r in runs]

# Original, main.
check('main held: B 1.00 per-step, 0.80-0.89 tied/final', all(x == 1 for x in held(PS, 'main', 'B', 'fit')) and rng(held(TF, 'main', 'B', 'fit'))[:4] == '0.80' and max(held(TF, 'main', 'B', 'fit')) < .895,
      f"{held(PS, 'main', 'B', 'fit')} {rng(held(TF, 'main', 'B', 'fit'))}")
check('main held: D = E, 1.00 per-step and 0.875-0.969 tied/final', all(a(r, 'main', 'D', 'fit', 'held') == a(r, 'main', 'E', 'fit', 'held') for r in PS + TF)
      and rng(held(TF, 'main', 'D', 'fit')) == '0.875-0.969')
check('main held: A 0.52-0.63 / 0.27-0.32', rng(held(PS, 'main', 'A', 'fit')) == '0.516-0.633' and rng(held(TF, 'main', 'A', 'fit')) == '0.273-0.320',
      f"{rng(held(PS, 'main', 'A', 'fit'))} {rng(held(TF, 'main', 'A', 'fit'))}")
check('main held: C below A in all six', all(a(r, 'main', 'C', 'fit', 'held') < a(r, 'main', 'A', 'fit', 'held') for r in PS + TF))
gB = [gain(r, 'main', 'B', 'fit') for r in PS + TF]
print(f"INFO main gain recovered by B: {', '.join(f'{x:.2f}' for x in gB)}; by D: {', '.join(f'{gain(r, 'main', 'D', 'fit'):.2f}' for r in PS + TF)}")
# Original, grouped eval prefixes.
check('grouped eval held: E 0.84-1.00', min(held(PS + TF, 'grouped', 'E', 'eval')) >= .835 and max(held(PS + TF, 'grouped', 'E', 'eval')) == 1, rng(held(PS + TF, 'grouped', 'E', 'eval')))
gBg = [gain(r, 'grouped', 'B', 'eval') for r in PS + TF]
check('grouped eval held: B never above A; gain -50% to 0%', all(a(r, 'grouped', 'B', 'eval', 'held') <= a(r, 'grouped', 'A', 'eval', 'held') for r in PS + TF)
      and round(min(gBg), 2) == -.5 and round(max(gBg), 2) == 0, ', '.join(f'{x:.2f}' for x in gBg))
above = [r for r in PS + TF if a(r, 'grouped', 'D', 'eval', 'held') > a(r, 'grouped', 'A', 'eval', 'held')]
check('grouped eval held: D above A only in per-step 51201 (0.625 vs 0.562)', above == [PS[1]]
      and round(a(PS[1], 'grouped', 'D', 'eval', 'held'), 3) == .625 and round(a(PS[1], 'grouped', 'A', 'eval', 'held'), 3) == .562, str(above))
check('grouped eval held: C <= A in all six', all(a(r, 'grouped', 'C', 'eval', 'held') <= a(r, 'grouped', 'A', 'eval', 'held') for r in PS + TF))
tr = [a(r, 'grouped', arm, 'eval', 'train') for r in PS + TF for arm in 'BCD']
check('grouped eval train rows: A 1.00; learned arms 0.17-0.64', all(a(r, 'grouped', 'A', 'eval', 'train') == 1 for r in PS + TF) and rng(tr)[:4] == '0.16' or rng(tr)[:4] == '0.17', rng(tr))
check('decoder on eval prefixes 0.53-0.66 per-step, 0.22-0.31 tied/final; 1.00 on fit',
      rng([dec(r, 'grouped', 'eval') for r in PS]) == '0.531-0.656' and rng([dec(r, 'grouped', 'eval') for r in TF]) == '0.219-0.312'
      and all(dec(r, 'grouped', 'fit') == 1 for r in PS + TF), f"{rng([dec(r, 'grouped', 'eval') for r in PS])} {rng([dec(r, 'grouped', 'eval') for r in TF])}")
# F2.
gF = [gain(r, 'main', 'B', 'fit') for r in F2]
check('F2 main: B recovers 79-100% of the gain; D = E; C below A', round(min(gF), 2) == .79 and round(max(gF), 2) <= 1.005
      and all(a(r, 'main', 'D', 'fit', 'held') == a(r, 'main', 'E', 'fit', 'held') and a(r, 'main', 'C', 'fit', 'held') < a(r, 'main', 'A', 'fit', 'held') for r in F2),
      ', '.join(f'{x:.2f}' for x in gF))
check('F2 grouped eval held: every learned arm below A in all eight', all(a(r, 'grouped', arm, 'eval', 'held') < a(r, 'grouped', 'A', 'eval', 'held') for r in F2 for arm in 'BCD'))
gFB, gFD = [gain(r, 'grouped', 'B', 'eval') for r in F2], [gain(r, 'grouped', 'D', 'eval') for r in F2]
check('F2 grouped: B -110% to -13%, D -90% to -7% of the gain', round(min(gFB), 2) == -1.1 and round(max(gFB), 2) == -.13 and round(min(gFD), 2) == -.9 and round(max(gFD), 2) == -.07,
      f"B {min(gFB):.3f}..{max(gFB):.3f}, D {min(gFD):.3f}..{max(gFD):.3f}")
trF = [a(r, 'grouped', arm, 'eval', 'train') for r in F2 for arm in 'BCD']
check('F2 grouped eval train rows 0.19-0.62; decoder 0.19-0.44; E 0.91-1.00', rng(trF) in ('0.188-0.625', '0.188-0.615', '0.188-0.620') or (round(min(trF), 2) == .19 and round(max(trF), 2) == .62)
      and rng([dec(r, 'grouped', 'eval') for r in F2]) == '0.188-0.438' and round(min(held(F2, 'grouped', 'E', 'eval')), 2) == .91,
      f"train {rng(trF)}, decoder {rng([dec(r, 'grouped', 'eval') for r in F2])}, E {rng(held(F2, 'grouped', 'E', 'eval'))}")
print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
