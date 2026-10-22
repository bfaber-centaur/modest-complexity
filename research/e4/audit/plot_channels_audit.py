#!/usr/bin/env python3
"""Figure for QUESTION.md section 2 and addendum 2 (the mechanisms channel split) from
channels_check.py and lane_check.py output.

uv run --locked --group analysis python research/e4/audit/plot_channels_audit.py OUT_STEM RESULTS_DIR
"""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

out, res = Path(sys.argv[1]), Path(sys.argv[2])
rows = [r for r in csv.DictReader(open(res/'channels-check.csv'))]
lc = json.loads((res/'lane-check.json').read_text())
SEEDS = (30000, 30001, 30002)
LS = {30000: '-', 30001: '--', 30002: ':'}
COLOR = {('B', 1.): '#2a5fb6', ('rf', 1.): '#d95f02', ('B', 0.): '#8fb3e8', ('A', 1.): '#8c8c8c'}
NAME = {('B', 1.): 'B, wd 1', ('rf', 1.): 'random-feature start (C1), wd 1', ('B', 0.): 'B, wd 0', ('A', 1.): 'A (unedited), wd 1'}


def series(arm, wd, seed, key, scale=1.):
    rr = sorted((r for r in rows if r['arm'] == arm and float(r['wd']) == wd and int(r['seed']) == seed), key=lambda r: int(r['step']))
    return [int(r['step']) for r in rr], [scale*float(r[key]) for r in rr]


def xaxis(a):
    a.set(xscale='symlog', xlim=(0, 2.2e4), xlabel='update after the edit (linear to 10, log after)')
    a.set_xticks([0, 10, 100, 1000, 10000])
    a.set_xticklabels(['0', '10', '100', '1,000', '10,000'])
    a.grid(True, color='0.92', lw=.6)


fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), layout='constrained')
a = ax[0]
for key in (('B', 1.), ('rf', 1.), ('A', 1.), ('B', 0.)):
    for s in SEEDS:
        x, y = series(*key, s, 'acc_ref_only')
        a.plot(x, y, color=COLOR[key], ls=LS[s], lw=1.5 if key[1] else 1)
        if key in (('B', 1.), ('rf', 1.)):
            x, y = series(*key, s, 'acc_full')
            a.plot(x, y, color=COLOR[key], ls=LS[s], lw=.7, alpha=.5)
for arm in ('B', 'rf'):
    for s in SEEDS:
        v = lc['c']['values'][f'{arm}/{s}']
        for t in ('5000', '20000'):
            a.plot(int(t)*(1.08 if arm == 'rf' else .93), v[t]['sham_median'], 'x', color=COLOR[(arm, 1.)], ms=6, mew=1.4)
xaxis(a)
a.set(ylim=(0, 1), ylabel='accuracy on the 483 common pairs',
      title='Reference channel alone (thick) against the full network (thin):\nhalf right after the collapse in B and C1 alike; A gets 0.16–0.28;\npermuting k_j across units (×, 8 draws, median) gives chance')

a = ax[1]
for key in (('B', 1.), ('rf', 1.), ('B', 0.)):
    for s in SEEDS:
        for ch, lw in (('ref', 1.6), ('off', 1.6)):
            x, y = series(*key, s, f'mS_{ch}_per_np', 1e3)
            a.plot(x, y, color=COLOR[key], ls=LS[s] if ch == 'ref' else LS[s], lw=lw if key[1] else 1,
                   marker='o' if ch == 'ref' else 'v', ms=3, mfc=COLOR[key] if ch == 'ref' else 'white')
a.axhline(0, color='0.5', lw=.8)
xaxis(a)
a.set(ylabel='mean margin of a+b over S, per 1,000 units of ‖(U, V)‖·‖W‖',
      title='Channels in a scale no channel enters (post hoc): reference (●) falls\nby half at wd 1, off-reference (▽) grows from zero at both wd,\n2.5× more at wd 1 by 2,000; C1 follows B')
a.text(11, -.6, 'off-reference', fontsize=8, color='0.3')
a.text(11, 4.2, 'reference', fontsize=8, color='0.3')

a = ax[2]
for arm in ('B', 'rf'):
    for s in SEEDS:
        v = lc['c']['values'][f'{arm}/{s}']
        st = sorted(int(k[6:]) for k in v if k.startswith('start_'))
        a.plot(st, [v[f'start_{t}']['s_t'] for t in st], color=COLOR[(arm, 1.)], ls=LS[s], lw=1.5, marker='o', ms=3)
        a.plot(st, [v[f'start_{t}']['cos'] for t in st], color=COLOR[(arm, 1.)], ls=LS[s], lw=1, marker='s', ms=3, mfc='white')
xaxis(a)
a.set(ylim=(0, 1.02), ylabel='reference readout W_ref,t against the start W_ref,0 (wd 1)',
      title='"Not erased" is the direction, not the edited weights:\nthe decayed start is at most 0.8% of W_ref by 5,000 (●, ρ‖W_ref,0‖/‖W_ref,t‖),\nyet W_ref,t keeps cosine 0.72–0.83 with it (□)')
ax[2].legend(handles=[Line2D([], [], color=COLOR[k], lw=2, label=NAME[k]) for k in COLOR]
             + [Line2D([], [], color='0.3', ls=ls, label=f'init {s}') for s, ls in LS.items()],
             loc='lower left', fontsize=8, frameon=False)
fig.suptitle('e4 audit of the mechanisms channel split (PR #35): the reference channel is not manufactured by the basis, '
             'but it is the same in the random-feature runs and is rebuilt by the gradient', fontsize=10.5)
fig.text(.01, -.01, 'Channels: label offset μ·W, reference (h − μ)·W_ref (each unit\'s readout row at its update-0 dominant input frequency k_j), '
         'off-reference (h − μ)·W_off; they sum to the label-centred logits (max relative error 1.7e-15; the lane\'s channels.py agrees with mine to 7e-16 '
         'on 175 states). Floor: the same split on A. Sham: k_j permuted across live units (seed 41100). Start content uses W_t = ρ_t·W_0 + ΔW_t, '
         'ρ_t = (1 − lr·wd)^t, exact for experiment.adam. Sources: research/e4/audit/results/channels-check.csv, lane-check.json.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
