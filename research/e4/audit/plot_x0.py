#!/usr/bin/env python3
"""Figure for control X0 (QUESTION.md addendum 4) from x0_check.py output: exempt-reference decay (xref)
from five starts, each beside the same start under uniform decay.

uv run --locked --group analysis python research/e4/audit/plot_x0.py OUT_STEM RESULTS_DIR
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
rows = list(csv.DictReader(open(res/'x0-series.csv')))
x0 = json.loads((res/'x0-check.json').read_text())
SEEDS = (30000, 30001, 30002)
LS = {30000: '-', 30001: '--', 30002: ':'}
XREF, WD1, WD0 = '#2a5fb6', '#3a3a3a', '#a8a8a8'
PANELS = [('B (edited 50k start)', 'B xref', 'B wd 1', 'B wd 0', None),
          ('I (untrained init)', 'I', 'I wd 1 (e1)', None, 'e4-au-init-xref-w128-i{}-a1'),
          ('RF (random-feature start, C1)', 'RF', 'RF wd 1', 'RF wd 0', 'e4-au-rf-xref-w128-i{}-a1'),
          ('A (unedited 50k start)', 'A', 'A wd 1', None, 'e4-au-keep-xref-w128-i{}-a1'),
          ('S (B start, k permuted across units)', 'S', 'B wd 1', None, 'e4-au-filt-xrefsham-w128-i{}-a1')]


def series(group, seed, key):
    rr = [r for r in rows if r['group'] == group and int(r['seed']) == seed and r[key] != '']
    return [int(r['step']) for r in rr], [float(r[key]) for r in rr]


fig, ax = plt.subplots(2, 5, figsize=(19, 7.4), sharex=True, layout='constrained')
for j, (title, arm, one, zero, pat) in enumerate(PANELS):
    a, b = ax[0, j], ax[1, j]
    for s in SEEDS:
        for grp, color, lw in ((zero, WD0, 1.), (one, WD1, 1.), (arm, XREF, 1.6)):
            if grp is None:
                continue
            x, y = series(grp, s, 'own')
            a.plot(x, y, color=color, ls=LS[s], lw=lw)
        x, y = series(arm, s, 'decay_norm')
        _, n = series(arm, s, 'norm')
        nn = dict(zip(*series(arm, s, 'norm')))
        b.plot(x, [d/(1e-3*nn[t - 10]) if t - 10 in nn else float('nan') for t, d in zip(x, y)], color=XREF, ls=LS[s], lw=1.4)
        if one:
            x, y = series(one, s, 'decay_norm')
            nn = dict(zip(*series(one, s, 'norm')))
            if x:
                b.plot(x, [d/(1e-3*nn[t - 10]) if t - 10 in nn else float('nan') for t, d in zip(x, y)], color=WD1, ls=LS[s], lw=.8)
        if pat:
            acc = x0['runs'][pat.format(s)]['acc']['5000']
            a.plot(5000, acc, 'o', color=XREF, ms=5, mfc='white', zorder=5)
    if j == 0:
        for s in SEEDS:
            a.plot(5000, x0['comparators'][f'B xref/{s}']['5000'], 'o', color=XREF, ms=5, mfc='white', zorder=5)
    a.axhline(.9, color='0.6', lw=.7, ls=(0, (4, 3)))
    a.axhline(.3, color='0.6', lw=.7, ls=(0, (1, 2)))
    lab = x0['labels'].get(arm, dict(arm='lane result'))['arm']
    a.set(title=f'{title}\nxref label: {lab}', ylim=(0, 1.02))
    b.set(ylim=(0, 1.05), xlabel='update')
    for q in (a, b):
        q.grid(True, color='0.92', lw=.6)
        q.set_xlim(0, 5000)
ax[0, 0].set_ylabel('own held-out accuracy (662 pairs)')
ax[1, 0].set_ylabel('decay this update / (lr·wd·‖w‖)\n= share of the weights that decays')
ax[0, 4].legend(handles=[Line2D([], [], color=XREF, lw=2, label='xref (decay on all but each unit\'s k_j part)'),
                         Line2D([], [], color=WD1, lw=1.5, label='uniform wd 1, same start'),
                         Line2D([], [], color=WD0, lw=1.5, label='wd 0, same start'),
                         Line2D([], [], color=XREF, marker='o', mfc='white', lw=0, label='common pairs (483) at 5,000')]
                + [Line2D([], [], color='0.3', ls=ls, label=f'init {s}') for s, ls in LS.items()],
                loc='center right', fontsize=7.5, frameon=False)
fig.suptitle('Control X0: exempting each unit\'s own frequency from decay rescues every start, the untrained init included; '
             'a permuted frequency has not by 5,000 (0.14–0.33, still rising)', fontsize=11)
fig.text(.01, -.01, 'xref decays w − P(w), where P keeps each unit\'s cosine and sine at its update-0 dominant input frequency k_j '
         '(token axis of U and V, label axis of W) and DC belongs to the rest. Labels at 5,000 on the common pairs: rescued ≥ 0.90 (dashed), '
         'not rescued ≤ 0.30 (dotted), stated in QUESTION.md addendum 4 before the runs. B xref: the mechanisms lane\'s runs (PR #40). '
         'I under uniform wd 1 is the e1 run from the same init (logged every 100). Bottom: logged decay norm over lr·wd·‖w‖ of the previous '
         'logged state (10 updates earlier), so uniform decay sits at 1. Sources: research/e4/audit/results/x0-series.csv, x0-check.json.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
