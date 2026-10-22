#!/usr/bin/env python3
"""Figure for control C1 (research/e4/audit/QUESTION.md, section 1) from c1_check.py output.

uv run --locked --group analysis python research/e4/audit/plot_c1.py OUT_STEM RESULTS_DIR
"""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

out, res = Path(sys.argv[1]), Path(sys.argv[2])
j = json.loads((res/'c1-check.json').read_text())
R = j['runs']
COLOR = {'B': '#2a5fb6', 'rf': '#d95f02', 'A': '#8c8c8c'}
LS = {30000: '-', 30001: '--', 30002: ':'}
NAME = {'B': 'B: filtered failed run (e3)', 'rf': 'random-feature start, filtered (C1)', 'A': 'A: unedited failed run (e3)'}


def series(run, key):
    x = sorted(int(s) for s in run[key])
    return x, [run[key][str(s)] for s in x]


def ck(run, key):
    x = sorted(int(s) for s in run['checkpoints'])
    return x, [run['checkpoints'][str(s)][key] for s in x]


fig, ax = plt.subplots(2, 2, figsize=(13, 9.5), layout='constrained')
for col, wd in enumerate((1., 0.)):
    a = ax[0, col]
    for run in R.values():
        if run['wd'] != wd or run['arm'] == 'A':
            continue
        x, y = series(run, 'dense_own_heldout')
        a.plot(x, y, color=COLOR[run['arm']], ls=LS[run['seed']], lw=1.4)
        x, y = ck(run, 'common_acc')
        a.plot(x, y, 'o', color=COLOR[run['arm']], ms=3.5, mec='white', mew=.4)
        dec = run['decline']
        for name, mk in (('update0', 'v'), ('fixed010', 'X')):
            t = dec[f'T_{name}']
            if isinstance(t, int):
                a.plot(t, run['dense_own_heldout'][str(t)], mk, color=COLOR[run['arm']], ms=8, mec='black', mew=.6)
    a.axhline(1/31, color='0.5', lw=.8, ls=(0, (1, 2)))
    a.text(12, 1/31 + .012, 'chance', fontsize=7.5, color='0.4')
    a.set(xscale='symlog', xlim=(0, 2.2e4), ylim=(0, 1), xlabel='update after the edit (linear to 10, log after)',
          ylabel='held-out accuracy', title=f'weight decay {wd:g}: '
          + ('both starts rise, then fall below their update-0 level and below 0.10' if wd else 'both starts rise and hold, ending at 0.56–0.62'))
    a.set_xticks([0, 10, 100, 1000, 10000])
    a.set_xticklabels(['0', '10', '100', '1,000', '10,000'])

a = ax[1, 0]
for run in R.values():
    if run['wd'] != 1. or run['arm'] == 'A':
        continue
    x, y = series(run, 'dense_train')
    a.plot(x, y, color=COLOR[run['arm']], ls=LS[run['seed']], lw=1.4)
a.set(xscale='symlog', xlim=(0, 2000), ylim=(0, 1.02), xlabel='update after the edit (linear to 10, log after)', ylabel='train accuracy (299 pairs)',
      title='The random-feature start begins less fit (0.16–0.27 against 0.69–0.79)\nand first fits every training pair at update 140, against 40–70 for B (wd 1)')
a.set_xticks([0, 10, 100, 1000])
a.set_xticklabels(['0', '10', '100', '1,000'])

a = ax[1, 1]
for run in R.values():
    x, y = ck(run, 'P_yS')
    a.plot(x, y, color=COLOR[run['arm']], ls=LS[run['seed']], lw=1.4 if run['wd'] else .9, marker='o' if run['wd'] else 's',
           ms=3.5, mfc=COLOR[run['arm']] if run['wd'] else 'white', alpha=1 if run['wd'] else .75)
a.set(xscale='symlog', xlim=(0, 2.2e4), ylim=(0, 1), xlabel='update after the edit (linear to 10, log after)',
      ylabel='P(a+b over S), 483 common pairs',
      title='Rank trace at 20,000, wd 1 (filled): P(a+b over S) is 0.39–0.46 for both\nfiltered starts, against 0.14–0.18 for the unedited A (open: wd 0)')
a.set_xticks([0, 10, 100, 1000, 10000])
a.set_xticklabels(['0', '10', '100', '1,000', '10,000'])
for a in ax.flat:
    a.grid(True, color='0.92', lw=.6)
ax[0, 0].legend(handles=[Line2D([], [], color=COLOR[k], lw=2, label=NAME[k]) for k in ('B', 'rf')]
                + [Line2D([], [], color='0.3', ls=ls, label=f'init {s}') for s, ls in LS.items()]
                + [Line2D([], [], color='0.3', marker='o', ls='none', ms=4, label='483 common pairs (checkpoints)'),
                   Line2D([], [], color='0.3', marker='v', ls='none', mec='black', label='first below update-0 level'),
                   Line2D([], [], color='0.3', marker='X', ls='none', mec='black', label='first below 0.10')],
                loc='upper right', fontsize=7.5, frameon=False)
ax[1, 1].legend(handles=[Line2D([], [], color=COLOR[k], lw=2, label=NAME[k]) for k in ('B', 'rf', 'A')],
                loc='center left', fontsize=7.5, frameon=False)
rule = j['rule']
fig.suptitle('e4 audit control C1: a random-feature start, filtered the same way, rises and collapses like B '
             f"(rise to update 100: {', '.join(f'{rule[s]['rise_rf']:.2f}' for s in rule)} against B's "
             f"{', '.join(f'{rule[s]['rise_B']:.2f}' for s in rule)}; verdict {j['verdict']})", fontsize=10.5)
fig.text(.01, -.01, 'Lines: own held-out accuracy (cal ∪ audit, 662 pairs) every 10 updates from metrics.jsonl; dots: my forward pass on the 483 common pairs at published checkpoints. '
         'Random-feature start: untrained U, V of the same init, ridge readout on the 299 training labels, the e2 per-unit dominant-frequency filter, '
         'then U, V and W rescaled to B\'s norms (research/e4/audit/rf_starts.py). Both arms: fresh Adam, lr 1e-3, 20,000 updates, the same 299 pairs. '
         'Decline markers use the dynamics lane\'s definitions (peak over 0–1,000; first logged update after it below θ). '
         'Runs: e4-au-rf[-wd0]-w128-i{30000,30001,30002}-a1; B: e3-me-filt[-wd0]-w128-i…-a1; A: e3-me-keep-w128-i…-a1. Source: research/e4/audit/results/c1-check.json.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
