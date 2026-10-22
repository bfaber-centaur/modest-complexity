#!/usr/bin/env python3
"""Figure for research/e3/audit/QUESTION-4.md: the early-rise probe and the weight-decay-0
continuations, from probe.py and mech_check.py output.

uv run --locked --group analysis python research/e3/audit/plot_probe.py OUT_STEM RESULTS_DIR
"""
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

out, res = Path(sys.argv[1]), Path(sys.argv[2])
num = lambda r: {k: (v if k in ('run_id', 'arm', 'condition') else (float(v) if v != '' else np.nan)) for k, v in r.items()}
probe = [num(r) for r in csv.DictReader(open(res/'probe.csv'))]
ck = [num(r) for r in csv.DictReader(open(res/'mech-checkpoints.csv'))]
dense = [num(r) for r in csv.DictReader(open(res/'mech-dense.csv'))]
SRC = {30000: 'o', 30001: 's', 30002: '^'}
COND = {'reference': ('#2166ac', 'Adam lr 1e-3 (the runs)', 1e-3), 'small Adam': ('#74add1', 'Adam lr 1e-4', 1e-4),
        'gradient descent': ('#1b7837', 'gradient descent, step matched to Adam\'s first', 1e-3)}
ARM = {'filt': ('#2166ac', '-', 'B, wd 1'), 'filt-wd0': ('#2166ac', '--', 'B, wd 0'),
       'sham': ('#d6604d', '-', 'C, wd 1'), 'sham-wd0': ('#d6604d', '--', 'C, wd 0')}
X = dict(xscale='symlog', xlim=(0, 25000), xticks=[0, 10, 100, 1000, 10000], xticklabels=['0', '10', '100', '1k', '10k'])

fig, ax = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')
for a, xkey in ((ax[0, 0], 'decay'), (ax[0, 1], 'train_ce')):
    for cond, (color, _, rate) in COND.items():
        for s, mk in SRC.items():
            rr = [r for r in probe if r['condition'] == cond and r['source'] == s]
            if xkey == 'train_ce':  # up to the lowest train cross-entropy: under gradient descent, decay later pushes it back up
                low = min(range(len(rr)), key=lambda i: rr[i]['train_ce'])
                rr = rr[:low + 1]
            x = [r['update']*rate for r in rr] if xkey == 'decay' else [r['train_ce'] for r in rr]
            a.plot(x, [r['common_acc'] for r in rr], color=color, lw=1.2)
            fit = next(r for r in rr if r['train_acc'] == 1.0)
            a.plot(fit['update']*rate if xkey == 'decay' else fit['train_ce'], fit['common_acc'], marker=mk, color=color,
                   ms=7, mec='white', mew=.6, ls='none', zorder=3)
ax[0, 0].set(xscale='log', xlim=(1e-4, 3.5), ylim=(0, 1), xlabel='decay time: decoupled decay rate × updates (lr × updates for Adam)',
             ylabel='accuracy on the 483 common never-trained pairs',
             title='The early rise does not need the restart\'s ±lr steps\n(markers: first update with train accuracy 1)')
ax[0, 1].set(xscale='log', ylim=(0, 1), xlabel='train cross-entropy (falls to the right)', ylabel='accuracy on the common pairs',
             title='The rise happens while the training errors are repaired, under all three rules\n(each curve up to its lowest train cross-entropy)')
ax[0, 1].invert_xaxis()
ax[0, 0].legend(handles=[Line2D([], [], color=c, lw=2, label=l) for c, l, _ in COND.values()], fontsize=8, frameon=False, loc='upper left')

for a, key, title in ((ax[1, 0], 'common_acc', 'Without weight decay B keeps most of its held-out gain\n(solid: wd 1; dashed: wd 0; lines to 300: my replay)'),
                      (ax[1, 1], 'P_cS', 'In ranks too: P(a+b over S) stays near 0.9 without decay')):
    for arm, (color, ls, _) in ARM.items():
        for s, mk in SRC.items():
            if key == 'common_acc':
                rr = sorted([r for r in dense if r['arm'] == arm and r['source'] == s], key=lambda r: r['update'])
                a.plot([r['update'] for r in rr], [r[key] for r in rr], color=color, ls=ls, lw=1.1)
            rr = sorted([r for r in ck if r['arm'] == arm and r['source'] == s], key=lambda r: r['step'])
            a.plot([r['step'] for r in rr], [r[key] for r in rr], color=color, ls=ls, lw=1.1, marker=mk, ms=4, mec='white', mew=.3)
    if key == 'P_cS':
        rr = [r for r in ck if r['arm'] == 'keep']
        a.axhspan(min(r['P_cS'] for r in rr), max(r['P_cS'] for r in rr), color='0.85', lw=0)
        a.text(.4, .2, 'band: A, unchanged', fontsize=7.5, color='0.35')
    a.set(**X, ylim=(-.02, 1.02), xlabel='continuation update', ylabel='accuracy on the common pairs' if key == 'common_acc'
          else 'fraction of operand labels ranked below a+b', title=title)
ax[1, 0].axhline(1/31, color='0.4', lw=.8, ls=(0, (1, 2)))
ax[1, 0].legend(handles=[Line2D([], [], color=c, ls=ls, lw=2, label=l) for c, ls, l in ARM.values()] +
                [Line2D([], [], color='0.4', marker=m, ls='none', label=f'source init {s}') for s, m in SRC.items()],
                fontsize=8, frameon=False, loc='upper left', ncol=2)
for a in ax.flat:
    a.grid(True, color='0.92', lw=.6)
fig.suptitle('e3 mechanisms, audit probe (D13) and the weight-decay-0 continuations, from the three filtered (B) and sham (C) start files', fontsize=11)
fig.text(.01, -.01, 'Top: diagnostic replays with my own code from the published B start files on the 299 training pairs; no run IDs. Gradient descent '
         'uses the runs\' decoupled decay (× (1 − 1e-3) per update) and a step size that gives its first update the norm of Adam\'s first. '
         'Bottom: my own forward pass on the published checkpoints of e3-me-<filt|sham>[-wd0]-w128-i<seed>-a1 and an independent replay of '
         'their first 300 updates. Under gradient descent, late in the replay decay outweighs the gradient steps: train cross-entropy rises again while train accuracy stays 1 and held-out accuracy falls to 0 (top left). Sources: research/e3/audit/results/probe.csv, mech-checkpoints.csv, mech-dense.csv.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
