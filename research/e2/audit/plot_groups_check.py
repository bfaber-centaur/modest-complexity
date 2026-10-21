#!/usr/bin/env python3
"""Figure for QUESTION-2.md from groups_check.py output: do frequency groups carry an
own-frequency term aligned with a+b in runs that fit a+b labels without weight decay
(add0), or that fit shuffled labels (shuf)? And are group ablations matched to random
sets on the held-out logit change?

uv run --locked --group analysis python research/e2/audit/plot_groups_check.py OUT_STEM GROUPS_CHECK.csv
"""
import csv
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
rows = list(csv.DictReader(open(sys.argv[2])))
for r in rows:
    for k, v in r.items():
        try:
            r[k] = float(v)
        except ValueError:
            pass
COND = [('shuf', 'shuffled labels,\ndecay 1', '#9e3d22'), ('add0', 'a+b labels,\nno decay', '#c9a227'),
        ('add30', 'a+b labels, decay 1,\n30% share', '#7b3294'), ('add50', 'a+b labels, decay 1,\n50% share', '#2a7bb6')]
cond = lambda run: 'add50' if 'tf50' in run else 'add0' if 'add0' in run else 'shuf' if 'shuf' in run else 'add30'
runs = sorted({r['run_id'] for r in rows}, key=lambda run: ([c for c, *_ in COND].index(cond(run)), run))
final = {run: max(r['step'] for r in rows if r['run_id'] == run) for run in runs}
first = {run: min(r['step'] for r in rows if r['run_id'] == run) for run in runs}
sel = lambda run, step, kind: [r for r in rows if r['run_id'] == run and r['step'] == step and r['kind'] == kind]
scale = {(run, s): sel(run, s, 'none')[0]['rms_logits_common'] for run in runs for s in (first[run], final[run])}
xpos = {run: [c for c, *_ in COND].index(cond(run)) + (i % 3 - 1)*.22 for i, run in enumerate(runs)}
color = {c: col for c, _, col in COND}

fig, ax = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
for run in runs:
    x = xpos[run]
    for step, filled in ((first[run], False), (final[run], True)):
        g = sel(run, step, 'group')
        rel = np.array([r['k_amplitude_common'] for r in g])/scale[run, step]
        ph = [r['k_phase_common'] for r in g]
        style = dict(color=color[cond(run)]) if filled else dict(facecolor='none', edgecolor='#a3a3a3')
        dx = .06 if filled else -.06
        ax[0, 0].scatter(np.full(len(g), x + dx), ph, s=4 + 2500*rel, alpha=.75 if filled else .9, lw=.8, **style)
        ax[0, 1].scatter(np.full(len(g), x + dx), rel, s=14, alpha=.75 if filled else .9, lw=.8, **style)
    g = sel(run, final[run], 'group')
    ax[0, 0].text(x, 3.35, f"{sum(abs(r['k_phase_common']) < math.pi/2 for r in g)}/{len(g)}", ha='center', fontsize=7)
    rs = sel(run, final[run], 'random set')
    ax[1, 0].scatter([x], [np.median([r['d_share_common'] for r in g])], color=color[cond(run)], s=30, zorder=3)
    ax[1, 0].scatter([x], [np.median([r['d_share_common'] for r in rs])], marker='_', color='0.25', s=120, lw=1.5, zorder=3)
    gs = sel(run, first[run], 'group')
    ax[1, 0].scatter([x], [np.median([r['d_share_common'] for r in gs])], facecolor='none', edgecolor='#a3a3a3', s=30)
    if cond(run) == 'add50':
        base = sel(run, final[run], 'none')[0]['common_acc']
        xr = np.array([r['rms_change_common'] for r in rs]); yr = np.array([r['common_acc'] - base for r in rs])
        ax[1, 1].scatter(xr, yr, color='#a3a3a3', s=6, alpha=.6, lw=0)
        ax[1, 1].scatter([r['rms_change_common'] for r in g], [r['common_acc'] - base for r in g],
                         color=color['add50'], s=16, zorder=3)
        fit = np.polyfit(xr, yr, 1); xs = np.linspace(xr.min(), xr.max(), 2)
        ax[1, 1].plot(xs, np.polyval(fit, xs), color='0.3', lw=.8)
ticks = dict(xticks=range(len(COND)), xticklabels=[lab for _, lab, _ in COND])
ax[0, 0].axhspan(-math.pi/2, math.pi/2, color='0.95', lw=0, zorder=0)
ax[0, 0].axhline(0, color='0.3', lw=.8)
ax[0, 0].set(ylim=(-math.pi, 3.7), yticks=[-math.pi, -math.pi/2, 0, math.pi/2, math.pi],
             yticklabels=['−π', '−π/2', '0 (peaks at a+b)', 'π/2', 'π'], ylabel='phase of the group\'s own-frequency term',
             title='Phase on the 483 common pairs (shaded: within ±π/2; count per seed at top)\n'
                   'marker area ∝ amplitude relative to the logit scale', **ticks)
ax[0, 1].set(yscale='log', ylabel='own-frequency amplitude / RMS of centred logits',
             title='Amplitude of each group\'s own-frequency term, relative to the run\'s logits', **ticks)
ax[1, 0].set(yscale='log', ylabel='share explained by a function of a+b−c (median over groups)',
             title='Group share (dot) against matched random sets (bar), common pairs', **ticks)
ax[1, 1].axhline(0, color='0.3', lw=.8)
ax[1, 1].set(xlabel='RMS change of the centred logits on the common pairs', ylabel='change in common-set accuracy',
             title='50% runs: ablate a group (blue) or a matched random set (gray)\nline: random-set trend per run')
for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.grid(True, axis='y', color='0.92', lw=.6)
from matplotlib.lines import Line2D
ax[0, 1].legend(handles=[Line2D([], [], marker='o', ls='none', color='0.4', label='step 50,000 (filled, colour = condition)'),
                         Line2D([], [], marker='o', ls='none', mfc='none', mec='#a3a3a3',
                                label='same run at step 0 (300 for 30% init 30002)')], fontsize=7, loc='lower right')
fig.suptitle('Width 128, units grouped by dominant input frequency at each checkpoint: does the aligned a+b−c term need '
             'a+b labels, weight decay, or neither? Three seeds per condition (left to right: init 30000, 30001, 30002)',
             fontsize=10)
fig.text(.01, -.02, 'Independent reimplementation of research/e2/mechanisms/groups.py (agrees with its groups.csv within '
         '5e-7 on all 255 shared group rows). Random sets: 16 per group, same count, total |u||w| within 10%, seed 35200. '
         'Source: research/e2/audit/results/groups-check.csv.', fontsize=7, color='0.3', ha='left', va='top', wrap=True)
out.parent.mkdir(parents=True, exist_ok=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
