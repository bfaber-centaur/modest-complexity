#!/usr/bin/env python3
"""Figure for QUESTION-3.md from unitfilter_check.py output: per-unit dominant-frequency
filter, on trained runs and on a constructed control (step-0 features, readout fitted in
closed form to the training pairs).

uv run --locked --group analysis python research/e2/audit/plot_unitfilter_check.py OUT_STEM UNITFILTER_CHECK.csv
"""
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
rows = list(csv.DictReader(open(sys.argv[2])))
for r in rows:
    for k in ('step', 'common_acc', 'train_acc'):
        r[k] = float(r[k])
COND = [('step0', 'step 0\n(3 inits)', '#a3a3a3', 'o'), ('ridge', 'step-0 features,\nreadout fitted to\n30% train pairs', '#333333', 'D'),
        ('ridge50', 'same, 50%\ntrain pairs\n(post hoc)', '#333333', 'd'), ('shuf', 'shuffled\nlabels', '#9e3d22', 'X'),
        ('add0', 'a+b labels,\nno decay', '#c9a227', 's'), ('add30', 'a+b labels,\ndecay 1, 30%', '#7b3294', 'o'),
        ('add50', 'a+b labels,\ndecay 1, 50%', '#2a7bb6', 'D')]
IDX = {c: i for i, (c, *_) in enumerate(COND)}


def cond(run, step):
    if run.startswith('ridge50'):
        return 'ridge50'
    if run.startswith('ridge'):
        return 'ridge'
    if step == 0:
        return 'step0'
    return 'add50' if 'tf50' in run else 'add0' if 'add0' in run else 'shuf' if 'shuf' in run else 'add30'


sel = lambda run, step, kind: [r for r in rows if r['run_id'] == run and r['step'] == step and r['kind'] == kind]
keys = list(dict.fromkeys((r['run_id'], r['step']) for r in rows))
final = {}
for run, step in keys:
    final[run] = max(final.get(run, 0), step)
fig, ax = plt.subplots(1, 3, figsize=(17, 5), layout='constrained', gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
rng = np.random.default_rng(0)
done_step0 = set()
for run, step in keys:
    c = cond(run, step)
    if c not in ('ridge', 'ridge50', 'step0') and step != final[run]:
        continue
    if c == 'step0':  # identical networks across conditions with the same init: draw each once
        init = run.split('-i')[1][:5]
        if init in done_step0:
            continue
        done_step0.add(init)
    x0 = IDX[c] + rng.uniform(-.18, .18)
    for kind, style in (('filter random non-dominant', dict(color='#d0d0d0', marker='.', s=10)),
                        ('filter dominant, phases scrambled', dict(color='#9a9a9a', marker='_', s=30))):
        v = [r['common_acc'] for r in sel(run, step, kind)]
        ax[0].scatter(np.full(len(v), x0 + .1), v, zorder=1, **style)
    _, _, color, marker = COND[IDX[c]]
    ax[0].scatter([x0], [sel(run, step, 'filter dominant')[0]['common_acc']], color=color, marker=marker, s=45, zorder=3,
                  edgecolor='white', lw=.5)
    ax[0].scatter([x0], [sel(run, step, 'none')[0]['common_acc']], facecolor='none', edgecolor=color, marker=marker, s=45, zorder=2)
ax[0].axhline(1/31, color='0.3', lw=.8, ls=(0, (1, 2)))
ax[0].text(len(COND) - .5, 1/31 + .01, 'chance 1/31', ha='right', fontsize=7, color='0.3')
ax[0].set(xticks=range(len(COND)), xticklabels=[lab for _, lab, *_ in COND], ylim=(-.02, 1.04),
          ylabel='accuracy on the 483 common pairs (argmax vs a+b)',
          title='Filter every unit to its dominant input frequency (filled) vs unedited (open), final step\n'
                'gray: nulls, random other frequency (dots) and phases scrambled (dashes), 8 draws each')
from matplotlib.lines import Line2D
ax[0].legend(handles=[Line2D([], [], color='0.3', marker='o', ls='none', label='filtered'),
                      Line2D([], [], mfc='none', mec='0.3', marker='o', ls='none', label='unedited'),
                      Line2D([], [], color='#d0d0d0', marker='.', ls='none', label='null: random other frequency'),
                      Line2D([], [], color='#9a9a9a', marker='_', ls='none', ms=8, label='null: phases scrambled')],
             fontsize=7, loc='center left', frameon=False)
for run in dict.fromkeys(r['run_id'] for r in rows):
    if run.startswith('ridge'):
        continue
    steps = sorted({s for r_, s in keys if r_ == run})
    c = cond(run, steps[-1]); color, marker = COND[IDX[c]][2:]
    ls = '-' if len(steps) > 2 else 'none'
    ax[1].plot([max(s, 100) for s in steps], [sel(run, s, 'filter dominant')[0]['common_acc'] for s in steps],
               color=color, marker=marker, ms=4, ls=ls, lw=1.3)
    ax[1].plot([max(s, 100) for s in steps], [sel(run, s, 'none')[0]['common_acc'] for s in steps],
               color=color, ls=':' if ls == '-' else 'none', marker=marker, mfc='none', ms=3, lw=1)
ax[1].axhline(1/31, color='0.3', lw=.8, ls=(0, (1, 2)))
ax[1].set(xscale='log', xlabel='update (step 0 drawn at 100)', ylabel='accuracy on the 483 common pairs', ylim=(-.02, 1.04),
          title='Over training: filtered (solid) and unedited (dotted)')
for run, step in keys:
    c = cond(run, step)
    color, marker = COND[IDX[c]][2:]
    f = sel(run, step, 'filter dominant')[0]
    ax[2].scatter([f['train_acc']], [f['common_acc']], color=color, marker=marker, s=22, alpha=.85, edgecolor='white', lw=.4)
ax[2].plot([0, 1], [0, 1], color='0.6', lw=.8, ls='--')
ax[2].set(xlim=(-.02, 1.04), ylim=(-.02, 1.04), xlabel='filtered train accuracy (own training labels)',
          ylabel='filtered accuracy on the 483 common pairs',
          title='Filtered models: held-out against train accuracy\n(every run, step and control)')
ax[2].legend(handles=[Line2D([], [], color=col, marker=m, ls='none', label=lab.replace('\n', ' ')) for _, lab, col, m in COND],
             fontsize=6.5, loc='upper left', frameon=False)
for a in ax:
    a.grid(True, color='0.92', lw=.6)
fig.suptitle('Width 128: what does the per-unit frequency filter need? Trained runs against a readout fitted in closed '
             'form to the training pairs on random step-0 features', fontsize=10)
fig.text(.01, -.02, 'Independent reimplementation of research/e2/mechanisms/unitfilter.py (filter rows agree with its CSV '
         'within 5e-7). Constructed control: U, V of the step-0 network, W = ridge fit (lambda 1e-6 and 1e-2 of the mean Gram '
         'eigenvalue) to one-hot a+b on the training pairs; no optimizer. Shuffled-label runs: train accuracy is against their '
         'own labels. Source: research/e2/audit/results/unitfilter-check.csv.', fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
