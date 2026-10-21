#!/usr/bin/env python3
"""Figure for the "rank precedes accuracy" check: width 128, 30% vs 50% share, the 483
pairs neither share trains on, every published checkpoint (rank_checks.py shares.jsonl).

uv run --locked --group analysis python research/e2/audit/plot_precedes.py OUT_STEM SHARES.jsonl
"""
import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
lines = [json.loads(s) for s in Path(sys.argv[2]).read_text().splitlines() if s.strip()]
chance, rows = lines[0], lines[1:]
COLOR = {'30%': '#9a9a9a', '50%': '#2a7bb6'}
MARK = {'30000': 'o', '30001': 's', '30002': '^'}
panels = [('rank_50mask', 'restricted rank of a+b\n(labels absent from the 50% training set)', .5, 'no preference'),
          ('rank_all_labels', 'rank of a+b among all 30 other labels', .5, 'no preference'),
          ('cross_entropy', 'cross-entropy (nats)', math.log(31), 'uniform, ln 31'),
          ('acc', 'accuracy', 1/31, 'chance 1/31')]

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, layout='constrained')
for run in dict.fromkeys(r['run'] for r in rows):
    rs = [r for r in rows if r['run'] == run]
    share, seed = rs[0]['share'], run.split('-i')[1][:5]
    x = [r['step'] for r in rs]
    # A run with two published checkpoints gets markers only, not a straight join.
    style = dict(color=COLOR[share], marker=MARK[seed], ms=4, lw=1.5, ls='-' if len(rs) > 3 else 'none')
    for ax, (key, *_) in zip(axes, panels):
        ax.plot(x, [r[key] for r in rs], label=f'{share} share, init {seed}', **style)
    if share == '30%' and len(rs) > 3:
        axes[0].plot(x, [r['rank_own_mask'] for r in rs], color=COLOR[share], lw=.8, ls=':')
band = [r['rank_50mask'] for r in rows if r['share'] == '30%' and r['step'] == 50000]
span = axes[0].axhspan(min(band), max(band), color=COLOR['30%'], alpha=.18, lw=0)
for ax, (key, label, ref, ref_label) in zip(axes, panels):
    ax.axhline(ref, color='0.3', lw=.8, ls=(0, (1, 2)))
    ax.text(.99, ref, ref_label, transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=7, color='0.3')
    ax.set(title=label, xlabel='update (symmetric log scale)', xscale='symlog')
    ax.set_xscale('symlog', linthresh=1000, linscale=.6); ax.set_xlim(-150, 60000)
    ax.set_xticks([0, 1000, 2000, 5000, 10000, 20000, 50000], ['0', '1k', '2k', '5k', '10k', '20k', '50k'])
    ax.grid(True, color='0.92', lw=.6)
axes[0].set_ylim(-.02, .62); axes[1].set_ylim(-.02, .9); axes[3].set_ylim(-.02, 1)
axes[3].legend(fontsize=7, frameon=False, loc='upper left')
axes[0].legend([span, plt.Line2D([], [], color=COLOR['30%'], lw=.8, ls=':')],
               ['range of the 30% runs at step 50,000', '30% runs ranked with their own mask'],
               fontsize=7, frameon=False, loc='lower left')
fig.suptitle('Does the restricted rank lead the shares apart before accuracy does? Width 128, the 483 pairs neither '
             'share trains on, published checkpoints', fontsize=10)
fig.text(.01, -.03, 'One line per run; the same init seed gives the same step-0 weights in both shares. Panel 1 ranks a+b among '
         f"the labels absent from the 50% training set (median {chance['absent_min_median_max'][1]:.0f} per pair, "
         f"{chance['absent_min_median_max'][0]}-{chance['absent_min_median_max'][2]}), which are absent from the 30% "
         'set too; dotted: the 30% runs with their own mask. Normalized rank: 0 = above all, 0.5 = no preference. '
         + ''.join(f"{r.split('-a1')[0]} has only steps {', '.join(str(x['step']) for x in rows if x['run'] == r)}. "
                   for r in dict.fromkeys(x['run'] for x in rows) if sum(x['run'] == r for x in rows) <= 3)
         + f'Source: {sys.argv[2]}.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
out.parent.mkdir(parents=True, exist_ok=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
