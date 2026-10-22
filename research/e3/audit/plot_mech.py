#!/usr/bin/env python3
"""Figure for research/e3/audit/QUESTION-3.md from mech_check.py output.

uv run --locked --group analysis python research/e3/audit/plot_mech.py OUT_STEM RESULTS_DIR
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

out, res = Path(sys.argv[1]), Path(sys.argv[2])
num = lambda r: {k: (v if k in ('run_id', 'arm') else (float(v) if v != '' else np.nan)) for k, v in r.items()}
ck = [num(r) for r in csv.DictReader(open(res/'mech-checkpoints.csv'))]
dense = [num(r) for r in csv.DictReader(open(res/'mech-dense.csv'))]
cross = json.loads((res/'mech-check.json').read_text())['cross_source']
ARM = {'keep': ('#7f7f7f', 'A, unchanged'), 'filt': ('#2166ac', 'B, dominant-frequency filter'),
       'sham': ('#d6604d', 'C, input-energy-matched sham')}
SRC = {30000: 'o', 30001: 's', 30002: '^'}
X = dict(xscale='symlog', xlim=(0, 25000), xticks=[0, 10, 100, 1000, 10000], xticklabels=['0', '10', '100', '1k', '10k'])


def series(rows, arm, s, key, x='step'):
    rr = sorted([r for r in rows if r['arm'] == arm and r['source'] == s and np.isfinite(r[key])], key=lambda r: r[x])
    return [r[x] for r in rr], [r[key] for r in rr]


fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), layout='constrained')
a = ax[0, 0]
for arm, (color, _) in ARM.items():
    for s, mk in SRC.items():
        a.plot(*series(dense, arm, s, 'common_acc', 'update'), color=color, lw=1.2)
        a.plot(*series(ck, arm, s, 'common_acc'), color=color, lw=1.2, ls='--', marker=mk, ms=4, mec='white', mew=.3)
a.axhline(1/31, color='0.4', lw=.8, ls=(0, (1, 2)))
a.text(22000, 1/31 + .01, 'chance 1/31', ha='right', fontsize=7, color='0.35')
a.set(**X, ylim=(-.02, 1), xlabel='continuation update (linear to 10, log after)', ylabel='accuracy on the 483 common never-trained pairs',
      title='B rises for ~70–90 updates, then falls to chance\n(solid: my replay, every update to 300; dashed + markers: published checkpoints)')

a = ax[0, 1]
for arm, (color, _) in ARM.items():
    for s, mk in SRC.items():
        a.plot(*series(ck, arm, s, 'P_cS'), color=color, lw=1.2, ls='--', marker=mk, ms=4, mec='white', mew=.3)
        a.plot(*series(ck, arm, s, 'P_cN'), color=color, lw=1.2, marker=mk, ms=4, mec='white', mew=.3)
a.set(**X, ylim=(-.02, 1.02), xlabel='continuation update', ylabel='fraction of the set ranked below a+b',
      title='In ranks, B keeps part of the edit at 20k\n(solid: P(a+b over N); dashed: P(a+b over S))')

a = ax[0, 2]
a.axhspan(.927, .948, color='0.88', lw=0)
a.text(.4, .921, 'band: trained 30% runs at 50k,\nwith and without decay', fontsize=7, color='0.35', va='top')
for arm, (color, _) in ARM.items():
    for s, mk in SRC.items():
        a.plot(*series(ck, arm, s, 'P_SN'), color=color, lw=1.2, marker=mk, ms=4, mec='white', mew=.3)
a.set(**X, ylim=(.55, 1.0), xlabel='continuation update', ylabel='P(S over N)',
      title='Operand-label preference returns in both edited arms\n(band: investigation 1)')

a = ax[1, 0]
st = [c['step'] for c in cross]
lo, hi = [min(c['A_A_agreement']) for c in cross], [max(c['A_A_agreement']) for c in cross]
a.fill_between(st, lo, hi, color='0.85', lw=0, label='A against A of another source')
for arm in ('filt', 'sham'):
    for s, mk in SRC.items():
        a.plot(*series(ck, arm, s, 'agree_with_A'), color=ARM[arm][0], lw=1.2, marker=mk, ms=4, mec='white', mew=.3)
a.set(**X, ylim=(0, .7), xlabel='continuation update', ylabel='argmax agreement with A (same source, same update)',
      title='Neither edited arm becomes A again: agreement\nends where independent A runs agree with each other')
a.legend(fontsize=7.5, frameon=False, loc='upper left')

a = ax[1, 1]
for arm in ('filt', 'sham'):
    for s, mk in SRC.items():
        a.plot(*series(ck, arm, s, 'distance_from_A'), color=ARM[arm][0], lw=1.2, marker=mk, ms=4, mec='white', mew=.3)
a.set(**X, ylim=(0, 1.05), xlabel='continuation update', ylabel='‖w − w_A‖ / ‖w_A‖ (U, V, W together)',
      title='B moves further from A while it is erased')

a = ax[1, 2]
src = list(SRC)
get = lambda arm, s, step, key: next(r[key] for r in ck if r['arm'] == arm and r['source'] == s and r['step'] == step)
peak = {s: max(r['common_acc'] for r in dense if r['arm'] == 'filt' and r['source'] == s) for s in src}
parts = [('supplied by the edit\nB(0) − A(0)', [get('filt', s, 0, 'common_acc') - get('keep', s, 0, 'common_acc') for s in src]),
         ('added by training\npeak − B(0)', [peak[s] - get('filt', s, 0, 'common_acc') for s in src]),
         ('retained at 20k\nB − A', [get('filt', s, 20000, 'common_acc') - get('keep', s, 20000, 'common_acc') for s in src])]
for i, (name, vals) in enumerate(parts):
    for j, (s, v) in enumerate(zip(src, vals)):
        a.bar(i + (j - 1)*.26, v, width=.24, color=ARM['filt'][0], edgecolor='white', lw=1)
        a.text(i + (j - 1)*.26, v + .01, f'{v:.2f}', ha='center', fontsize=7, color='0.25')
a.set(xticks=range(3), xticklabels=[p[0] for p in parts], ylim=(0, .7), ylabel='common-set accuracy',
      title='Edit against training, per source (bars: i30000, i30001, i30002)')
for a in ax.flat:
    a.grid(True, color='0.92', lw=.6)
ax[1, 1].legend(handles=[Line2D([], [], color=c, lw=2, label=l) for c, l in ARM.values()] +
                [Line2D([], [], color='0.4', marker=m, ls='none', label=f'source: 50k checkpoint of init {s}') for s, m in SRC.items()],
                loc='lower right', fontsize=8, frameon=False)
fig.suptitle('e3 mechanisms packet 1, independently recomputed: 20,000 updates after a fresh-Adam restart from the 50k checkpoints of the three failed 30% runs',
             fontsize=11)
fig.text(.01, -.01, 'All values from my own forward pass on the published checkpoints (research/e3/mechanisms/runs), and an independent replay of updates 0–300 '
         'from the start files (research/audit/independent.py; it matches every published checkpoint to 1e-9 of the weight scale). Common pairs: the 483 pairs '
         'outside the 50% training set of the data-seed-31000 ladder; S, N: e2 sets. Agreement is on the argmax over all 31 labels. One split and width; the '
         'three sources share their evaluation pairs, so pairs are not replicates. Source: research/e3/audit/results/mech-checkpoints.csv, mech-dense.csv, mech-check.json.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
