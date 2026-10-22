#!/usr/bin/env python3
"""Figure for research/e3/audit/QUESTION.md from preference_check.py output, with the e2
audit cycle-4 max-based gap alongside. Also writes the values at matched training
cross-entropy.

uv run --locked --group analysis python research/e3/audit/plot_preference.py OUT_STEM PREFERENCE.csv E2_MARGINS.csv MATCHED.csv
"""
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

out = Path(sys.argv[1])
num = lambda r: {k: (v if k in ('kind', 'run_id', 'share', 'invariant') or v == '' else float(v)) for k, v in r.items()}
rows = [num(r) for r in csv.DictReader(open(sys.argv[2]))]
gap = {(r['run_id'], int(float(r['step']))): float(r['sigma_gap_S_over_N']) for r in csv.DictReader(open(sys.argv[3]))}
SH = {'30%': ('#9a9a9a', 'o', 299), '40%': ('#9ecae1', 's', 359), '45%': ('#4292c6', '^', 417), '50%': ('#08519c', 'D', 478)}
CTRL = {'no decay 30%': ('#c9a227', 'a+b labels, no decay, 30%'), 'shuffled 30%': ('#9e3d22', 'shuffled labels, 30% (a+b sets)')}
COLS = ('P_SN', 'P_SN_pooled', 'P_SN_transpose_groups', 'P_SN_median', 'P_DN', 'P_cN', 'P_cS', 'acc')
lad = [r for r in rows if r['kind'] == 'a+b, decay 1']
runs = {run: sorted([r for r in rows if r['run_id'] == run], key=lambda r: r['step'])
        for run in dict.fromkeys(r['run_id'] for r in rows if r['step'] != '')}


def at_ce(rr, col, ce):
    """Linear in log training cross-entropy between the checkpoints that first bracket `ce` (as in e2 audit cycle 4)."""
    for x, y in zip(rr, rr[1:]):
        if x['train_ce'] >= ce > y['train_ce']:
            t = (np.log(ce) - np.log(x['train_ce']))/(np.log(y['train_ce']) - np.log(x['train_ce']))
            return x[col] + t*(y[col] - x[col])


matched = [dict(run_id=run, share=rr[0]['share'], init_seed=int(rr[0]['init_seed']), train_ce=1e-4,
                **{c: round(at_ce(rr, c, 1e-4), 6) for c in COLS})
           for run, rr in runs.items() if rr[0]['kind'] == 'a+b, decay 1']
with open(sys.argv[4], 'w', newline='') as f:
    w = csv.DictWriter(f, list(matched[0])); w.writeheader(); w.writerows(matched)

fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.5), layout='constrained')
a = ax[0, 0]
for run, rr in runs.items():
    kind = rr[0]['kind']
    color, marker = (SH[rr[0]['share']][:2] if kind == 'a+b, decay 1' else (CTRL[kind][0], 'x'))
    a.plot([max(r['step'], 100) for r in rr], [r['P_SN'] for r in rr], color=color, marker=marker, ms=3.5, lw=1.3,
           ls='-' if kind == 'a+b, decay 1' else '--', mec='white' if marker != 'x' else color, mew=.3 if marker != 'x' else 1)
a.axhline(.5, color='0.4', lw=.8, ls=(0, (1, 2)))
a.set(xscale='log', ylim=(.4, 1.0), xlabel='update (step 0 drawn at 100)',
      ylabel='P(S over N): chance that an operand-associated\nlabel outranks an unrelated one (mean over 483 pairs)',
      title='Operand-label preference, free of logit scale: no growth in training')
a.legend(handles=[Line2D([], [], color=c, marker=m, ms=5, label=f'a+b labels, decay 1, {s} ({n} pairs)') for s, (c, m, n) in SH.items()] +
         [Line2D([], [], color=c, marker='x', ls='--', label=lab) for c, lab in CTRL.values()],
         fontsize=7, frameon=False, loc='center right')

a = ax[0, 1]
x = {s: n for s, (_, _, n) in SH.items()}
for s, (color, marker, n) in SH.items():
    v50 = [r['P_SN'] for r in lad if r['share'] == s and r['step'] == 50000]
    vce = [r['P_SN'] for r in matched if r['share'] == s]
    a.scatter(np.full(len(v50), n - 4), v50, color=color, marker=marker, s=34, edgecolor='0.3', lw=.4, zorder=3)
    a.scatter(np.full(len(vce), n + 4), vce, facecolor='none', edgecolor=color, marker=marker, s=34, lw=1.1, zorder=3)
for lam, mk in ((1e-6, 'P'), (1e-2, 'X')):
    mem = [r for r in rows if r['kind'] == f'memorizer lambda {lam:g}']
    a.scatter([x[r['share']] for r in mem], [r['P_SN'] for r in mem], color='#d95f02', marker=mk, s=26, zorder=3)
    a.plot(list(x.values()), [np.mean([r['P_SN'] for r in mem if r['share'] == s]) for s in x], color='#d95f02', lw=.9)
a.set(xticks=list(x.values()), xticklabels=[f'{s}\n{n}' for s, n in x.items()], ylim=(.68, 1.0),
      xlabel='training share and pairs (nested sets)', ylabel='P(S over N)',
      title='It falls a little with share; a memorizer shows no share effect')
a.legend(handles=[Line2D([], [], color='0.4', marker='o', ls='none', mec='0.3', label='trained, at 50k (filled)'),
                  Line2D([], [], color='0.4', marker='o', ls='none', mfc='none', label='trained, at training cross-entropy 1e-4 (open)'),
                  Line2D([], [], color='#d95f02', marker='P', ls='none', label='memorizer, ridge λ 1e-6'),
                  Line2D([], [], color='#d95f02', marker='X', ls='none', label='memorizer, ridge λ 1e-2')],
         fontsize=7, frameon=False, loc='center right')

a = ax[1, 0]
for run, rr in runs.items():
    if rr[0]['kind'] != 'a+b, decay 1':
        continue
    color, marker, _ = SH[rr[0]['share']]
    st = [max(r['step'], 100) for r in rr]
    a.plot(st, [r['P_cN'] for r in rr], color=color, marker=marker, ms=3, lw=1.3, mec='white', mew=.3)
    a.plot(st, [r['P_cS'] for r in rr], color=color, lw=1.1, ls='--')
a.axhline(.5, color='0.4', lw=.8, ls=(0, (1, 2)))
a.set(xscale='log', ylim=(-.02, 1.02), xlabel='update (step 0 drawn at 100)',
      ylabel='fraction of the set ranked below a+b', title='The correct label climbs past N (solid), then past S (dashed)')
a.legend(handles=[Line2D([], [], color='0.4', ls='-', label='P(a+b over N)'), Line2D([], [], color='0.4', ls='--', label='P(a+b over S)')],
         fontsize=7.5, frameon=False, loc='upper left')

a = ax[1, 1]
for r in lad:
    if r['step'] >= 300:
        color, marker, _ = SH[r['share']]
        a.scatter(gap[(r['run_id'], int(r['step']))], r['P_SN'], color=color, marker=marker, s=14, alpha=.85, edgecolor='white', lw=.3)
a.set(xlabel='S − N gap in units of σ (max-based; e2 audit cycle 4)', ylabel='P(S over N)',
      title='The rank measure and the max-based gap agree\n(every ladder checkpoint from 300 updates on; colours as top left)')
for a in ax.flat:
    a.grid(True, color='0.92', lw=.6)
fig.suptitle('Width 128, the 483 common never-trained pairs of the e2 ladder: operand-label preference without logit scale', fontsize=11)
fig.text(.01, -.01, 'P(A over B) = mean over a in A, b in B of 1[z_a > z_b] (+ 0.5 for ties; none occurred), per pair, then averaged over pairs. '
         'Identical per pair under positive scaling and common offsets (checked at every row). S, D, N: e2 sets from the frozen 30% and 50% '
         'training sets; no set is empty. Pairs and their transposes are not independent; seeds are the replicate unit. Memorizer: step-0 '
         'features with a readout fitted in closed form to each share\'s training pairs (e2 audit cycle 4). Sources: '
         'research/e3/audit/results/preference.csv, preference-matched-ce.csv; research/e2/audit/results/margins-check.csv.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
