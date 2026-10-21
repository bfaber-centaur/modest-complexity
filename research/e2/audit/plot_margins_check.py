#!/usr/bin/env python3
"""Figure for QUESTION-4.md from margins_check.py output: the S - N gap and the margin over
N in logits and in units of each pair's wrong-label spread, and a training-free memorizer
at the same nested shares. Also writes the values at matched training cross-entropy.

uv run --locked --group analysis python research/e2/audit/plot_margins_check.py OUT_STEM MARGINS.csv RIDGE.csv MATCHED.csv
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
num = lambda r: {k: (v if k in ('run_id', 'share') else float(v)) for k, v in r.items()}
rows = [num(r) for r in csv.DictReader(open(sys.argv[2]))]
ridge = [num(r) for r in csv.DictReader(open(sys.argv[3]))]
SH = {'30%': ('#9a9a9a', 'o', 299), '40%': ('#9ecae1', 's', 359), '45%': ('#4292c6', '^', 417), '50%': ('#08519c', 'D', 478)}
COLS = ('median_gap_S_over_N', 'sigma_gap_S_over_N', 'within_sigma_gap_S_over_N', 'median_margin_N', 'sigma_margin_N',
        'within_sigma_margin_N', 'median_margin_S', 'sigma_margin_S', 'median_sigma', 'common_acc')
runs = {run: sorted([r for r in rows if r['run_id'] == run], key=lambda r: r['step']) for run in dict.fromkeys(r['run_id'] for r in rows)}
seed = lambda run: int(run.split('-i')[1][:5])


def at_ce(rr, col, ce):
    """Linear in log training cross-entropy between the checkpoints that first bracket `ce`."""
    for x, y in zip(rr, rr[1:]):
        if x['train_ce'] >= ce > y['train_ce']:
            t = (np.log(ce) - np.log(x['train_ce']))/(np.log(y['train_ce']) - np.log(x['train_ce']))
            return x[col] + t*(y[col] - x[col])
    return None


matched = []
for ce in (1e-2, 1e-3, 1e-4):
    for run, rr in runs.items():
        matched.append(dict(run_id=run, share=rr[0]['share'], init_seed=seed(run), train_ce=ce,
                            **{c: round(at_ce(rr, c, ce), 6) for c in COLS}))
with open(sys.argv[4], 'w', newline='') as f:
    w = csv.DictWriter(f, list(matched[0])); w.writeheader(); w.writerows(matched)

fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.5), layout='constrained')
panels = ((ax[0, 0], 'median_gap_S_over_N', 'S − N gap, logits\n(strongest S label minus strongest N label, median over pairs)',
           'In logits the gap grows with training, more slowly at higher share'),
          (ax[0, 1], 'sigma_gap_S_over_N', 'S − N gap in units of σ\n(σ = s.d. of the pair\'s 30 wrong-label logits)',
           'In units of the logit spread it stays near 2σ at every share'),
          (ax[1, 0], 'sigma_margin_N', 'margin over N in units of σ\n(correct label minus strongest N label)',
           'The margin over unrelated labels, in units of σ, is graded by share'))
for a, col, ylab, title in panels:
    for run, rr in runs.items():
        color, marker, _ = SH[rr[0]['share']]
        rr = [r for r in rr if r['step'] >= 300]
        a.plot([r['step'] for r in rr], [r[col] for r in rr], color=color, marker=marker, ms=3.5, lw=1.4,
               mec='white', mew=.3)
    a.set(xscale='log', xlabel='update', ylabel=ylab, title=title)
ax[0, 1].set_ylim(0, 2.6)
ax[1, 0].axhline(0, color='0.4', lw=.8, ls=(0, (1, 2)))
ax[0, 0].legend(handles=[Line2D([], [], color=c, marker=m, ms=5, label=f'{s} training share ({n} pairs), 3 init seeds')
                         for s, (c, m, n) in SH.items()], fontsize=7.5, frameon=False, loc='upper left')

a = ax[1, 1]
x = [n for _, _, n in SH.values()]
base = {(seed(r['run_id']), c): r[c] for r in matched if r['share'] == '30%' and r['train_ce'] == 1e-4 for c in COLS}
series = [('trained runs, at training cross-entropy 1e-4', '#333333', 'o',
           lambda s, c: [r[c]/base[(r['init_seed'], c)] for r in matched if r['share'] == s and r['train_ce'] == 1e-4])]
for lam, mk in ((1e-6, 'D'), (1e-2, 'v')):
    rb = {(int(r['init_seed']), c): r[c] for r in ridge if r['share'] == '30%' and r['ridge'] == lam for c in COLS if c in r}
    series.append((f'memorizer: step-0 features, readout fitted\nin closed form (ridge λ = {lam:g})', '#d95f02', mk,
                   lambda s, c, lam=lam, rb=rb: [r[c]/rb[(int(r['init_seed']), c)] for r in ridge if r['share'] == s and r['ridge'] == lam]))
handles = []
for label, color, mk, get in series:
    for col, ls, fill in (('median_gap_S_over_N', '-', True), ('sigma_gap_S_over_N', '--', False)):
        vals = [get(s, col) for s in SH]
        for xi, v in zip(x, vals):
            a.scatter(np.full(len(v), xi) + (4 if not fill else -4)*(mk != 'o'), v, color=color, marker=mk, s=16,
                      facecolor=color if fill else 'none', lw=.8, zorder=3)
        a.plot(x, [np.mean(v) for v in vals], color=color, ls=ls, lw=1.4 if mk != 'v' else .9)
    handles.append(Line2D([], [], color=color, marker=mk, ls='none', label=label))
handles += [Line2D([], [], color='0.3', ls='-', label='gap in logits (filled)'),
            Line2D([], [], color='0.3', ls='--', label='gap in units of σ (open)')]
a.axhline(1, color='0.6', lw=.8)
a.set(xticks=x, xticklabels=[f'{s}\n{n}' for s, (_, _, n) in SH.items()], xlabel='training share and pairs (nested sets)',
      ylabel='S − N gap relative to the same init seed at 30%', ylim=(.45, 1.25),
      title='Dilution check: with more pairs per operand, a memorizer\'s\ngap shrinks in logits but not in units of σ')
a.legend(handles=handles, fontsize=7, frameon=False, loc='lower left')
for a in ax.flat:
    a.grid(True, color='0.92', lw=.6)
fig.suptitle('Width 128, the 483 common never-trained pairs: is the operand-label gap a competitor or the logit spread?', fontsize=11)
fig.text(.01, -.01, 'Independent recomputation of research/e2/dynamics/margins.py (all 197 rows of margins_boundary.csv agree within '
         '5e-10). For each pair the margin over S equals the margin over N minus the S − N gap exactly, so "correct needs the margin '
         'over N to exceed the gap" is arithmetic. Memorizer: the step-0 U, V of each init seed with W fitted to one-hot a+b on that '
         "share's training pairs; held-out accuracy ≤ 0.014 at every share; it fits 80–85% of its 478 training pairs at 50%. "
         'Sources: research/e2/audit/results/margins-check.csv, margins-ridge.csv, margins-matched-ce.csv.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
