#!/usr/bin/env python3
"""Final-checkpoint held-out rank statistic (heldout_rank.py output) for every run,
by width, with the no-decay and shuffled-label controls at width 128.

uv run --locked --group analysis python research/audit/plot_rank_by_width.py OUT_STEM RANK.jsonl ...
"""
import json
from pathlib import Path
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
runs = [json.loads(s) for f in sys.argv[2:] for s in Path(f).read_text().splitlines() if s.strip()]
SEED_MARK = {30000: 'o', 30001: 's', 30002: '^'}
COND = {'add': ('addition, decay 1', '#2a7bb6', 0), 'add0': ('addition, no decay', '#c9a227', -.07),
        'shuf': ('shuffled labels, decay 1 (probe: a+b)', '#9e3d22', .07)}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout='constrained')
seen = set()
table = []
for r in runs:
    cond, width, seed = re.match(r'e1-\w+?-(add0|add|shuf)-w(\d+)-i(\d+)-a\d', r['run']).groups()
    width, seed = int(width), int(seed)
    last = max(r['rows'], key=lambda z: z['step'])
    label, color, dx = COND[cond]
    x = width * 2**(dx + .035*(seed - 30001))  # small per-seed offset so equal values stay visible
    for ax, key in zip(axes, ('true_rank_among_absent_mean', 'true_top_among_absent')):
        ax.plot(x, last[key], SEED_MARK[seed], color=color, ms=6, mfc='none' if cond != 'add' else color,
                label=label if (cond, key) not in seen else None)
        seen.add((cond, key))
    table.append((r['run'], last['step'], r['pairs_with_probe_absent'], last['true_rank_among_absent_mean'],
                  last['true_top_among_absent'], r['chance_top_among_absent']))
add_chance = next(r['chance_top_among_absent'] for r in runs if r['trained_on'] == 'addition')
axes[0].axhline(.5, color='0.55', lw=.8, ls=(0, (1, 2)))
axes[1].axhline(add_chance, color='0.55', lw=.8, ls=(0, (1, 2)))
for ax, ylabel, ylim in ((axes[0], 'mean normalized rank of a+b among\nabsent labels (0 = top)', (0, .7)),
                         (axes[1], 'fraction of held-out pairs where a+b\ntops all absent labels', (0, .45))):
    ax.set_xscale('log', base=2); ax.set_xticks([16, 32, 64, 128, 256, 512], ['16', '32', '64', '128', '256', '512'])
    ax.set(xlabel='width (markers offset slightly by seed and condition)', ylabel=ylabel, ylim=ylim)
    ax.grid(True, color='0.9', lw=.6)
for seed, m in SEED_MARK.items():
    axes[1].plot([], [], m, color='0.3', mfc='none', label=f'init seed {seed}')
axes[1].legend(fontsize=7, frameon=False, loc='upper left')
fig.suptitle('Held-out pairs, final checkpoint (step 50,000): rank of the addition label among labels absent from '
             'training co-occurrence', fontsize=9.5)
fig.text(.01, -.02, 'One marker per run. Absent: never paired with either operand of the pair among the run\'s training '
         'targets. Dotted: no preference (0.5 left; mean 1/#absent for addition runs, right; the shuffled run\'s own '
         'reference is in the table). Shuffled run uses only pairs whose a+b is absent.', fontsize=6.5, color='0.3',
         ha='left', va='top', wrap=True)
out.parent.mkdir(parents=True, exist_ok=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
with open(f'{out}.csv', 'w') as f:
    f.write('run_id,step,pairs_used,rank_among_absent_mean,top_among_absent,chance_top_among_absent\n')
    for row in sorted(table):
        f.write(','.join(map(str, row)) + '\n')
