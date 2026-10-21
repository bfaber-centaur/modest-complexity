#!/usr/bin/env python3
"""Figure from heldout_rank.py output: held-out true-label standing among the labels
that never co-occur with either operand in training, at each published checkpoint.

uv run --locked --group analysis python research/audit/plot_heldout_rank.py \
    research/audit/results/preview-1-heldout-rank.jsonl research/audit/figures/heldout-rank
"""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {32: '#d36b1f', 128: '#3a9a5b', 512: '#5a3f9e'}  # report_map.py width colors
src, out = Path(sys.argv[1]), Path(sys.argv[2])
runs = [json.loads(s) for s in src.read_text().splitlines() if s.strip()]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout='constrained', sharex=True)
for r in runs:
    width = int(r['run'].split('-w')[1].split('-')[0])
    rows = [z for z in r['rows'] if z['step'] > 0]
    xs = [z['step'] for z in rows]
    for ax, key in zip(axes, ('true_rank_among_absent_mean', 'true_top_among_absent')):
        ax.plot(xs, [z[key] for z in rows], 'o-', color=COLORS.get(width, 'k'), ms=4, lw=1.2,
                label=f"{r['run']} (step 0: {r['rows'][0][key]:.3f})")
axes[0].axhline(.5, color='0.55', lw=.8, ls=(0, (1, 2)))
axes[1].axhline(runs[0]['chance_top_among_absent'], color='0.55', lw=.8, ls=(0, (1, 2)))
axes[0].set_ylim(0, 1); axes[0].set_ylabel('mean normalized rank of true label\namong absent labels (0 = top)')
axes[1].set_ylim(0, .5); axes[1].set_ylabel('fraction of held-out pairs where the\ntrue label tops all absent labels')
for ax in axes:
    ax.set_xscale('log'); ax.set_xlabel('update step (log scale; published checkpoints only)')
    ax.grid(True, color='0.9', lw=.6)
axes[0].legend(fontsize=6.5, frameon=False, loc='lower left')
fig.suptitle('Held-out pairs (cal + audit, 662), init seed 30000: true label vs labels absent from training co-occurrence',
             fontsize=9.5)
fig.text(.01, -.02, 'Absent label: never paired with either operand of the held-out pair in a training example; the true '
         'label is always absent in modular addition. Dotted: no preference among absent labels (0.5 left; mean of 1/#absent right). Markers are the '
         'only published checkpoints; lines join them and do not show intermediate values. Step 0 values in legend.',
         fontsize=6.5, color='0.3', ha='left', va='top', wrap=True)
out.parent.mkdir(parents=True, exist_ok=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
