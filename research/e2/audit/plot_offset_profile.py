#!/usr/bin/env python3
"""Figure for QUESTION.md: mean normalized rank of the label a+b+k among the pair's
absent labels, by offset k, one line per run (offset_profile.py output). k = 0 is
the restricted-rank statistic itself; k != 0 is ranked with a+b left out.

uv run --locked --group analysis python research/e2/audit/plot_offset_profile.py OUT_STEM PROFILE_50K.jsonl PROFILE_STEP0.jsonl
"""
import json
from pathlib import Path
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
load = lambda f: [json.loads(s) for s in Path(f).read_text().splitlines() if s.strip()]
final, init = load(sys.argv[2]), load(sys.argv[3])
WIDTH = {16: '#c6dbef', 32: '#9ecae1', 64: '#6baed6', 128: '#3182bd', 256: '#08519c', 512: '#08306b'}
ks = list(range(-15, 16))


def curve(r):
    by = {x['k']: x for x in r['rows']}
    return [by[k % 31]['rank_mean'] if k == 0 else by[k % 31]['rank_mean_without_sum'] for k in ks]


def parse(run):
    c, w, s, tf = re.match(r'e1f?\d?-\w+?-(add0|add|shuf)-w(\d+)-i(\d+)(-tf50)?', run).groups()
    return c + ('50' if tf else ''), int(w), int(s)


fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True, layout='constrained')
table = []
for r in sorted(final, key=lambda r: parse(r['run'])[1:]):  # legend in width order
    c, w, s = parse(r['run']); y = curve(r)
    table.append((r['run'], r['step'], *y))
    style = dict(lw=1, marker='o', ms=2.5)
    if c == 'add':
        axes[0].plot(ks, y, color=WIDTH[w], label=f'width {w}' if s == 30001 else None, **style)
        if w == 128:
            axes[1].plot(ks, y, color='0.6', label='30% share (e1)' if s == 30001 else None, **style)
    elif c == 'add50':
        axes[1].plot(ks, y, color='#2a7bb6', label='50% share (follow-up 2)' if s == 30001 else None, **style)
    else:
        color, label = ('#c9a227', 'no decay, width 128') if c == 'add0' else ('#9e3d22', 'shuffled labels, width 128')
        axes[2].plot(ks, y, color=color, label=label if s == 30001 else None, **style)
for r in init:
    y = curve(r); table.append((r['run'], r['step'], *y))
    axes[2].plot(ks, y, color='0.75', lw=.8, label='any run at step 0' if r is init[0] else None, zorder=0)
titles = ['addition, 30% share, decay 1 (18 runs)', 'width 128: 30% vs 50% share (3 seeds each)',
          'controls at step 50,000, and step 0']
for ax, t in zip(axes, titles):
    ax.axhline(.5, color='0.4', lw=.8, ls=(0, (1, 2)))
    ax.axvline(0, color='0.85', lw=.8, zorder=0)
    ax.set(title=t, xlabel='offset k of the probe label a+b+k (mod 31)', ylim=(-.02, .72), xticks=range(-15, 16, 5))
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.grid(True, color='0.92', lw=.6)
axes[0].set_ylabel('mean normalized rank among absent labels\n(0 = above all, 0.5 = no preference)')
fig.suptitle('Is the held-out preference specific to a+b? Rank of a+b+k among labels never seen with either operand '
             'in training (held-out pairs, step 50,000 unless marked)', fontsize=10)
fig.text(.01, -.03, 'One line per run (all published runs; k = 0 is the restricted-rank statistic, k != 0 ranked with a+b '
         'left out of the comparison set). Only pairs whose a+b and a+b+k are both absent: 287-662 pairs per point '
         '(30% share), 93-483 (50% share), 52-197 (shuffled). Source: research/e2/audit/figures/offset-profile.csv.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
out.parent.mkdir(parents=True, exist_ok=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
with open(f'{out}.csv', 'w') as f:
    f.write('run_id,step,' + ','.join(f'k{k}' for k in ks) + '\n')
    for row in sorted(table):
        f.write(','.join(map(str, row)) + '\n')
