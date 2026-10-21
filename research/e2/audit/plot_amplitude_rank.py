#!/usr/bin/env python3
"""Post hoc figure (added after QUESTION-2.md's diagnostic was computed): median relative
amplitude of the frequency groups' own-frequency a+b-c term (groups_check.py) against the
run's restricted rank at step 50,000 (rank_checks.py definition.jsonl, each run's own mask).

uv run --locked --group analysis python research/e2/audit/plot_amplitude_rank.py OUT_STEM DEFINITION.jsonl GROUPS.csv ...
"""
import csv
import json
from pathlib import Path
import re
import statistics as st
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
rank = {r['run']: r['rank_mean'] for r in map(json.loads, Path(sys.argv[2]).read_text().splitlines())}
rows = [r for f in sys.argv[3:] for r in csv.DictReader(open(f)) if r['step'] == '50000']
WIDTH = {16: '#c6dbef', 32: '#9ecae1', 64: '#6baed6', 128: '#3182bd', 256: '#08519c', 512: '#08306b'}
fig, ax = plt.subplots(figsize=(7, 4.8), layout='constrained')
table, seen = [], set()
for run in dict.fromkeys(r['run_id'] for r in rows):
    if run in seen or run not in rank:
        continue
    seen.add(run)
    g = [r for r in rows if r['run_id'] == run and r['kind'] == 'group']
    scale = float(next(r for r in rows if r['run_id'] == run and r['kind'] == 'none')['rms_logits_common'])
    amp = st.median(float(r['k_amplitude_common']) for r in g)/scale
    w = int(re.search(r'-w(\d+)-', run).group(1))
    if 'tf50' in run:
        style, label = dict(color='#2a7bb6', marker='D'), 'a+b labels, decay 1, 50% share (width 128)'
    elif 'add0' in run:
        style, label = dict(color='#c9a227', marker='s'), 'a+b labels, no decay (width 128)'
    elif 'shuf' in run:
        style, label = dict(color='#9e3d22', marker='X'), 'shuffled labels (width 128; rank of a+b, own mask)'
    else:
        style, label = dict(color=WIDTH[w], marker='o', mec='0.3', mew=.4), f'a+b labels, decay 1, 30% share, width {w}'
    ax.plot([amp], [rank[run]], ls='none', ms=7, label=label, **style)
    table.append((run, round(amp, 5), rank[run]))
handles, labels = ax.get_legend_handles_labels()
keep = dict(zip(labels, handles))
order = sorted(keep, key=lambda s: (int(m.group(1)) if (m := re.search(r'width (\d+)$', s)) else 10**4, 'no decay' not in s, s))
ax.legend([keep[k] for k in order], order, fontsize=7, frameon=False, loc='lower left')
ax.axhline(.5, color='0.4', lw=.8, ls=(0, (1, 2)))
ax.set(xscale='log', xlabel='own-frequency a+b−c amplitude of the frequency groups\n(median over groups, relative to the RMS of the centred logits; 483 common pairs)',
       ylabel='restricted rank of a+b at step 50,000\n(own mask, held-out pairs; 0.5 = no preference)', ylim=(-.02, .56),
       title='One point per run (all published width-128 controls and the 30% width map)')
ax.grid(True, color='0.92', lw=.6)
fig.text(.01, -.02, 'Post hoc: added after the QUESTION-2 diagnostic was computed. Sources: research/e2/audit/results/'
         'groups-check.csv, groups-by-width.csv, rank-checks/definition.jsonl.', fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=150, bbox_inches='tight')
with open(f'{out}.csv', 'w') as f:
    f.write('run_id,relative_amplitude,restricted_rank\n' + ''.join(','.join(map(str, t)) + '\n' for t in sorted(table)))
