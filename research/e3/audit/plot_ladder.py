#!/usr/bin/env python3
"""Figure for research/e3/audit/QUESTION-5.md from ladder_check.py and dynamics_check.py output.

uv run --locked --group analysis python research/e3/audit/plot_ladder.py OUT_STEM RESULTS_DIR
"""
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

out, res = Path(sys.argv[1]), Path(sys.argv[2])
rows = [r for r in csv.DictReader(open(res/'ladder-check.csv'))]
lc = json.loads((res/'ladder-check.json').read_text())
rep = {i: json.loads((res/f'dynamics-replay-i{i}.json').read_text()) for i in (30000, 30001)}
sens = json.loads((res/'dynamics-sensitivity.json').read_text())
track = json.loads((res/'dynamics-track-i30001.json').read_text())
step = json.loads((res/'dynamics-onestep-i30001.json').read_text())
one = json.loads((res/'dynamics-track-step-i30001.json').read_text())
LAD = {41001: '#1b9e77', 41002: '#d95f02', 41003: '#7570b3', 31000: '#666666'}
INIT = {30000: '-', 30001: '--'}
MK = {30000: 'o', 30001: 's'}

fig, ax = plt.subplots(2, 2, figsize=(13, 10), layout='constrained')

a = ax[0, 0]
for seed, color in LAD.items():
    for init, ls in INIT.items():
        rr = sorted([r for r in rows if int(r['data_seed']) == seed and float(r['share']) == .4 and int(r['init_seed']) == init],
                    key=lambda r: int(r['step']))
        x, y = [int(r['step']) for r in rr], [float(r['margin_N_sigma']) for r in rr]
        a.plot(x, y, color=color, ls=ls, lw=2 if seed == 41002 else 1.2, marker=MK[init], ms=4 if len(x) > 1 else 7, mec='white', mew=.4)
a.axhline(0, color='0.5', lw=.8, ls=(0, (1, 2)))
a.set(xscale='log', xlabel='update', ylabel='median margin of a+b over N (σ units), own evaluation pairs',
      title='40% runs: 41002 leads for init 30001 from 5k,\nfor init 30000 only from 30k (31000: 50k only)')

a = ax[0, 1]
fixed = lc['rule4']['fixed_pairs']['values']
noise = lc['post_hoc']['subset53_sd_of_median_sigma_margin']
final = {(int(r['data_seed']), int(r['init_seed'])): float(r['margin_N_sigma']) for r in rows
         if float(r['share']) == .4 and int(r['step']) == 50000}
for (seed, init), own in final.items():
    sh = fixed[f'{seed}/{init}']['margin_N_sigma']
    a.plot([0, 1], [own, sh], color=LAD[seed], ls=INIT[init], lw=2 if seed == 41002 else 1.2, marker=MK[init], ms=7, mec='white', mew=.5)
    a.errorbar(1.04 + .02*list(LAD).index(seed) + .01*(init - 30000), sh, yerr=noise[f'{seed}/{init}'], color=LAD[seed], lw=.9, capsize=2)
a.set(xticks=[0, 1], xticklabels=['own evaluation pairs\n(479–483, differ by ladder)', 'the 53 pairs outside all four\n50% training sets'],
      xlim=(-.25, 1.3), ylabel='median margin of a+b over N (σ units), update 50,000',
      title='Holding the pairs fixed, 41002 is no longer ahead for either init\n(bars, post hoc: ±1 SD of a 53-pair median over random subsets of own pairs)')

a = ax[1, 0]
ratio = lc['post_hoc']['criterion_ratio']
q = lc['post_hoc']['null_ratio_quantiles']
KEYS = ['acc', 'P_yN', 'P_yS', 'P_SN', 'margin_N_sigma', 'rank_absent']
NAMES = ['accuracy', 'P(a+b>N)', 'P(a+b>S)', 'P(S>N)', 'σ-margin', 'own-absent\nrank']
SH = {.3: '#bdbdbd', .4: '#d39a1f', .45: '#b03a8c', .5: '#2a5fb6'}
a.axhspan(q['0.05'], q['0.95'], color='0.92', lw=0)
a.axhspan(q['0.25'], q['0.75'], color='0.84', lw=0)
a.axhline(1, color='0.3', lw=.9, ls='--')
for j, k in enumerate(KEYS):
    for n, (share, c) in enumerate(SH.items()):
        a.plot(j + (n - 1.5)*.13, ratio[f'{k} {share}'], 'o', color=c, ms=7, mec='white', mew=.5)
a.text(5.45, 1.04, "lane's criterion", fontsize=7.5, color='0.3', ha='right')
a.set(yscale='log', yticks=[.2, .3, .5, 1, 1.5, 2], yticklabels=['0.2', '0.3', '0.5', '1', '1.5', '2'], xticks=range(6), xticklabels=NAMES, xlim=(-.5, 5.5),
      ylabel='range of the three e3 ladder means ÷ largest init difference',
      title=f"Ladder against init: 7 of 24 cells cross 1; chance per cell is {lc['rule4']['P0']:.2f}\n(bands: 50% and 90% of the no-ladder-effect distribution)")
a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
a.legend(handles=[Line2D([], [], marker='o', ls='none', color=c, label=f'{round(100*s)}%') for s, c in SH.items()],
         fontsize=8, frameon=False, loc='lower left', ncol=4)

a = ax[1, 1]
for i, mk in MK.items():
    e = rep[i]['early']
    st = sorted(e, key=int)
    a.plot([int(s) for s in st], [e[s]['max_abs_diff']/(e[s]['tolerance']/1e-9) for s in st], marker=mk, color='#2166ac', ls=INIT[i], lw=1.2,
           label=f'mine from the independent init against the run (init {i})')
scale = rep[30001]['late']['max_abs_weight']
tr = track['mine_vs_other']
a.plot([r['step'] for r in tr], [r['max_abs_diff']/scale for r in tr], color='#b2182b', marker='s', ms=4, lw=1.2,
       label='mine from 30k against the shared code resumed\nfrom 30k (= the run: bitwise at 50k; init 30001)')
a.plot(50000, rep[30000]['late']['max_abs_diff']/rep[30000]['late']['max_abs_weight'], marker='o', color='#b2182b', ms=7, ls='none',
       label='mine from 30k against the run at 50k (init 30000)')
a.plot(30001, one['mine_vs_other'][0]['max_abs_diff']/scale, marker='D', color='#b2182b', ms=6, ls='none',
       label='one update from 30k, mine against the shared code')
a.plot(30001, step['shared_rounding_of_logit_gradient_vs_shared_code_30001']['max_abs_diff']/scale, marker='D', color='0.2', ms=6, ls='none',
       label='one update with the shared code\'s rounding of\nsoftmax − onehot, against the shared code')
a.plot(50000, sens['mine_vs_mine_pairs_permuted']['max_abs_diff']/scale, marker='P', color='0.2', ms=9, ls='none',
       label='mine against mine with the pairs reordered (init 30001)')
a.axhline(1e-9, color='0.5', lw=.8, ls=(0, (1, 2)))
a.text(330, 1.6e-9, 'pre-stated early tolerance', fontsize=7.5, color='0.35')
a.set(xscale='log', yscale='log', ylim=(1e-17, 30), xlabel='update', ylabel='largest |weight difference| ÷ largest |weight|',
      title='Replays of the 41002 40% pair: exact to 2k. Late, the two codes round 1 − p(a+b) ≈ 1e-5\ndifferently in the logit gradient, and the gap grows to 7% of the weight norm')
a.legend(fontsize=7, frameon=False, loc='upper left')
for a in ax.flat:
    a.grid(True, color='0.92', lw=.6)
ax[0, 0].legend(handles=[Line2D([], [], color=c, lw=2, label=f'data seed {s}' + (' (e2 ladder)' if s == 31000 else '')) for s, c in LAD.items()] +
                [Line2D([], [], color='0.3', ls=ls, marker=MK[i], label=f'init {i}') for i, ls in INIT.items()],
                loc='upper left', ncol=2, fontsize=8, frameon=False)
fig.suptitle('e3 dynamics, audit investigation 5: is the 41002 lead at 40% a ladder effect?', fontsize=11)
fig.text(.01, -.01, 'All values from my own forward pass on the published checkpoints (research/e3/dynamics/runs and the e1/e2 runs of the 31000 ladder); '
         'they match the lane\'s ladders.csv to 4e-16 in all 200 rows. σ-margin: median over pairs of (a+b logit − max over N) / standard deviation of '
         'the 30 wrong-label logits; N, S per ladder (D6). Two inits per cell; pairs within a ladder share operands and labels and are not replicates. '
         'Bottom right: my replays use research/audit/independent.py; the shared code is experiment.py at 72cfb5d resumed in scratch (audit probe, no run ID). '
         'Sources: research/e3/audit/results/ladder-check.csv, ladder-check.json, dynamics-replay-i3000{0,1}.json, dynamics-sensitivity.json, '
         'dynamics-track-i30001.json, dynamics-track-step-i30001.json, dynamics-onestep-i30001.json.',
         fontsize=7, color='0.3', ha='left', va='top', wrap=True)
for ext in ('png', 'svg'):
    fig.savefig(f'{out}.{ext}', dpi=140, bbox_inches='tight')
