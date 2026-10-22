#!/usr/bin/env python3
"""First figure (QUESTION.md): final accuracy by arm in the main and grouped evaluations, and the
symbol and centroid error of the corrected h_2. Every base seed is shown; source rows go to
figures/correction.csv.

uv run --locked --group analysis python research/composition-c3/correction/corr_figure.py
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = {'main': HERE/'runs'/'c3-corr-main-a1', 'grouped': HERE/'runs'/'c3-corr-grouped-a1'}
CONDS = ('per-step readout', 'tied/final')
SEEDS = (51200, 51201, 51202)
ARMS = ('A', 'B', 'C', 'D', 'E')
ARM_LABELS = ('A\nnone', 'B\nmap→c(x)', 'C\nrecon.', 'D\ndecode→c', 'E\noracle ref.')
SEED_COLORS = dict(zip(SEEDS, ['#1b6ca8', '#d1495b', '#3a7d44']))
MARKERS = dict(zip(SEEDS, 'osD'))
OFFSET = dict(zip(SEEDS, (-0.18, 0., 0.18)))


def read(name, fname):
    return list(csv.DictReader((RUNS[name]/fname).open()))


def main():
    acc = {s: read(s, 'accuracy.csv') for s in RUNS}
    st = {s: read(s, 'states.csv') for s in RUNS}
    eval_prefixes = dict(main='fit', grouped='eval')  # main: the fit prefixes are the evaluated ones
    src = []

    def pick(table, **kw):
        out = [r for r in table if all(r[k] == str(v) for k, v in kw.items())]
        assert len(out) == 1, (kw, len(out))
        return out[0]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True)
    titles = ['Main: seen prefixes, new op 3\nfinal accuracy (filled: held-out programs,\nhollow: training programs)',
              'Grouped: adapter-held-out prefixes\n(not globally unseen states)\nfinal accuracy, same fill code',
              'Symbol of the corrected h$_2$\n(nearest fit centroid; filled: grouped\neval prefixes, hollow: main)',
              'Centroid error of the corrected h$_2$\n||ĥ − c(x)||² / mean centroid distance²\n(same fill code; log scale)']
    for r, cond in enumerate(CONDS):
        for c in range(4):
            ax = axes[r, c]
            ax.axvspan(3.5, 4.5, color='0.93', zorder=0)
            for seed in SEEDS:
                for i, arm in enumerate(ARMS):
                    x = i + OFFSET[seed]
                    pts = []
                    if c < 2:
                        split = ('main', 'grouped')[c]
                        for rows, filled in (('held', True), ('train', False)):
                            v = pick(acc[split], condition=cond, seed=seed, arm=arm, prefixes=eval_prefixes[split], rows=rows)
                            pts.append((float(v['accuracy']), filled))
                            src.append(dict(panel=f'{split} final accuracy', condition=cond, seed=seed, arm=arm,
                                            prefixes=v['prefixes'], rows=rows, n=v['n'], value=v['accuracy']))
                    else:
                        meas = ('symbol_acc', 'centroid_err')[c-2]
                        for split, filled in (('grouped', True), ('main', False)):
                            v = pick(st[split], condition=cond, seed=seed, arm=arm, prefixes=eval_prefixes[split])
                            val = float(v[meas])
                            if meas == 'centroid_err':
                                val = max(val, 1e-3)  # exact centroid (D, E) drawn at the floor
                            pts.append((val, filled))
                            src.append(dict(panel=meas, condition=cond, seed=seed, arm=arm, prefixes=v['prefixes'],
                                            rows='', n=v['n'], value=v[meas]))
                    # hollow markers are larger and drawn underneath, so equal values show as a ring around a dot
                    for y, filled in sorted(pts, key=lambda t: t[1]):
                        ax.scatter(x, y, s=26 if filled else 62, marker=MARKERS[seed], zorder=4 if filled else 3, linewidths=1.2,
                                   facecolors=SEED_COLORS[seed] if filled else 'white', edgecolors=SEED_COLORS[seed])
            if c == 3:
                ax.set_yscale('log'); ax.set_ylim(5e-4, 20)
                ax.text(3.5, 2.2e-3, 'D, E: exact (0),\ndrawn at 1e-3', ha='center', fontsize=7, color='0.35')
            else:
                ax.set_ylim(-0.03, 1.05); ax.axhline(1/8, color='black', lw=0.7, alpha=0.6)
                ax.text(-0.45, 1/8 + 0.015, 'chance', fontsize=7, color='0.35')
            ax.set_xticks(range(5)); ax.set_xticklabels(ARM_LABELS, fontsize=8)
            ax.grid(axis='y', color='0.9', lw=0.6); ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(titles[c], fontsize=9)
        axes[r, 0].set_ylabel(f'{cond}\n(frozen 10k, 3 seeds)', fontsize=10)
    handles = [plt.Line2D([], [], color=SEED_COLORS[s], marker=MARKERS[s], ls='', ms=7, label=f'base seed {s}') for s in SEEDS]
    handles += [plt.Line2D([], [], color='0.3', marker='o', ls='', ms=6, label='filled / hollow: see panel titles'),
                plt.Line2D([], [], color='0.3', marker='o', ls='', ms=9, markerfacecolor='white', label='(a ring around a dot: equal values)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('c3 learned correction at depth 2: main evaluation reuses the fit states (D2 overlap); '
                 'grouped evaluation holds out prefixes from the correction only', fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = HERE/'figures'; out.mkdir(exist_ok=True)
    fig.savefig(out/'correction.png', dpi=130); fig.savefig(out/'correction.svg')
    with (out/'correction.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(src[0])); w.writeheader(); w.writerows(src)
    print(out/'correction.png', len(src), 'rows')


if __name__ == '__main__':
    main()
