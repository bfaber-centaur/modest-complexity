#!/usr/bin/env python3
"""Figures of the generalization lane, drawn from committed run outputs only.

length    accuracy against program length 1-6 for the recovered r1 tied/inter and tied/final
          checkpoints (QUESTION.md first-figure plan), with reference systems and chance.

uv run --locked python research/composition-c2/generalization/gen_figures.py length
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGS = HERE/'figures'
FROZEN = HERE/'runs'/'c2-gen-frozen-orig-a2'
SEED_STYLE = {51200: ('#2a78d6', 'o'), 51201: ('#eb6834', 's'), 51202: ('#1baf7a', '^')}  # validated categorical slots 1-3
GREY, INK = '#8a8a86', '#3d3d3a'
CHANCE = 1/8


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def length_figure():
    rows = read(FROZEN/'length.csv')
    refs = read(FROZEN/'references.csv')
    keep = [r for r in rows if r['subset'] in ('all', 'held-out programs')]
    FIGS.mkdir(exist_ok=True)
    cols = ('run_id', 'ckpt_step', 'condition', 'init_seed', 'length', 'subset', 'n_rows', 'n_programs', 'accuracy')
    with open(FIGS/'length.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols + ('source',))
        for r in keep:
            w.writerow([r[c] for c in cols] + [f'runs/{FROZEN.name}/length.csv'])
        for r in refs:
            w.writerow([r['system'], '', 'reference', '', r['length'], 'all', '', '', r['accuracy'], f'runs/{FROZEN.name}/references.csv'])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    lengths = range(1, 7)
    titles = dict(inter='tied/inter: shared transition and readout,\nreadout trained on h1, h2, h3',
                  final='tied/final: shared transition and readout,\nreadout trained on h3 only')
    for ax, cond in zip(axes, ('inter', 'final')):
        ax.axvspan(2.6, 3.4, color='#ecebe6', zorder=0)
        y = 0.30 if cond == 'inter' else 0.42  # clear of every plotted mark in that panel
        band = 'trained length\n48 of 64\nprograms' if cond == 'inter' else 'trained\nlength'  # final: fits under the peak
        ax.text(3, y if cond == 'inter' else 0.39, band, ha='center', va='bottom', fontsize=7.5, color=INK)
        note = 'prefixes labelled\nthrough the\nshared readout' if cond == 'inter' else 'readout never\ntrained at h1, h2'
        ax.text(1.5, y, note, ha='center', va='bottom', fontsize=7.5, color=INK)
        ax.text(5, y, 'never trained:\nnew syntax,\nnew functions', ha='center', va='bottom', fontsize=7.5, color=INK)
        ax.axhline(CHANCE, color=GREY, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(6.15, CHANCE, 'chance', va='center', fontsize=7.5, color=INK)
        ref_one = [float(r['accuracy']) for r in refs if r['system'].startswith(('oracle', 'P2'))]
        assert all(a == 1.0 for a in ref_one)
        ax.plot(lengths, [1.0]*6, color=GREY, lw=1.2, ls=(0, (1, 2)), zorder=1,
                label='references at 1.0: oracle, P2 iterated\n(left panel also: own one-step map re-encoded)')
        if cond == 'final':
            for i, seed in enumerate(SEED_STYLE):
                own = [float(r['accuracy']) for r in refs if r['system'] == f'own one-step map re-encoded (r1-comp-rnn-final-s{seed}-a1)']
                ax.plot(lengths, own, color=GREY, lw=0.8, alpha=0.8, zorder=1,
                        label='own one-step map re-encoded (per seed)' if i == 0 else None)
        for seed, (c, m) in SEED_STYLE.items():
            sel = [r for r in keep if r['condition'] == cond and int(r['init_seed']) == seed]
            acc = [float(next(r for r in sel if int(r['length']) == L and r['subset'] == 'all')['accuracy']) for L in lengths]
            held = float(next(r for r in sel if r['subset'] == 'held-out programs')['accuracy'])
            ax.plot(lengths, acc, color=c, marker=m, ms=6, lw=2, label=f'init {seed}, all programs', zorder=3)
            ax.plot([3], [held], marker=m, ms=9, mfc='white', mec=c, mew=1.6, ls='none', zorder=4,
                    label=f'init {seed}, length 3 held-out only')
        ax.set_title(titles[cond], fontsize=9.5, color=INK, loc='left')
        ax.set_xticks(list(lengths)); ax.set_xlim(0.7, 6.6); ax.set_ylim(0, 1.04)
        ax.set_xlabel('program length (ops applied)', fontsize=9)
        ax.grid(axis='y', color='#e4e3dd', lw=0.6); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel('final-answer accuracy, all 8·4^L rows', fontsize=9)
    h, l = axes[1].get_legend_handles_labels()
    order = [l.index(x) for x in l if x.startswith('init')] + [l.index(x) for x in l if not x.startswith('init')]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc='lower center', ncol=3, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Frozen r1 RNNs (ckpt-10000, original task ops51000-ls51100) on programs of lengths 1-6; exhaustive rows, no fitting',
                 fontsize=9.5, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.13, 1, 0.97))
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'length.{ext}', dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('which', choices=('length',))
    args = ap.parse_args()
    dict(length=length_figure)[args.which]()


if __name__ == '__main__':
    main()
