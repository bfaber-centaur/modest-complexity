#!/usr/bin/env python3
"""Figures for the e3 continuation runs from continuation.py's tables (results/checkpoints.csv,
results/dense.csv). One column per source checkpoint, no pooling over seeds, no smoothing.
In every cell the narrow left panel is what the edit did (the 50k source checkpoint, then the
edited weights at update 0, before any update) and the wide right panel is what training did
afterwards (continuation updates 10..20,000 on a log axis; update 0 is only in the left panel).

uv run --locked --group analysis python research/e3/mechanisms/plot_continuation.py [--results DIR] [--out DIR]
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = {'keep': ('A unchanged', '#555555'), 'filt': ('B dominant-frequency filter', '#0072B2'),
        'sham': ('C input-energy-matched sham', '#D55E00')}
SEEDS = (30000, 30001, 30002)


def read(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def series(rows, arm, seed, key):
    rs = [r for r in rows if r['arm'] == arm and r['init_seed'] == seed and r.get('phase', 'continuation') == 'continuation']
    rs.sort(key=lambda r: r['step'])
    return [r['step'] for r in rs], [r[key] for r in rs]


def source_value(ck, arm, seed, key):
    return next(r[key] for r in ck if r['arm'] == arm and r['init_seed'] == seed and r['phase'] == 'source')


def figure(ck, dense, panels, title, caption, path):
    fig = plt.figure(figsize=(15, 2.35*len(panels) + 1.4))
    outer = fig.add_gridspec(len(panels), len(SEEDS), wspace=.28, hspace=.42, left=.07, right=.99, top=.91,
                             bottom=.07 + .012*caption.count('\n'))
    for i, (key, label, source, opts) in enumerate(panels):
        for j, seed in enumerate(SEEDS):
            inner = outer[i, j].subgridspec(1, 2, width_ratios=(1, 4), wspace=.05)
            ea, ta = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            for arm, (name, color) in ARMS.items():
                tab = dense if source == 'dense' else ck
                x, y = series(tab, arm, seed, key)
                edit = opts.get('edit_key', key)
                before = source_value(ck, arm, seed, edit)
                after = series(ck, arm, seed, edit)[1][0]
                ea.plot([0, 1], [before, after], '-o', color=color, ms=4, lw=1.4)
                keep = [k for k, s in enumerate(x) if s > 0]
                ta.plot([x[k] for k in keep], [y[k] for k in keep], '-' if source == 'dense' else '-o',
                        color=color, lw=1.2, ms=3, label=name)
                if 'dense_overlay' in opts:
                    xd, yd = series(dense, arm, seed, opts['dense_overlay'])
                    ta.plot(xd[1:], yd[1:], '-', color=color, lw=.6, alpha=.5)
                if 'second' in opts:
                    x2, y2 = series(ck, arm, seed, opts['second'])
                    ta.plot(x2[1:], y2[1:], '--', color=color, lw=1)
            ta.set_xscale('log')
            ta.set_xlim(8, 25000)
            ea.set_xlim(-.4, 1.4)
            ea.set_xticks([0, 1], ['50k', 'edit'], fontsize=8)
            for ax in (ea, ta):
                if opts.get('log'):
                    ax.set_yscale('log')
                if 'ylim' in opts:
                    ax.set_ylim(*opts['ylim'])
                if 'ref' in opts:
                    ax.axhline(opts['ref'], color='k', lw=.7, ls=':')
                ax.tick_params(labelsize=8)
                ax.grid(alpha=.25, lw=.5)
            if 'ylim' not in opts:  # the two panels of a cell share one y axis
                lo, hi = min(ea.get_ylim()[0], ta.get_ylim()[0]), max(ea.get_ylim()[1], ta.get_ylim()[1])
                ea.set_ylim(lo, hi)
                ta.set_ylim(lo, hi)
            ta.set_yticklabels([])
            if j == 0:
                ea.set_ylabel(label, fontsize=9)
            if i == 0:
                ta.set_title(f'source init seed {seed}', fontsize=10)
                ea.set_title('the edit', fontsize=8)
            if i == len(panels) - 1:
                ta.set_xlabel('continuation updates (log; update 0 in the left panel)', fontsize=8)
    handles = [plt.Line2D([], [], color=c, marker='o', ms=4, lw=1.4) for _, c in ARMS.values()]
    fig.legend(handles, [n for n, _ in ARMS.values()], loc='upper center', ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(.53, .975))
    fig.suptitle(title, fontsize=12, y=.995)
    fig.text(.07, .01, caption, fontsize=8, va='bottom')
    for ext in ('png', 'svg'):
        fig.savefig(path.with_suffix('.' + ext), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=HERE/'results')
    ap.add_argument('--out', type=Path, default=HERE/'figures')
    args = ap.parse_args()
    ck, dense = read(args.results/'checkpoints.csv'), read(args.results/'dense.csv')
    args.out.mkdir(parents=True, exist_ok=True)
    common = ('Each 50k source is a failed 30% width-128 run (299 training pairs, wd 1). All nine branches: fresh Adam '
              '(moments zero, update count from 1), the original recipe, 20,000 updates.\n'
              'Markers: published checkpoints (updates 10, 30, 100, 300, 1000, 2000, 5000, 10000, 15000, 20000 and the first update '
              'with train accuracy >= .99); lines without markers: the every-10 metric log. No smoothing.\n'
              'Dotted: chance 1/31, no preference 0.5 or zero margin. Run IDs e3-me-<keep|filt|sham>-w128-i<seed>-a1.')
    figure(ck, dense, [
        ('common_acc', 'held-out accuracy\n(483 common pairs)', 'ck', dict(ylim=(0, 1), ref=1/31, dense_overlay='audit_acc')),
        ('train_acc', 'train accuracy', 'dense', dict(ylim=(0, 1.02))),
        ('train_ce', 'train cross-entropy\n(log)', 'dense', dict(log=True)),
        ('P_cN', 'P(a+b over N)\n(common pairs)', 'ck', dict(ylim=(0, 1), ref=.5)),
        ('dist_from_update0', 'relative distance\nsolid: from update 0\ndashed: from 50k', 'ck',
         dict(second='dist_from_source', edit_key='dist_from_source')),
    ], 'What the edit did (left of each cell) and what training did afterwards (right)',
        common + '\nTop row, thin lines: accuracy on the run\'s own 544 audit pairs (they include the 483), every 10 updates. '
        'Bottom row, left panels: distance of the update-0 weights from the 50k source (the edit size).',
        args.out/'continuation')
    figure(ck, dense, [
        ('input_concentration', 'input concentration\n(mean top share)', 'ck', dict(ylim=(0, 1.02))),
        ('share_at_source_dominant', "share at the unit's\n50k dominant freq.", 'ck', dict(ylim=(0, 1.02))),
        ('dominant_as_at_update0', 'units with update-0\ndominant freq.', 'ck', dict(ylim=(0, 1.02))),
        ('P_SN', 'P(S over N)\n(common pairs)', 'ck', dict(ylim=(0, 1), ref=.5)),
        ('margin_N_sigma', '(a+b - max N) / sigma\nmedian, linear', 'ck', dict(ref=0)),
    ], 'Frequency structure and operand-label preference through the continuation',
        common + '\nShares are of each live unit\'s non-DC input energy (U and V). sigma = std of the pair\'s 30 wrong-label logits.',
        args.out/'structure')


if __name__ == '__main__':
    main()
