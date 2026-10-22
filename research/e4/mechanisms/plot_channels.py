#!/usr/bin/env python3
"""First e4 mechanisms figure from channels.py's table (results/channels.csv).

Each panel overlays the three source checkpoints (one line each, no pooling, no smoothing). B at
wd 1 and wd 0 use the dense replays to 5,000 and the published e3 checkpoints after that; A uses
its published e3 checkpoints. Update 0 (the edited start) is drawn at the left edge of the log axis.
Dotted verticals: the D5 decline window, updates 300 and 2,000.

uv run --locked --group analysis python research/e4/mechanisms/plot_channels.py [--results DIR] [--out DIR]
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ZERO = 5  # x position of update 0 on the log axis
BRANCHES = {('filt', 1.): ('B filter, wd 1', '#0072B2'), ('filt', 0.): ('B filter, wd 0', '#56B4E9'),
            ('keep', 1.): ('A unchanged, wd 1', '#777777')}
SEEDS = (30000, 30001, 30002)
PANELS = [
    ('common_acc', 'accuracy, 483 common pairs', {}),
    ('P_cS', 'P(a+b over S), common pairs', {}),
    ('margin_N_sigma', 'median (a+b − max N)/σ', {'ref': 0}),
    ('acc_ref_only', 'accuracy of the reference channel alone', {}),
    ('margin_S_ref', 'mean margin over S: reference channel', {'ref': 0}),
    ('margin_S_off', 'mean margin over S: off-reference channel', {'ref': 0}),
    ('margin_N_ref', 'mean margin over N: reference channel', {'ref': 0}),
    ('margin_N_off', 'mean margin over N: off-reference channel', {'ref': 0}),
    ('UV_ref_norm', '‖U, V‖ at reference frequencies', {}),
    ('UV_off_norm', '‖U, V‖ at other non-DC frequencies', {}),
    ('W_ref_norm', '‖W‖ at reference frequencies', {}),
    ('W_off_norm', '‖W‖ at other frequencies (non-DC)', {}),
    ('alignment_R', 'a+b-mode alignment R', {}),
    ('coherence', 'a+b-mode coherence', {}),
    ('gates_changed_from_update100', 'gates changed since update 100', {}),
    ('margin_S_offset', 'mean margin over S: label offset μW', {'ref': 0}),
]


def read(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
        for part in ('ref', 'off'):
            r[f'UV_{part}_norm'] = (r[f'U_{part}_norm']**2 + r[f'V_{part}_norm']**2)**.5
    return rows


def series(rows, arm, wd, seed, key):
    rs = [r for r in rows if r['arm'] == arm and r['wd'] == wd and r['init_seed'] == seed]
    dense = {r['step'] for r in rs if r['source'] == 'replay'}
    rs = sorted((r for r in rs if r['source'] == 'replay' or r['step'] not in dense), key=lambda r: r['step'])
    return [max(r['step'], ZERO) for r in rs], [r[key] for r in rs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', type=Path, default=HERE/'results')
    ap.add_argument('--out', type=Path, default=HERE/'figures')
    args = ap.parse_args()
    rows = read(args.results/'channels.csv')
    fig, axes = plt.subplots(4, 4, figsize=(16, 13.5))
    for ax, (key, label, opts) in zip(axes.flat, PANELS):
        for (arm, wd), (name, color) in BRANCHES.items():
            for i, seed in enumerate(SEEDS):
                x, y = series(rows, arm, wd, seed, key)
                ax.plot(x, y, color=color, lw=1.1, alpha=.85, label=name if i == 0 else None)
        ax.set_xscale('log')
        ax.set_xlim(ZERO*.9, 25000)
        ax.set_xticks([ZERO, 10, 100, 1000, 10000], ['0', '10', '100', '1k', '10k'])
        for t in (300, 2000):
            ax.axvline(t, color='k', lw=.6, ls=':')
        if 'ref' in opts:
            ax.axhline(opts['ref'], color='k', lw=.6)
        ax.set_title(label, fontsize=9.5)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('updates since the edit', fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('e4 mechanisms: B (filtered starts) at wd 1 and wd 0, with A (unedited memorizer), three sources each', fontsize=11)
    fig.text(.01, .005,
             'Channels: logits on the common pairs split exactly into label offset μW (μ = mean hidden activation over the 299 training pairs),'
             ' reference (h − μ)W_ref and off-reference (h − μ)W_off,\nW_ref = each unit\'s readout row at its update-0 dominant input frequency.'
             ' Margin over S (N): logit of a+b minus the mean logit over the operand-associated (unrelated) wrong labels; margins add across channels.'
             '\nAlignment R = Σ Re A_j / Σ|A_j| and coherence = Σ_k |Σ_{k_j = k} A_j| / Σ|A_j|, A_j = unit j\'s coefficient of the a+b mode at k_j'
             ' (all 961 pairs). Norms: Frobenius norms of the parts. Dotted verticals: updates 300 and 2,000.',
             fontsize=8, va='bottom')
    fig.tight_layout(rect=(0, .05, 1, .97))
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(args.out/f'channels.{ext}', dpi=130)
    print('wrote', args.out/'channels.png')


if __name__ == '__main__':
    main()
