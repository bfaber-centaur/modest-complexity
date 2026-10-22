#!/usr/bin/env python3
"""Channels and weight parts of the dynamics lane's decay-off runs (dynamics D13), from channels.py's
results/channels.csv. Analysis only: no new runs.

The six runs resume the e3 wd 1 runs of B at update 300 or 1,000 and continue at wd 0 to 20,000
updates. Comparators are the same start files' wd 1 and wd 0 runs (D7 dense replays to 5,000, then
the published e3 checkpoints). The question is the dynamics lane's: which channels and weight parts
are lost between updates 300 and 1,000 at wd 1, and do they stay absent while the decay-off runs
regrow to the wd 0 norm?

No reading was stated beforehand; the output is descriptive. One derived quantity was chosen after
seeing the table and is marked post hoc: the competitor ratio -margin_S_off / margin_S_ref, the
off-reference channel's mean margin over S against the reference channel's (scale-free).

Outputs: results/decay_off.csv (selected measures per run at updates 300, 1,000, 2,000, 5,000 and
20,000) and figures/decay_off.png/svg (one line per source, no pooling). Update 0 is not drawn.

uv run --locked --group analysis python research/e4/mechanisms/decay_off.py
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from plot_channels import read  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS = (30000, 30001, 30002)
STEPS = (300, 1000, 2000, 5000, 20000)
COLORS = {'wd 1': '#0072B2', 'wd 0': '#56B4E9', 'off from 300': '#E69F00', 'off from 1000': '#CC79A7'}
LABELS = {'wd 1': 'B, wd 1', 'wd 0': 'B, wd 0', 'off from 300': 'wd 1, then wd 0 from update 300',
          'off from 1000': 'wd 1, then wd 0 from update 1,000'}
PANELS = [
    ('common_acc', 'accuracy, 483 common pairs', None), ('acc_ref_only', 'accuracy of the reference channel alone', None),
    ('competitor_ratio', 'post hoc: −(off-reference margin)/(reference margin), S', 0),
    ('alignment_R', 'a+b-mode alignment R', None),
    ('margin_S_ref', 'mean margin over S: reference channel', 0), ('margin_S_off', 'mean margin over S: off-reference channel', 0),
    ('margin_N_ref', 'mean margin over N: reference channel', 0), ('margin_N_off', 'mean margin over N: off-reference channel', 0),
    ('UV_ref_norm', '‖U, V‖ at reference frequencies', None), ('UV_off_norm', '‖U, V‖ at other non-DC frequencies', None),
    ('UV_dc_norm', '‖U, V‖ at DC', None), ('norm', 'weight norm, all parameters', None),
    ('W_ref_norm', '‖W‖ at reference frequencies', None), ('W_off_norm', '‖W‖ at other frequencies (non-DC)', None),
    ('W_dc_norm', '‖W‖ at DC (constant over labels)', None), ('gates_changed_from_update100', 'gates changed since update 100', None),
]
KEYS = ['common_acc', 'acc_ref_only', 'P_cS', 'margin_S_total', 'margin_S_ref', 'margin_S_off', 'margin_S_offset',
        'margin_N_total', 'margin_N_ref', 'margin_N_off', 'alignment_R', 'useful_amplitude', 'UV_ref_norm', 'UV_dc_norm',
        'UV_off_norm', 'W_ref_norm', 'W_dc_norm', 'W_off_norm', 'norm', 'gates_changed_from_update100',
        'competitor_ratio']


def runs(rows, seed):
    """Per condition, step -> row for one source. For wd 1 and wd 0 the dense replay row wins where
    a replay and an e3 checkpoint share a step (they are equal: results/replay_check.json)."""
    out = {}
    for name, wd in (('wd 1', 1.), ('wd 0', 0.)):
        rs = [r for r in rows if r['arm'] == 'filt' and r['wd'] == wd and r['init_seed'] == seed
              and r['source'] in ('e3', 'replay')]
        out[name] = {r['step']: r for r in sorted(rs, key=lambda r: r['source'] == 'replay')}  # replay last
    for u in (300, 1000):
        out[f'off from {u}'] = {r['step']: r for r in rows if r['source'] == 'offswitch' and r['switch_step'] == u
                                and r['init_seed'] == seed}
    return {name: dict(sorted(steps.items())) for name, steps in out.items()}


def main():
    rows = read(HERE/'results/channels.csv')
    for r in rows:
        r['competitor_ratio'] = -r['margin_S_off']/r['margin_S_ref']  # post hoc
    by_seed = {seed: runs(rows, seed) for seed in SEEDS}
    table = [dict(condition=name, init_seed=seed, step=step, **{k: float(f'{steps[step][k]:.6g}') for k in KEYS})
             for seed, conds in by_seed.items() for name, steps in conds.items() for step in STEPS if step in steps]
    with open(HERE/'results/decay_off.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(table[0]))
        w.writeheader()
        w.writerows(table)
    fig, axes = plt.subplots(4, 4, figsize=(16, 13.5))
    for ax, (key, label, ref) in zip(axes.flat, PANELS):
        for name, color in COLORS.items():
            for i, seed in enumerate(SEEDS):
                steps = {s: r for s, r in by_seed[seed][name].items() if s >= 100}
                ax.plot(list(steps), [r[key] for r in steps.values()], color=color, lw=1.1, alpha=.85,
                        label=LABELS[name] if i == 0 else None)
        ax.set_xscale('log')
        ax.set_xlim(90, 22000)
        ax.set_xticks([100, 300, 1000, 3000, 10000], ['100', '300', '1k', '3k', '10k'])
        for t in (300, 1000):
            ax.axvline(t, color='k', lw=.6, ls=':')
        if ref is not None:
            ax.axhline(ref, color='k', lw=.6)
        ax.set_title(label, fontsize=9.5)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('updates since the edit', fontsize=9)
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    fig.suptitle('e4 mechanisms: channels and weight parts when decay is switched off mid-decline (dynamics D13; three sources each)',
                 fontsize=11)
    fig.text(.01, .005, 'Same measures and reference frequencies as the first figure (each unit\'s update-0 dominant input frequency'
             ' k_j). Gates: against each run\'s own update 100; decay-off runs, against the wd 1 run they resume, from whose'
             ' checkpoint at the switch update they start.\nwd 1 and wd 0: dense replays to 5,000, then e3 checkpoints at 10k, 15k, 20k. Decay-off runs: every published'
             ' checkpoint. Dotted verticals: the switch updates 300 and 1,000. Update 0 is not drawn.', fontsize=8, va='bottom')
    fig.tight_layout(rect=(0, .035, 1, .97))
    (HERE/'figures').mkdir(exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(HERE/f'figures/decay_off.{ext}', dpi=130)
    print(len(table), 'table rows; wrote', HERE/'figures/decay_off.png')


if __name__ == '__main__':
    main()
