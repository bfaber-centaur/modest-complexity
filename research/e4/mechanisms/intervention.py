#!/usr/bin/env python3
"""The mechanisms intervention (INTERVENTION.md; DECISIONS D14, D15): table and figure from
channels.py's results/channels.csv.

Per source, f = (acc_arm - acc_wd1) / (acc_wd0 - acc_wd1) on the 483 common pairs at updates 2,000
and 5,000, with the wd 1 and wd 0 D7 replays as references, and the pre-stated reading applied:
f >= 0.7 in all three sources, f <= 0.3 in all three, or mixed. The figure overlays the two arms
on the replays (one line per source, no pooling); update 0 is drawn at the left edge.

Post hoc (not part of the pre-stated reading): results/decay_magnitude.csv sets the logged decay
step norm ||lr * wd * decaying part|| (metrics.jsonl, optimizer_update.decay_norm) beside common
accuracy at 2,000 and 5,000 for the two arms, the wd 1 replays and the dynamics lane's uniform
wd 0.1 and 0.3 runs (accuracy from their manifest-verified published checkpoints), to ask whether
the size of the decay, rather than where it acts, could account for the arms.

uv run --locked --group analysis python research/e4/mechanisms/intervention.py
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from plot_channels import ZERO, read  # noqa: E402
from channels import e3  # noqa: E402  (manifest-verified loading, common pairs)

HERE = Path(__file__).resolve().parent
SEEDS = (30000, 30001, 30002)
LINES = {('replay', 1., ''): ('B, wd 1 (replay)', '#0072B2'), ('replay', 0., ''): ('B, wd 0 (replay)', '#56B4E9'),
         ('intervention', 1., 'rest'): ('wd 1, reference parts exempt from decay', '#009E73'),
         ('intervention', 1., 'reference'): ('wd 1, decay on reference parts only', '#D55E00')}
ARMS = {'xref': 'rest', 'onlyref': 'reference'}
PANELS = [
    ('common_acc', 'accuracy, 483 common pairs', None), ('acc_ref_only', 'accuracy of the reference channel alone', None),
    ('P_cS', 'P(a+b over S)', None), ('gates_changed_from_update100', 'gates changed since update 100', None),
    ('margin_S_ref', 'mean margin over S: reference channel', 0), ('margin_S_off', 'mean margin over S: off-reference channel', 0),
    ('margin_N_ref', 'mean margin over N: reference channel', 0), ('margin_N_off', 'mean margin over N: off-reference channel', 0),
    ('UV_ref_norm', '‖U, V‖ at reference frequencies', None), ('UV_off_norm', '‖U, V‖ at other non-DC frequencies', None),
    ('W_ref_norm', '‖W‖ at reference frequencies', None), ('W_off_norm', '‖W‖ at other frequencies (non-DC)', None),
]


def pick(rows, source, wd, decay_on, seed):
    return sorted((r for r in rows if r['source'] == source and r['wd'] == wd and r['decay_on'] == decay_on
                   and r['init_seed'] == seed), key=lambda r: r['step'])


def at(rows, step, key='common_acc'):
    return next(r[key] for r in rows if r['step'] == step)


def table(rows):
    out = []
    for arm, decay_on in ARMS.items():
        for seed in SEEDS:
            x, lo, hi = (pick(rows, *k, seed) for k in (('intervention', 1., decay_on), ('replay', 1., ''), ('replay', 0., '')))
            row = dict(arm=arm, decay_on=decay_on, init_seed=int(seed))
            for step in (2000, 5000):
                a, a1, a0 = at(x, step), at(lo, step), at(hi, step)
                row.update({f'acc_{step}': a, f'acc_wd1_{step}': a1, f'acc_wd0_{step}': a0, f'f_{step}': (a - a1)/(a0 - a1)})
            out.append(row)
    return out


def magnitude():
    runs = [('mechanisms', f'e4-me-filt-wd1-{kind}-w128-i{s}-a1') for kind in ('replay', 'xref', 'onlyref') for s in SEEDS]
    runs += [('dynamics', f'e4-dy-filt-wd{wd}-w128-i{s}-a1') for wd in ('0p1', '0p3') for s in SEEDS]
    out = []
    for lane, run_id in runs:
        run = HERE.parent/lane/'runs'/run_id
        logged = {x['step']: x for x in map(json.loads, (run/'metrics.jsonl').read_text().splitlines())}
        with np.load(run/'data.npz') as d:
            a, b, y = d['a'][e3.COMMON], d['b'][e3.COMMON], d['yadd'][e3.COMMON]
        row = dict(run_id=run_id)
        for step in (2000, 5000):
            w = e3.load_weights(run, step)
            row[f'decay_norm_{step}'] = logged[step]['optimizer_update']['decay_norm']
            row[f'acc_{step}'] = float(np.mean((np.maximum(w[0][a] + w[1][b], 0) @ w[2]).argmax(1) == y))
        out.append(row)
    return out


def reading(rows, arm):
    f = [r['f_5000'] for r in rows if r['arm'] == arm]
    return 'f >= 0.7 in all three' if min(f) >= .7 else 'f <= 0.3 in all three' if max(f) <= .3 else 'mixed or intermediate'


def main():
    rows = read(HERE/'results/channels.csv')
    tab = table(rows)
    with open(HERE/'results/intervention.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(tab[0]))
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in tab)
    for arm in ARMS:
        print(arm, [round(r['f_5000'], 3) for r in tab if r['arm'] == arm], '->', reading(tab, arm))
    mag = magnitude()
    with open(HERE/'results/decay_magnitude.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(mag[0]))
        w.writeheader()
        w.writerows({k: (float(f'{v:.6g}') if isinstance(v, float) else v) for k, v in r.items()} for r in mag)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.5))
    for ax, (key, label, ref) in zip(axes.flat, PANELS):
        for (source, wd, decay_on), (name, color) in LINES.items():
            for i, seed in enumerate(SEEDS):
                rs = pick(rows, source, wd, decay_on, seed)
                ax.plot([max(r['step'], ZERO) for r in rs], [r[key] for r in rs], color=color, lw=1.1, alpha=.85,
                        label=name if i == 0 else None)
        ax.set_xscale('log')
        ax.set_xlim(ZERO*.9, 6000)
        ax.set_xticks([ZERO, 10, 100, 1000], ['0', '10', '100', '1k'])
        for t in (300, 2000):
            ax.axvline(t, color='k', lw=.6, ls=':')
        if ref is not None:
            ax.axhline(ref, color='k', lw=.6)
        ax.set_title(label, fontsize=9.5)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('updates since the edit', fontsize=9)
    axes[0, 0].legend(fontsize=7.5, frameon=False)
    f5 = {arm: ', '.join(f"{r['f_5000']:.2f}" for r in tab if r['arm'] == arm) for arm in ARMS}
    fig.suptitle('e4 mechanisms intervention: where decoupled decay acts (wd 1, three sources each)', fontsize=11)
    fig.text(.01, .005, 'Reference parts: each unit\'s cosine and sine at its update-0 dominant frequency k_j in U, V (token axis) and W'
             ' (label axis); DC and other frequencies are the rest. Nothing frozen; only the decay is restricted.\n'
             f'f at 5,000 = (acc − acc wd 1)/(acc wd 0 − acc wd 1), per source 30000/30001/30002: exempt {f5["xref"]};'
             f' reference only {f5["onlyref"]}. Dotted verticals: updates 300 and 2,000.', fontsize=8, va='bottom')
    fig.tight_layout(rect=(0, .04, 1, .97))
    (HERE/'figures').mkdir(exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(HERE/f'figures/intervention.{ext}', dpi=130)


if __name__ == '__main__':
    main()
