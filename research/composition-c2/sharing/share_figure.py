#!/usr/bin/env python3
"""First figure of the sharing lane (QUESTION.md) and its source tables.

figures/sharing_grid.{png,svg}: 2x2 trajectory panels laid out like the design (rows: transition
tied / per-step; columns: readout tied / per-step), every seed, held-out final (solid), notN
(dotted) and training final (thin dashed) accuracy; right column: 10k endpoints by cell with
paired seeds joined. figures/decode_table.{png,svg}: decoder-across-position tables at 10k per
cell and seed, plus the ckpt-0 table (identical for the four cells of a seed).

Sources: figures/trajectories.csv, figures/endpoints.csv, figures/decode_table.csv (copied from
out/decode_table.csv, written by share_decode.py). Missing runs are listed on the figure.
uv run --locked python research/composition-c2/sharing/share_figure.py
"""
import csv
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import share_runs as S  # noqa: E402

FIG = HERE/'figures'
SEED_COLOR = {51200: '#0072B2', 51201: '#D55E00', 51202: '#009E73'}
TITLE = dict(TT='TT: transition tied, readout tied', TU='TU: transition tied, readouts per-step',
             UT='UT: transitions per-step, readout tied', UU='UU: transitions per-step, readouts per-step')
PARAMS = dict(TT=19208, TU=21272, UT=53256, UU=55320)
THRESH, CHANCE = 0.95, 0.125


def first(traj, pred):
    return next((r['step'] for r in traj if pred(r)), None)


def threshold_step(traj, k=5):
    """First logged step starting a run of k consecutive evaluations with held-out final >= THRESH."""
    run = 0
    for i, r in enumerate(traj):
        run = run + 1 if r['held_final'] >= THRESH else 0
        if run == k:
            return traj[i-k+1]['step']
    return None


def load_all():
    T, nbr = S.task()
    traj, missing = {}, []
    for c in S.CELLS:
        for s in S.SEEDS:
            try:
                traj[c, s] = S.trajectory(c, s, T, nbr)
            except FileNotFoundError:
                missing.append(S.run_id(c, s))
    return traj, missing


def write_tables(traj):
    FIG.mkdir(exist_ok=True)
    with (FIG/'trajectories.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['cell', 'seed', 'run_id', 'step', 'train_final', 'train_step1', 'train_step2', 'held_final', 'N', 'notN'])
        for (c, s), rows in traj.items():
            for r in rows:
                w.writerow([c, s, S.run_id(c, s), r['step'], r['train_final'], r['train_step1'], r['train_step2'], r['held_final'], r['N'], r['notN']])
    ends = []
    for (c, s), rows in traj.items():
        e = rows[-1]
        ends.append(dict(cell=c, seed=s, run_id=S.run_id(c, s), parameters=PARAMS[c], last_step=e['step'],
                         train_final=e['train_final'], held_final=e['held_final'], N=e['N'], notN=e['notN'],
                         first_step_train_all_heads_1=first(rows, lambda r: min(r['train_final'], r['train_step1'], r['train_step2']) == 1.0),
                         first_step_held_ge_095_x5=threshold_step(rows)))
    with (FIG/'endpoints.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ends[0])); w.writeheader(); w.writerows(ends)
    return ends


def trajectory_panel(ax, c, traj):
    for s in S.SEEDS:
        if (c, s) not in traj:
            continue
        rows = traj[c, s]; x = [r['step'] for r in rows]; col = SEED_COLOR[s]
        ax.plot(x, [r['held_final'] for r in rows], '-', color=col, lw=1.6)
        ax.plot(x, [r['notN'] for r in rows], ':', color=col, lw=1.6)
        ax.plot(x, [r['train_final'] for r in rows], '--', color=col, lw=0.8, alpha=0.8)
    ax.axhline(CHANCE, color='0.4', lw=0.8); ax.axhline(THRESH, color='0.75', lw=0.8)
    ax.set_xlim(0, 10000); ax.set_ylim(-0.02, 1.02); ax.set_title(f'{TITLE[c]}\n{PARAMS[c]:,} parameters', fontsize=9.5)
    ax.grid(alpha=0.25)


def endpoint_panel(ax, ends, keys, title):
    x = {c: i for i, c in enumerate(S.CELLS)}
    marks = dict(held_final=('o', 'held-out (128)'), notN=('s', '¬N (42)'), N=('^', 'N (86)'), train_final=('x', 'train (384)'))
    off = {k: (i - (len(keys)-1)/2)*0.18 for i, k in enumerate(keys)}
    for s in S.SEEDS:
        es = sorted((e for e in ends if e['seed'] == s and e['last_step'] == 10000), key=lambda e: x[e['cell']])
        for k in keys:
            xs = [x[e['cell']] + off[k] for e in es]; ys = [e[k] for e in es]
            ax.plot(xs, ys, '-', color=SEED_COLOR[s], lw=0.7, alpha=0.6)
            ax.plot(xs, ys, marks[k][0], color=SEED_COLOR[s], ms=5, mfc='none' if k in ('notN', 'train_final') else SEED_COLOR[s])
    for k in keys:
        ax.plot([], [], marks[k][0], color='0.3', mfc='none' if k in ('notN', 'train_final') else '0.3', label=marks[k][1])
    ax.axhline(CHANCE, color='0.4', lw=0.8)
    ax.set_xticks(range(4), S.CELLS); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=9.5); ax.legend(fontsize=7.5, loc='lower left')


def grid_figure(traj, ends, missing):
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.85], hspace=0.38, wspace=0.22, top=0.84, bottom=0.13)
    for (i, j), c in zip(((0, 0), (0, 1), (1, 0), (1, 1)), S.CELLS):
        ax = fig.add_subplot(gs[i, j]); trajectory_panel(ax, c, traj)
        if i == 1: ax.set_xlabel('update')
        if j == 0: ax.set_ylabel('accuracy')
    handles = [Line2D([], [], color=SEED_COLOR[s], lw=1.6, label=f'init seed {s}') for s in S.SEEDS] + [
        Line2D([], [], color='0.3', ls='-', lw=1.6, label='held-out final (128 rows)'),
        Line2D([], [], color='0.3', ls=':', lw=1.6, label='held-out ¬N (42 rows, no neighbour copy)'),
        Line2D([], [], color='0.3', ls='--', lw=0.8, label='train final (384 rows)')]
    fig.legend(handles=handles, ncol=6, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 0.02))
    endpoint_panel(fig.add_subplot(gs[0, 2]), ends, ['held_final', 'train_final'], 'at 10,000 updates: held-out final and train final')
    endpoint_panel(fig.add_subplot(gs[1, 2]), ends, ['N', 'notN'], 'at 10,000 updates: N and ¬N held-out rows')
    fig.suptitle('Tying transitions and readouts: r1 task ops51000-ls51100, width 128, intermediate supervision (step weights 1 1 1), '
                 'Adam lr 1e-3, wd 0.1\nTT and TU are the original r1 runs; UT and UU are c2-share runs. Lines: held-out final (solid), '
                 '¬N (dotted), train final (dashed); grey: chance 0.125 and 0.95' + (f'\nMISSING: {", ".join(missing)}' if missing else ''),
                 fontsize=9.5, y=0.985)
    for ext in ('png', 'svg'):
        fig.savefig(FIG/f'sharing_grid.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def decode_figure():
    src = HERE/'out'/'decode_table.csv'
    if not src.exists():
        return
    rows = list(csv.DictReader(src.open()))
    (FIG/'decode_table.csv').write_text(src.read_text())
    tab = {}
    for r in rows:
        tab.setdefault((r['cell'], int(r['seed']), int(r['step'])), np.zeros((4, 4)))[int(r['fit_position']), int(r['score_position'])] = float(r['held_acc'])
    lines = [(c, 10000) for c in S.CELLS] + [('TT', 0)]
    fig, axes = plt.subplots(len(lines), 3, figsize=(8.4, 2.3*len(lines)), squeeze=False)
    for i, (c, step) in enumerate(lines):
        for j, s in enumerate(S.SEEDS):
            ax = axes[i, j]; m = tab.get((c, s, step))
            if m is None:
                ax.text(0.5, 0.5, 'missing', ha='center', transform=ax.transAxes); ax.set_axis_off(); continue
            ax.imshow(m, vmin=0, vmax=1, cmap='viridis')
            for a in range(4):
                for b in range(4):
                    ax.text(b, a, f'{m[a, b]:.2f}', ha='center', va='center', fontsize=7, color='w' if m[a, b] < 0.6 else 'k')
            ax.set_xticks(range(4)); ax.set_yticks(range(4))
            ax.set_title(f'{c if step else "init (all cells)"} {s}' + ('' if step else ', ckpt-0'), fontsize=8.5)
            if j == 0: ax.set_ylabel('probe fit on h_i (train)', fontsize=7.5)
            if i == len(lines)-1: ax.set_xlabel('scored on h_j (held-out programs)', fontsize=7.5)
    fig.suptitle('Decoder across position at 10,000 updates (c2probe.decode_table, probe seed 64300)\n'
                 'h1, h2 of held-out programs are training-prefix states; low off-diagonal ≠ absent, high ≠ used', fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(FIG/f'decode_table.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    traj, missing = load_all()
    ends = write_tables(traj)
    grid_figure(traj, ends, missing)
    decode_figure()
    for e in ends:
        print(e)


if __name__ == '__main__':
    main()
