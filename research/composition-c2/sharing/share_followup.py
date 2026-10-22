#!/usr/bin/env python3
"""Follow-up figure: the 2x2 at wd 0 (c2-share-wd0-*) beside the same cells at wd 0.1.

figures/followup_wd0.{png,svg}: 2x2 trajectory panels at wd 0 (held-out final solid, notN dotted,
train final dashed; the same seed's wd 0.1 held-out final as a faint line), and 10k endpoints by
cell at wd 0.1 (open markers) and wd 0 (filled). figures/followup_decode_wd0.{png,svg}: the
decoder-across-position table at 10k for the wd 0 runs (from out/decode_table_wd0.csv).
Sources: figures/followup_trajectories.csv, figures/followup_endpoints.csv,
figures/followup_decode_wd0.csv.

wd 0 is a different recipe, not a control isolating regularization: a change in a tying effect
between the two says the effect depends on decay under this recipe.
uv run --locked python research/composition-c2/sharing/share_followup.py
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
from share_figure import CHANCE, FIG, PARAMS, SEED_COLOR, THRESH, TITLE, first, threshold_step  # noqa: E402

WD = dict(main=0.1, wd0=0.0)


def load():
    T, nbr = S.task()
    traj, missing = {}, []
    for v in WD:
        for c in S.CELLS:
            for s in S.SEEDS:
                try:
                    traj[v, c, s] = S.trajectory(c, s, T, nbr, v)
                except FileNotFoundError:
                    missing.append(S.run_id(c, s, v))
    return traj, missing


def tables(traj):
    FIG.mkdir(exist_ok=True)
    with (FIG/'followup_trajectories.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['wd', 'cell', 'seed', 'run_id', 'step', 'train_final', 'train_step1', 'train_step2', 'held_final', 'N', 'notN'])
        for (v, c, s), rows in traj.items():
            if v == 'wd0':
                for r in rows:
                    w.writerow([WD[v], c, s, S.run_id(c, s, v), r['step'], r['train_final'], r['train_step1'], r['train_step2'], r['held_final'], r['N'], r['notN']])
    ends = []
    for (v, c, s), rows in traj.items():
        e = rows[-1]
        ends.append(dict(wd=WD[v], cell=c, seed=s, run_id=S.run_id(c, s, v), parameters=PARAMS[c], last_step=e['step'],
                         train_final=e['train_final'], held_final=e['held_final'], N=e['N'], notN=e['notN'],
                         first_step_train_all_heads_1=first(rows, lambda r: min(r['train_final'], r['train_step1'], r['train_step2']) == 1.0),
                         first_step_held_ge_095_x5=threshold_step(rows)))
    with (FIG/'followup_endpoints.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(ends[0])); w.writeheader(); w.writerows(ends)
    return ends


def panel(ax, c, traj):
    for s in S.SEEDS:
        col = SEED_COLOR[s]
        if ('main', c, s) in traj:
            rows = traj['main', c, s]
            ax.plot([r['step'] for r in rows], [r['held_final'] for r in rows], '-', color=col, lw=1.0, alpha=0.3)
        if ('wd0', c, s) not in traj:
            continue
        rows = traj['wd0', c, s]; x = [r['step'] for r in rows]
        ax.plot(x, [r['held_final'] for r in rows], '-', color=col, lw=1.6)
        ax.plot(x, [r['notN'] for r in rows], ':', color=col, lw=1.6)
        ax.plot(x, [r['train_final'] for r in rows], '--', color=col, lw=0.8, alpha=0.8)
    ax.axhline(CHANCE, color='0.4', lw=0.8); ax.axhline(THRESH, color='0.75', lw=0.8)
    ax.set_xlim(0, 10000); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25)
    ax.set_title(f'{TITLE[c]}\n{PARAMS[c]:,} parameters, wd 0', fontsize=9.5)


def endpoints(ax, ends, key, title):
    x = {c: i for i, c in enumerate(S.CELLS)}
    for wd, dx, filled in ((0.1, -0.12, False), (0.0, 0.12, True)):
        for s in S.SEEDS:
            es = sorted((e for e in ends if e['seed'] == s and e['wd'] == wd and e['last_step'] == 10000), key=lambda e: x[e['cell']])
            xs = [x[e['cell']] + dx for e in es]; ys = [e[key] for e in es]
            ax.plot(xs, ys, '-', color=SEED_COLOR[s], lw=0.7, alpha=0.6 if filled else 0.3)
            ax.plot(xs, ys, 'o', color=SEED_COLOR[s], ms=5, mfc=SEED_COLOR[s] if filled else 'none')
    ax.plot([], [], 'o', color='0.3', mfc='none', label='wd 0.1 (initial comparison)')
    ax.plot([], [], 'o', color='0.3', label='wd 0 (follow-up)')
    ax.axhline(CHANCE, color='0.4', lw=0.8)
    ax.set_xticks(range(4), S.CELLS); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=9.5); ax.legend(fontsize=7.5, loc='lower left')


def grid(traj, ends, missing):
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.85], hspace=0.38, wspace=0.22, top=0.84, bottom=0.13)
    for (i, j), c in zip(((0, 0), (0, 1), (1, 0), (1, 1)), S.CELLS):
        ax = fig.add_subplot(gs[i, j]); panel(ax, c, traj)
        if i == 1: ax.set_xlabel('update')
        if j == 0: ax.set_ylabel('accuracy')
    endpoints(fig.add_subplot(gs[0, 2]), ends, 'held_final', 'held-out final at 10,000 updates')
    endpoints(fig.add_subplot(gs[1, 2]), ends, 'notN', '¬N (42 rows) at 10,000 updates')
    handles = [Line2D([], [], color=SEED_COLOR[s], lw=1.6, label=f'init seed {s}') for s in S.SEEDS] + [
        Line2D([], [], color='0.3', ls='-', lw=1.6, label='held-out final, wd 0'),
        Line2D([], [], color='0.3', ls=':', lw=1.6, label='¬N, wd 0'),
        Line2D([], [], color='0.3', ls='--', lw=0.8, label='train final, wd 0'),
        Line2D([], [], color='0.3', ls='-', lw=1.0, alpha=0.3, label='held-out final, wd 0.1 (same seed)')]
    fig.legend(handles=handles, ncol=7, fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 0.02))
    fig.suptitle('Follow-up: the same 2x2 without weight decay (wd 0; otherwise the initial recipe, paired init seeds 51200-51202)\n'
                 'wd 0 is a different recipe, not a control isolating regularization' + (f'\nMISSING: {", ".join(missing)}' if missing else ''),
                 fontsize=9.5, y=0.985)
    for ext in ('png', 'svg'):
        fig.savefig(FIG/f'followup_wd0.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def decode():
    src = HERE/'out'/'decode_table_wd0.csv'
    if not src.exists():
        return
    (FIG/'followup_decode_wd0.csv').write_text(src.read_text())
    tab = {}
    for r in csv.DictReader(src.open()):
        if int(r['step']) == 10000:
            tab.setdefault((r['cell'], int(r['seed'])), np.zeros((4, 4)))[int(r['fit_position']), int(r['score_position'])] = float(r['held_acc'])
    fig, axes = plt.subplots(4, 3, figsize=(8.4, 9.2), squeeze=False)
    for i, c in enumerate(S.CELLS):
        for j, s in enumerate(S.SEEDS):
            ax, m = axes[i, j], tab.get((c, s))
            if m is None:
                ax.text(0.5, 0.5, 'missing', ha='center', transform=ax.transAxes); ax.set_axis_off(); continue
            ax.imshow(m, vmin=0, vmax=1, cmap='viridis')
            for a in range(4):
                for b in range(4):
                    ax.text(b, a, f'{m[a, b]:.2f}', ha='center', va='center', fontsize=7, color='w' if m[a, b] < 0.6 else 'k')
            ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_title(f'{c} {s}, wd 0', fontsize=8.5)
            if j == 0: ax.set_ylabel('probe fit on h_i (train)', fontsize=7.5)
            if i == 3: ax.set_xlabel('scored on h_j (held-out programs)', fontsize=7.5)
    fig.suptitle('Decoder across position at 10,000 updates, wd 0 (probe seed 64300)\n'
                 'h1, h2 of held-out programs are training-prefix states; low off-diagonal ≠ absent, high ≠ used', fontsize=9)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(FIG/f'followup_decode_wd0.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    traj, missing = load()
    ends = tables(traj)
    grid(traj, ends, missing)
    decode()
    for e in ends:
        if e['wd'] == 0.0:
            print(e)


if __name__ == '__main__':
    main()
