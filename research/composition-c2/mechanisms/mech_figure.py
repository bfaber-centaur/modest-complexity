#!/usr/bin/env python3
"""Figures from the committed run c2-mech-interchange-a1 (cells.csv), recovered 10k checkpoints.

interchange.png/.svg/.csv (primary): per condition and seed, the suffix-success matrix S(i, j)
  (donor depth i, recipient depth j, all 262,144 pairs per cell), annotated with history
  consistency C; right column: S against program length L = 3 + i - j for every seed, the
  semantic-centroid patch S_cent of every cell at its L, and the two random-state controls.
centroid.png/.svg/.csv (supporting): S_cent(i, j) per condition and seed, with natural S beside it.

uv run --locked python research/composition-c2/mechanisms/mech_figure.py
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = 'c2-mech-interchange-a1'
STEP = '10000'
CONDS = ['tied/inter', 'per-step readout/inter', 'tied/final']
SEEDS = ['51200', '51201', '51202']
SEED_COLORS = dict(zip(SEEDS, ['#1b6ca8', '#d1495b', '#3a7d44']))
CMAP = 'viridis'


def load():
    rows = [r for r in csv.DictReader((HERE/'runs'/RUN/'cells.csv').open()) if r['step'] == STEP]
    out = {}
    for r in rows:
        out[(r['condition'], r['seed'], int(r['i']), int(r['j']))] = {k: float(r[k]) for k in
                                                                      ('L', 'S', 'C', 'S_cent', 'hit_perm', 'hit_marg')}
    return out


def matrix(d, cond, seed, key):
    return np.array([[d[(cond, seed, i, j)][key] for j in range(4)] for i in range(4)])


def heatmap(ax, M, lines, title):
    im = ax.imshow(M, vmin=0, vmax=1, cmap=CMAP)
    for i in range(4):
        for j in range(4):
            color = 'black' if M[i, j] > 0.6 else 'white'
            ax.text(j, i - 0.12, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=9, color=color)
            ax.text(j, i + 0.25, lines[i][j], ha='center', va='center', fontsize=6.5, color=color)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xlabel('recipient depth j', fontsize=8); ax.set_ylabel('donor depth i', fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=9)
    return im


def write_csv(path, d):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id_source', 'condition', 'seed', 'step', 'i', 'j', 'L', 'S', 'C', 'S_cent', 'hit_perm', 'hit_marg'])
        for (cond, seed, i, j), v in sorted(d.items()):
            w.writerow([RUN, cond, seed, STEP, i, j, int(v['L']), v['S'], v['C'], v['S_cent'], v['hit_perm'], v['hit_marg']])


def primary(d):
    fig, axes = plt.subplots(3, 4, figsize=(15, 11), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.35], hspace=0.42, wspace=0.3))
    for r, cond in enumerate(CONDS):
        for c, seed in enumerate(SEEDS):
            S, C = matrix(d, cond, seed, 'S'), matrix(d, cond, seed, 'C')
            lines = [[('C –' if np.isnan(C[i, j]) else f'C {C[i, j]:.2f}') for j in range(4)] for i in range(4)]
            im = heatmap(axes[r, c], S, lines, f'{cond}, seed {seed}')
        ax = axes[r, 3]
        for k, seed in enumerate(SEEDS):
            byL = {}
            for (cc, ss, i, j), v in d.items():
                if cc == cond and ss == seed:
                    byL.setdefault(int(v['L']), []).append(v)
            Ls = sorted(byL)
            ax.plot(Ls, [byL[L][0]['S'] for L in Ls], '-', color=SEED_COLORS[seed], lw=1.8, marker='s', ms=4,
                    label=f'natural S, seed {seed}')
            off = (k - 1) * 0.12
            for L in Ls:
                ax.scatter([L + off] * len(byL[L]), [v['S_cent'] for v in byL[L]], s=16, facecolors='none',
                           edgecolors=SEED_COLORS[seed], lw=0.9, label='centroid S_cent (one per cell)' if (L == 0 and k == 0) else None)
            ax.plot(Ls, [np.mean([v['hit_perm'] for v in byL[L]]) for L in Ls], ':', color='0.45', lw=1,
                    label='control R1 (permuted donor), target hit' if k == 0 else None)
            ax.plot(Ls, [np.mean([v['hit_marg'] for v in byL[L]]) for L in Ls], '--', color='0.65', lw=1,
                    label='control R2 (per-unit marginals), target hit' if k == 0 else None)
        ax.axhline(1/8, color='black', lw=0.7, alpha=0.6)
        ax.text(6.4, 1/8 + 0.015, 'chance 1/8', fontsize=7, ha='right')
        ax.axvline(3, color='0.8', lw=0.8, zorder=0)
        ax.set_xlim(-0.4, 6.4); ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('program length L = 3 + i − j', fontsize=8)
        ax.set_ylabel('suffix success', fontsize=8); ax.tick_params(labelsize=8)
        ax.set_title(f'{cond}: by length (L = 3 is same-step)', fontsize=9)
        if r == 0:
            ax.legend(fontsize=6.5, loc='center left', bbox_to_anchor=(0.0, 0.47), frameon=True, framealpha=0.9)
    fig.colorbar(im, ax=axes[:, :3], shrink=0.5, label='S: patched answer = oracle suffix on donor symbol')
    fig.suptitle('Whole-state interchange in the recovered r1 RNNs at 10,000 updates (task ops51000-ls51100). '
                 'Shared transition: S(i, j) = accuracy on recombined programs of length 3 + i − j (checked bitwise).\n'
                 'Recipient readout V3 for per-step readouts (distribution shift off L = 3). '
                 'C = agreement of patched answers among different histories with the same donor symbol.',
                 fontsize=9.5)
    return fig


def supporting(d):
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11), gridspec_kw=dict(hspace=0.35))
    for r, cond in enumerate(CONDS):
        for c, seed in enumerate(SEEDS):
            S, Sc = matrix(d, cond, seed, 'S'), matrix(d, cond, seed, 'S_cent')
            lines = [[f'nat. {S[i, j]:.2f}' for j in range(4)] for i in range(4)]
            im = heatmap(axes[r, c], Sc, lines, f'{cond}, seed {seed}')
    fig.colorbar(im, ax=axes, shrink=0.5, label='S_cent: centroid-patched answer = oracle suffix on its symbol')
    fig.suptitle('Semantic-centroid patches: c_i(x) = mean training-row h_i with symbol x, placed at recipient depth j '
                 '(8 symbols × all suffixes per cell).\nNot an input recombination; (3, 3) is a linear readout of a class mean. '
                 'Recovered 10k checkpoints, run c2-mech-interchange-a1.', fontsize=9.5)
    return fig


def main():
    d = load()
    out = HERE/'figures'; out.mkdir(exist_ok=True)
    for name, make in (('interchange', primary), ('centroid', supporting)):
        fig = make(d)
        fig.savefig(out/f'{name}.png', dpi=130, bbox_inches='tight'); fig.savefig(out/f'{name}.svg', bbox_inches='tight')
        plt.close(fig)
        write_csv(out/f'{name}.csv', d)
    print('wrote', out)


if __name__ == '__main__':
    main()
