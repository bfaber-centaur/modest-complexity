#!/usr/bin/env python3
"""Figures of the generalization lane, drawn from committed run outputs only.

length       accuracy against program length 1-6 for the recovered r1 tied/inter and tied/final
             checkpoints (QUESTION.md first-figure plan), with reference systems and chance.
replication  Part B: held-out final and notN accuracy over training, and frozen accuracy by length
             at 10k, for the original task (r1 runs) and the four new tasks; summary table.
followup     follow-up: concentration of first errors at steps 4-6, held-out-window share of first
             errors, and length-6 accuracy with / without a held-out window, per tied/inter model.

uv run --locked python research/composition-c2/generalization/gen_figures.py length
uv run --locked python research/composition-c2/generalization/gen_figures.py replication
uv run --locked python research/composition-c2/generalization/gen_figures.py followup
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'common'), str(HERE.parents[1]/'regimes'/'common')]
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
                    label=f'init {seed}, length 3 held-out only (like-for-like with 4-6)')
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
    fig.suptitle('Frozen r1 RNNs (ckpt-10000, original task ops51000-ls51100) on programs of lengths 1-6; exhaustive rows, no fitting.\n'
                 'Filled points at length 3 cover all 64 programs, 75% of them training rows; the open held-out-only point is the '
                 'like-for-like comparison with lengths 4-6.', fontsize=9, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'length.{ext}', dpi=150)
    plt.close(fig)


R1_RUNS = HERE.parents[1]/'regimes'/'composition'/'runs'
FROZEN_REP = HERE/'runs'/'c2-gen-frozen-rep-a1'
INIT_STYLE = {63200: ('#eda100', 'D'), 63201: ('#e87ba4', 'v')}  # validated categorical slots 4-5: distinct from the r1 seeds


def trajectories():
    """Rows (task, condition, init, run_id, step, train_final, held_final, notN) for the six r1
    tied/tied runs (notN scored from the logged prediction strings) and the 16 Part B runs."""
    import c2task
    T = c2task.load(c2task.ORIGINAL)
    nbr = c2task.neighbour_answerable(T['P'], T['train'], T['held'])[0]
    y = c2task.rows(T['P'], T['held'])['states'][:, -1]
    out = []
    for cond in ('inter', 'final'):
        for seed in (51200, 51201, 51202):
            rid = f'r1-comp-rnn-{cond}-s{seed}-a1'
            for line in (R1_RUNS/rid/'metrics.jsonl').read_text().splitlines():
                r = json.loads(line)
                ok = [int(c) == t for c, t in zip(r['held_final_pred'], y)]
                out.append(dict(task=c2task.ORIGINAL, condition=cond, init_seed=seed, run_id=rid, step=r['step'],
                                train_final=r['train']['final']['acc'], held_final=r['held']['final']['acc'],
                                notN=sum(o for o, n in zip(ok, nbr) if not n) / int((~nbr).sum())))
    for d in sorted((HERE/'runs').glob('c2-gen-*-ops*')):
        cfg = json.loads((d/'config.json').read_text())
        cond = 'inter' if cfg['step_weights'] == [1., 1., 1.] else 'final'
        for line in (d/'metrics.jsonl').read_text().splitlines():
            r = json.loads(line)
            out.append(dict(task=cfg['task']['id'], condition=cond, init_seed=cfg['init_seed'], run_id=d.name, step=r['step'],
                            train_final=r['train']['final']['acc'], held_final=r['held']['final']['acc'],
                            notN=r['held_subset']['notN']))
    return out


def replication_figure():
    traj = trajectories()
    lengths = read(FROZEN/'length.csv') + read(FROZEN_REP/'length.csv')
    tasks = list(dict.fromkeys(r['task'] for r in traj))
    runs = list(dict.fromkeys((r['task'], r['condition'], r['init_seed'], r['run_id']) for r in traj))
    FIGS.mkdir(exist_ok=True)
    summary = []
    for task, cond, seed, rid in runs:
        tr = [r for r in traj if r['run_id'] == rid]
        last = tr[-1]
        fit = next((r['step'] for r in tr if r['train_final'] == 1.0), None)
        acc = {int(r['length']): float(r['accuracy']) for r in lengths if r['run_id'] == rid and r['subset'] == 'all'}
        summary.append(dict(task=task, condition=cond, init_seed=seed, run_id=rid, step=last['step'], train_final=last['train_final'],
                            first_step_train_final_1=fit, held_final=last['held_final'], notN=last['notN'],
                            **{f'length{L}': acc.get(L) for L in range(1, 7)}))
    with open(FIGS/'replication.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(summary[0])); w.writeheader(); w.writerows(summary)
    with open(FIGS/'replication_trajectories.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(traj[0])); w.writeheader(); w.writerows(traj)

    fig, axes = plt.subplots(3, len(tasks), figsize=(2.75*len(tasks), 8.2), sharey=True)
    style = {**SEED_STYLE, **INIT_STYLE}
    for j, task in enumerate(tasks):
        for i, key in enumerate(('held_final', 'notN')):
            ax = axes[i, j]
            for task_, cond, seed, rid in runs:
                if task_ != task:
                    continue
                tr = [r for r in traj if r['run_id'] == rid]
                c, m = style[seed]
                ax.plot([r['step'] for r in tr], [r[key] for r in tr], color=c, lw=1.6 if cond == 'inter' else 1.1,
                        ls='-' if cond == 'inter' else (0, (4, 2)), label=f'tied/{cond}, init {seed}')
            ax.axhline(CHANCE, color=GREY, lw=0.8, ls=(0, (1, 2)))
            ax.set_xlim(0, 10000); ax.set_xticks([0, 5000, 10000]); ax.set_xticklabels(['0', '5k', '10k'])
        ax = axes[2, j]
        for task_, cond, seed, rid in runs:
            if task_ != task:
                continue
            c, m = style[seed]
            acc = [next(float(r['accuracy']) for r in lengths if r['run_id'] == rid and r['subset'] == 'all' and int(r['length']) == L)
                   for L in range(1, 7)]
            ax.plot(range(1, 7), acc, color=c, marker=m, ms=5, lw=1.6 if cond == 'inter' else 1.1,
                    ls='-' if cond == 'inter' else (0, (4, 2)), mfc=c if cond == 'inter' else 'white')
        ax.axvspan(2.6, 3.4, color='#ecebe6', zorder=0)
        ax.axhline(CHANCE, color=GREY, lw=0.8, ls=(0, (1, 2)))
        ax.set_xticks(range(1, 7)); ax.set_xlim(0.7, 6.3)
        n_notN = next(s for s in summary if s['task'] == task)
        T_name = 'original (r1 runs)' if task == 'ops51000-ls51100' else 'new task'
        axes[0, j].set_title(f'{task}\n{T_name}', fontsize=8.5, color=INK, loc='left')
        for i in range(3):
            a = axes[i, j]
            a.grid(axis='y', color='#e4e3dd', lw=0.6); a.set_axisbelow(True); a.set_ylim(0, 1.04)
            a.tick_params(labelsize=7.5)
            for sp in ('top', 'right'):
                a.spines[sp].set_visible(False)
        axes[1, j].set_xlabel('updates', fontsize=8); axes[0, j].set_xlabel('updates', fontsize=8)
        axes[2, j].set_xlabel('program length (frozen, ckpt-10000)', fontsize=8)
    axes[0, 0].set_ylabel('held-out final accuracy\n(16 programs, 128 rows)', fontsize=8.5)
    axes[1, 0].set_ylabel('held-out accuracy on notN rows\n(no one-slot neighbour answers)', fontsize=8.5)
    axes[2, 0].set_ylabel('accuracy, all 8·4^L rows\n(L=3: 75% training rows;\nheld-out-only: top row at 10k)', fontsize=8)
    handles = []
    from matplotlib.lines import Line2D
    for seed, (c, m) in style.items():
        handles.append(Line2D([], [], color=c, marker=m, lw=1.6, label=f'init {seed}'))
    handles += [Line2D([], [], color=INK, lw=1.6, label='tied/inter (solid, filled)'),
                Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 2)), marker='o', mfc='white', label='tied/final (dashed, open)'),
                Line2D([], [], color=GREY, lw=0.8, ls=(0, (1, 2)), label='chance 0.125')]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=7.5, frameon=False)
    fig.suptitle('Replication: r1 recipe (width 128, Adam lr 1e-3, wd 0.1, 10k full-batch updates) on the original and four new tasks; '
                 'every run shown', fontsize=9, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'replication.{ext}', dpi=140)
    plt.close(fig)


FOLLOWUP = HERE/'runs'/'c2-gen-followup-anatomy-a1'


def followup_figure():
    summ = json.loads((FOLLOWUP/'summary.json').read_text())['per_model']
    wins = read(FOLLOWUP/'windows.csv')
    models = list(summ)
    style = {**SEED_STYLE, **INIT_STYLE}
    label = lambda rid: ('original' if rid.startswith('r1-') else summ[rid]['task_id'].replace('ops', '').replace('-ls', '/')) + f"\ninit {summ[rid]['init_seed']}"
    rows = []
    for rid in models:
        acc = {r['contains_held_out_window']: float(r['accuracy']) for r in wins if r['run_id'] == rid and r['length'] == '6'}
        rows.append(dict(run_id=rid, task_id=summ[rid]['task_id'], init_seed=summ[rid]['init_seed'], top4_share=summ[rid]['top4_share'],
                         first_error_window_is_held_out=summ[rid]['first_error_window_is_held_out'],
                         at_risk_window_is_held_out=summ[rid]['at_risk_window_is_held_out'],
                         length6_acc_with_held_window=acc['True'], length6_acc_without_held_window=acc['False'],
                         source=f'runs/{FOLLOWUP.name}/summary.json, windows.csv'))
    with open(FIGS/'followup.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharex=True)
    x = range(len(models))
    panels = [('top4_share', 'share of first errors (steps 4-6)\non the 4 most frequent of 32 transitions',
               [(4/32, 'diffuse 4/32', (0, (1, 2))), (0.5, 'F1 threshold 0.5', (0, (4, 3)))]),
              ('first_error_window_is_held_out', 'share of first errors whose last 3 ops\nform a held-out program',
               [(0.25, 'base rate 0.25', (0, (1, 2))), (0.375, 'F3 threshold 0.375', (0, (4, 3)))])]
    for ax, (key, ylab, lines) in zip(axes[:2], panels):
        for i, r in enumerate(rows):
            c, m = style[r['init_seed']]
            ax.plot(i, r[key], marker=m, color=c, ms=8, ls='none')
        for y, txt, ls in lines:
            ax.axhline(y, color=GREY, lw=1, ls=ls); ax.text(len(models)-0.4, y, txt, fontsize=7, color=INK, va='bottom', ha='right')
        ax.set_ylim(0, 0.6); ax.set_ylabel(ylab, fontsize=8.5)
    ax = axes[2]
    for i, r in enumerate(rows):
        c, m = style[r['init_seed']]
        ax.plot([i, i], [r['length6_acc_with_held_window'], r['length6_acc_without_held_window']], color=c, lw=1.2)
        ax.plot(i, r['length6_acc_without_held_window'], marker=m, color=c, ms=8, ls='none')
        ax.plot(i, r['length6_acc_with_held_window'], marker=m, mfc='white', mec=c, mew=1.5, ms=8, ls='none')
    ax.set_ylim(0.4, 1.0); ax.set_ylabel('length-6 accuracy: programs without (filled)\nand with (open) a held-out 3-op window', fontsize=8.5)
    for a in axes:
        a.set_xticks(list(x)); a.set_xticklabels([label(r) for r in models], fontsize=6.5, rotation=90)
        a.axvline(2.5, color='#d8d7d0', lw=0.8)
        a.grid(axis='y', color='#e4e3dd', lw=0.6); a.set_axisbelow(True); a.tick_params(axis='y', labelsize=8)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.suptitle('Follow-up: first errors of the 11 frozen tied/inter models (ckpt-10000) on exhaustive length-6 programs '
                 '(3 original, 8 new-task; c2-gen-followup-anatomy-a1)', fontsize=9, color=INK, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'followup.{ext}', dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('which', choices=('length', 'replication', 'followup'))
    args = ap.parse_args()
    dict(length=length_figure, replication=replication_figure, followup=followup_figure)[args.which]()


if __name__ == '__main__':
    main()
