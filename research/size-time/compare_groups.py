#!/usr/bin/env python3
"""Overlay labeled groups of runs (for example the e1 width-128 runs at a 30% training
share and the e1f2 runs at 50%) on the same axes, with a descriptor table.

Groups may use different splits: each run is scored on its own train and audit
pairs (taken from its data.npz), so held-out sets differ between groups. Thresholds
and descriptors are those of trajectories.py. heldout.png adds operand_labels.py's
fractions at every published checkpoint (written to operand_labels.jsonl).

uv run --locked --group analysis python research/size-time/compare_groups.py \
    --group "e1, 30% train" RUN_DIR ... --group "e1f2, 50% train" RUN_DIR ... --out DIR
"""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from operand_labels import check  # noqa: E402
from trajectories import SEED_STYLES, descriptors, load_rows, save, write_csv  # noqa: E402

GROUP_COLORS = ['#3a9a5b', '#2a5fb6', '#b03a8c', '#555555']
GROUP_MARKERS = [dict(marker='s', ms=7, mfc='none'), dict(marker='o', ms=4), dict(marker='^', ms=5), dict(marker='D', ms=4)]


def load_run(run_dir):
    cfg = json.loads((run_dir/'config.json').read_text())
    runtime = json.loads((run_dir/'runtime.json').read_text())
    with np.load(run_dir/'data.npz') as d:
        sizes = dict(n_train=len(d['train']), n_cal=len(d['cal']), n_audit=len(d['audit']))
    run = dict(run_id=cfg['run_id'], task=cfg['task'], wd=cfg['wd'], width=cfg['hidden'],
               init_seed=cfg['seeds']['init_seed'])
    extra = dict(train_fraction=cfg.get('train_fraction', .3), split_sha256=cfg['data_hashes']['split_sha256'], **sizes)
    return run, load_rows(run_dir), runtime.get('status'), extra, run_dir


def heldout(groups, seeds, out):
    """Held-out accuracy and cross-entropy by step, and where held-out predictions go at each checkpoint."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), layout='constrained')
    results = []
    for color, mk, (label, members) in zip(GROUP_COLORS, GROUP_MARKERS, groups):
        sizes = sorted({(e['n_train'], e['n_audit']) for *_, e, _ in members})
        name = f"{label} ({' / '.join(f'{t} train, {a} audit pairs' for t, a in sizes)})"
        for i, (run, rows, status, extra, run_dir) in enumerate(members):
            ls = SEED_STYLES[seeds.index(run['init_seed']) % 4]
            pts = [z for z in rows if z['step'] > 0]
            axes[0].plot([z['step'] for z in pts], [z['audit'] for z in pts], color=color, lw=1.1, ls=ls,
                         label=f"{name}, init {run['init_seed']}")
            axes[1].plot([z['step'] for z in pts], [z['objective']['audit']['cross_entropy'] for z in pts],
                         color=color, lw=1.1, ls=ls)
            steps = sorted(int(c.stem.split('_')[1]) for c in run_dir.glob('checkpoint_*.npz'))
            res = [check(run_dir, step) for step in steps if step > 0]
            results += res
            axes[2].plot([r['step'] for r in res], [r['pred_in_L'] for r in res], color=color, ls=ls, lw=.9, **mk)
            wrong = [r for r in res if r['wrong_in_L'] is not None]
            axes[3].plot([r['step'] for r in wrong], [r['wrong_in_L'] for r in wrong], color=color, ls=ls, lw=.9, **mk)
            if i == 0:
                axes[3].axhline(res[-1]['base_rate'], color=color, lw=.9, ls=(0, (4, 3)))
    axes[0].set_ylim(-.02, 1.02); axes[0].axhline(1/31, color='0.55', lw=.8, ls=(0, (1, 2)))
    axes[0].set_ylabel('held-out audit accuracy'); axes[1].set_ylabel('held-out audit cross-entropy (nats, log)')
    axes[1].set_yscale('log')
    axes[2].set_ylabel('share of all audit predictions in L(a,b)')
    axes[3].set_ylabel('share of wrong audit predictions in L(a,b)')
    axes[3].set_title('dashed: share expected if wrong predictions were uniform over wrong labels', fontsize=8)
    for ax in axes[2:]:
        ax.set_ylim(-.02, 1.02)
    for ax in axes:
        ax.set_xscale('log'); ax.grid(True, color='0.9', lw=.6); ax.set_xlabel('update step (log scale)')
    axes[0].legend(fontsize=7, frameon=False, loc='center left')
    fig.suptitle('Held-out behavior: ' + ' vs '.join(label for label, _ in groups), fontsize=11)
    save(fig, out, 'heldout', 'First two panels: every 100 updates, unsmoothed, step 0 omitted. Last two: published '
         'checkpoints only (runs publish different steps; squares = first group, circles = second group). L(a,b) = labels of training pairs (a,b\') or (a\',b); '
         'for addition the correct label is never in L, so a correct prediction is never in L. Each run is scored on '
         'its own split, so train and audit sets differ in size and membership between groups. Dotted line: chance 1/31.')
    (out/'operand_labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', nargs='+', action='append', required=True, metavar='LABEL RUN_DIR',
                    help='A label followed by run directories; repeat for each group')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    groups = [(g[0], [load_run(Path(d)) for d in g[1:]]) for g in args.group]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'summary.csv', [dict(group=label, **descriptors(run, rows, status), **extra)
                                       for label, members in groups for run, rows, status, extra, _ in members])
    seeds = sorted({run['init_seed'] for _, members in groups for run, *_ in members})
    lines = lambda: [(color, label, run, rows) for color, (label, members) in zip(GROUP_COLORS, groups)
                     for run, rows, *_ in members]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, layout='constrained')
    for color, label, run, rows in lines():
        pts = [z for z in rows if z['step'] > 0]
        style = dict(color=color, lw=1.1, ls=SEED_STYLES[seeds.index(run['init_seed']) % 4],
                     label=f"{label}, init {run['init_seed']}")
        for i, split in enumerate(('train', 'audit')):
            axes[i, 0].plot([z['step'] for z in pts], [z[split] for z in pts], **style)
            axes[i, 1].plot([z['step'] for z in pts], [z['objective'][split]['cross_entropy'] for z in pts], **style)
    for i, split in enumerate(('train', 'held-out audit')):
        axes[i, 0].set_ylim(-.02, 1.02); axes[i, 0].axhline(1/31, color='0.55', lw=.8, ls=(0, (1, 2)))
        axes[i, 0].set_ylabel(f'{split} accuracy'); axes[i, 1].set_ylabel(f'{split} cross-entropy (nats, log)')
        axes[i, 1].set_yscale('log')
    for ax in axes.flat:
        ax.set_xscale('log'); ax.grid(True, color='0.9', lw=.6)
    for ax in axes[1]:
        ax.set_xlabel('update step (log scale)')
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle('Labeled comparison: ' + ' vs '.join(label for label, _ in groups), fontsize=11)
    save(fig, args.out, 'curves', 'Unsmoothed, one line per run; step 0 omitted. Each run is scored on its own split: '
         'held-out audit pairs differ between groups. Dotted line: chance 1/31.')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    for color, label, run, rows in lines():
        x = [z['objective']['train']['cross_entropy'] for z in rows]
        style = dict(color=color, lw=1, ls=SEED_STYLES[seeds.index(run['init_seed']) % 4])
        axes[0].plot(x, [z['audit'] for z in rows], label=f"{label}, init {run['init_seed']}", **style)
        axes[1].plot(x, [z['objective']['audit']['cross_entropy'] for z in rows], **style)
        axes[2].plot(x, [z['objective']['audit']['margin_median'] for z in rows], **style)
    for ax, ylabel in zip(axes, ('audit accuracy', 'audit cross-entropy (nats)', 'audit median signed margin (logit)')):
        ax.set_xscale('log'); ax.invert_xaxis(); ax.grid(True, color='0.9', lw=.6); ax.set_ylabel(ylabel)
        ax.set_xlabel('train cross-entropy (nats, log scale, decreasing to the right)')
    axes[0].set_ylim(-.02, 1.02); axes[0].axhline(1/31, color='0.55', lw=.8, ls=(0, (1, 2)))
    axes[2].axhline(0, color='0.55', lw=.8)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Held-out behavior at comparable train loss: ' + ' vs '.join(label for label, _ in groups), fontsize=11)
    save(fig, args.out, 'loss_matched', 'Each line is one run traced through its logged steps, parametrized by its own '
         'train loss. Groups have different train and audit sets. Dotted line: chance 1/31.')
    heldout(groups, seeds, args.out)
    for label, members in groups:
        for run, rows, status, extra, _ in members:
            print(label, run['run_id'], status, extra['n_train'], extra['n_audit'], 'final audit', rows[-1]['audit'])


if __name__ == '__main__':
    main()
