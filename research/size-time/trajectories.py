#!/usr/bin/env python3
"""Per-run trajectory descriptors and width-by-time figures for the e1 map.

Descriptive only. Every planned run appears; unlaunched or unfinished runs are
listed with the steps they reached and their descriptors are censored there.
Checkpoints/log steps of one run are repeated measurements, not replicates.

Thresholds (visual/descriptive, revisable): fit = train accuracy >= .99,
generalize = audit accuracy >= .95; "sustained" = the threshold holds at every
later logged step. Matched-loss rows use the first logged step at which train
cross-entropy <= each level.

uv run --locked --group analysis python research/size-time/trajectories.py \
    --runs RUN_ROOT ... --out research/size-time/figures/trajectories
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from report_map import CONDITIONS, SEED_STYLES, WIDTH_COLORS, latest_status, load_rows  # noqa: E402

FIT, GEN = .99, .95
MATCHED_STEPS = [1000, 5000, 10000, 20000, 50000]
LOSS_LEVELS = [1e-1, 1e-2, 1e-3, 1e-4]


def condition_of(run):
    return next(name for name, (task, wd) in CONDITIONS.items() if (run['task'], run['wd']) == (task, wd))


def first_and_sustained(steps, ok):
    first = next((s for s, o in zip(steps, ok) if o), None)
    sustained = None
    for s, o in zip(reversed(steps), reversed(ok)):
        if not o:
            break
        sustained = s
    return first, sustained


def descriptors(run, rows, status):
    steps = [z['step'] for z in rows]
    tr = np.array([z['train'] for z in rows]); au = np.array([z['audit'] for z in rows])
    ce = np.array([z['objective']['train']['cross_entropy'] for z in rows])
    first_fit, sus_fit = first_and_sustained(steps, tr >= FIT)
    first_gen, sus_gen = first_and_sustained(steps, au >= GEN)
    after = [i for i, s in enumerate(steps) if first_fit is not None and s >= first_fit]
    # Largest rise of train CE over its running minimum (a late spike) after first fit.
    spike = max((ce[i]/ce[:i].min() for i in after if i > 0), default=None)
    last = rows[-1]
    o = last['objective']
    return dict(run_id=run['run_id'], condition=condition_of(run), width=run['width'], init_seed=run['init_seed'],
                status=status, last_logged_step=steps[-1], first_fit=first_fit, sustained_fit=sus_fit,
                fit_drops_after_first=int(sum(tr[i] < FIT for i in after)),
                first_gen=first_gen, sustained_gen=sus_gen, max_audit=float(au.max()),
                step_max_audit=steps[int(au.argmax())], final_train=last['train'], final_audit=last['audit'],
                final_train_ce=o['train']['cross_entropy'], final_audit_ce=o['audit']['cross_entropy'],
                min_train_ce=float(ce.min()), step_min_train_ce=steps[int(ce.argmin())],
                max_train_ce_rise_after_fit=spike, final_train_margin_min=o['train']['margin_min'],
                final_audit_margin_median=o['audit']['margin_median'], final_norm=last['norm'])


def matched(run, rows):
    by_step = {z['step']: z for z in rows}
    out = []
    for kind, levels in (('step', MATCHED_STEPS), ('train_ce_le', LOSS_LEVELS)):
        for level in levels:
            if kind == 'step':
                z = by_step.get(level)
            else:
                z = next((z for z in rows if z['objective']['train']['cross_entropy'] <= level), None)
            o = z['objective'] if z else None
            out.append(dict(run_id=run['run_id'], condition=condition_of(run), width=run['width'],
                            init_seed=run['init_seed'], match=kind, level=level, reached=z is not None,
                            step=z['step'] if z else None, train_acc=z['train'] if z else None,
                            audit_acc=z['audit'] if z else None,
                            train_ce=o['train']['cross_entropy'] if o else None,
                            audit_ce=o['audit']['cross_entropy'] if o else None,
                            train_margin_min=o['train']['margin_min'] if o else None,
                            audit_margin_median=o['audit']['margin_median'] if o else None,
                            norm=z['norm'] if z else None))
    return out


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))  # unlaunched runs have fewer keys
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def save(fig, out, name, note):
    fig.text(.01, -.01, note, fontsize=7, ha='left', va='top', color='0.3', wrap=True)
    for ext in ('png', 'svg'):
        fig.savefig(out/f'{name}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)


def overview(runs, curves, status, steps_total, p, out):
    """Heatmap: one row per planned run, one column per logged step > 0 (log x axis)."""
    steps = np.arange(100, steps_total+1, 100)
    edges = np.concatenate([[70], steps[:-1]+50, [steps[-1]+50]])
    chance = 1/p
    panels = [('train', 'train accuracy', 'viridis', 0, 1), ('audit', 'audit accuracy', 'viridis', 0, 1),
              ('audit', f'audit accuracy, zoomed 0 to 2/{p} (white = chance)', 'PuOr', 0, 2*chance)]
    fig, axes = plt.subplots(1, 3, figsize=(17, .26*len(runs)+1.8), sharey=True, layout='constrained')
    for ax, (split, title, cmap, lo, hi) in zip(axes, panels):
        z = np.full((len(runs), len(steps)), np.nan)
        for i, r in enumerate(runs):
            for row in curves[r['run_id']]:
                if row['step'] > 0:
                    z[i, row['step']//100-1] = row[split]
        cm = plt.get_cmap(cmap).with_extremes(over='#1a9850', bad='0.85')
        mesh = ax.pcolormesh(edges, np.arange(len(runs)+1), np.ma.masked_invalid(z), cmap=cm, vmin=lo, vmax=hi,
                             shading='flat')
        ax.set_xscale('log'); ax.set_xlim(70, steps_total*1.05); ax.set_title(title, fontsize=10)
        ax.set_xlabel('update step (log scale)')
        fig.colorbar(mesh, ax=ax, shrink=.8, location='bottom', extend='max' if hi < 1 else 'neither')
    labels = [f"{condition_of(r)} w{r['width']} i{r['init_seed']}" +
              ('' if status.get(r['run_id'], 'planned') == 'complete' else f" [{status.get(r['run_id'], 'planned')}]")
              for r in runs]
    axes[0].set_yticks(np.arange(len(runs))+.5, labels, fontsize=7); axes[0].invert_yaxis()
    save(fig, out, 'overview', f'Unsmoothed logged values every 100 steps; step 0 omitted (in the run metrics). '
         f'Light gray = not logged (planned, running or capped). Zoomed panel: chance 1/{p} is white, green = above '
         f'2/{p} (clipped). Rows sorted by condition, width, init seed; shuffled-label rows are scored against their '
         f'own shuffled labels.')


def run_label(r, status, prefix=''):
    st = status.get(r['run_id'], 'planned')
    return f"{prefix}init {r['init_seed']}" + ('' if st == 'complete' else f' ({st})')


def per_width(runs, curves, status, out, name, title, series, ylabel, yscale='linear', note='', n_params=None):
    widths = sorted({r['width'] for r in runs})
    fig, axes = plt.subplots(len(series), len(widths), figsize=(3*len(widths)+.6, 2.6*len(series)+.4),
                             sharex=True, sharey='row', squeeze=False, layout='constrained')
    seeds = sorted({r['init_seed'] for r in runs})
    for col, width in enumerate(widths):
        for row_i, (label, get) in enumerate(series):
            if get is None:  # raw norm divided by sqrt(parameter count) for this width
                get = lambda z, n=n_params(width): z['norm']/math.sqrt(n)
            ax = axes[row_i, col]
            for r in (r for r in runs if r['width'] == width):
                pts = [z for z in curves[r['run_id']] if z['step'] > 0 and get(z) is not None]
                if not pts:
                    continue
                ax.plot([z['step'] for z in pts], [get(z) for z in pts], SEED_STYLES[seeds.index(r['init_seed']) % 4],
                        color=WIDTH_COLORS.get(width, 'k'), lw=1.1, label=run_label(r, status))
                if status.get(r['run_id']) != 'complete':
                    ax.plot(pts[-1]['step'], get(pts[-1]), 'x', color='k', ms=5)
            ax.set_xscale('log'); ax.set_yscale(yscale); ax.grid(True, color='0.9', lw=.6)
            if row_i == 0:
                ax.set_title(f'width {width}')
            if col == 0:
                ax.set_ylabel(f'{label}\n{ylabel[row_i]}')
            if row_i == len(series)-1:
                ax.set_xlabel('update step (log scale)')
            if row_i == 0 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=11)
    save(fig, out, name, 'Unsmoothed, one line per run; step 0 omitted on the log axis; x = last logged step of an '
         'unfinished run. '+note)


COND_COLORS = {'add': '#3a9a5b', 'add0': '#b03a8c', 'shuf': '#555555'}


def loss_matched(runs, curves, status, out, name, title, by_condition=False):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    seeds = sorted({r['init_seed'] for r in runs})
    for r in runs:
        pts = curves[r['run_id']]
        if not pts:
            continue
        x = [z['objective']['train']['cross_entropy'] for z in pts]
        color = COND_COLORS[condition_of(r)] if by_condition else WIDTH_COLORS.get(r['width'], 'k')
        style = dict(color=color, lw=1, ls=SEED_STYLES[seeds.index(r['init_seed']) % 4])
        label = run_label(r, status, f"{condition_of(r)} " if by_condition else f"w{r['width']} ")
        axes[0].plot(x, [z['audit'] for z in pts], label=label, **style)
        axes[1].plot(x, [z['objective']['audit']['cross_entropy'] for z in pts], **style)
        axes[2].plot(x, [z['objective']['audit']['margin_median'] for z in pts], **style)
    for ax, ylabel in zip(axes, ('audit accuracy', 'audit cross-entropy (nats)', 'audit median signed margin (logit)')):
        ax.set_xscale('log'); ax.invert_xaxis(); ax.grid(True, color='0.9', lw=.6)
        ax.set_xlabel('train cross-entropy (nats, log scale, decreasing to the right)'); ax.set_ylabel(ylabel)
    axes[0].axhline(1/31, color='0.55', lw=.8, ls=(0, (1, 2)))
    top = max((z['audit'] for r in runs for z in curves[r['run_id']]), default=0)
    if top > .2:
        axes[0].set_ylim(-.02, 1.02)
    else:  # nothing approaches generalization; a 0..1 axis would flatten everything onto zero
        axes[0].set_ylim(-.002, .1); axes[0].set_ylabel('audit accuracy (zoomed: 0 to 0.1)')
    axes[2].axhline(0, color='0.55', lw=.8)
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    fig.suptitle(title, fontsize=11)
    save(fig, out, name, 'Each line is one run traced through its logged steps (every 100 updates), parametrized by its own '
         'train loss, so runs are compared at comparable training loss rather than at matched steps. '
         'Dotted line: chance 1/31. Margin = correct logit minus best incorrect logit.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', type=Path, default=Path('research/plan.json'))
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--registries', type=Path, nargs='*', help='Default: research/*/registry.jsonl')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    registries = args.registries if args.registries is not None else sorted(Path('research').glob('*/registry.jsonl'))
    status = {k: v['status'] for k, v in latest_status(registries).items()}
    order = list(CONDITIONS)
    runs = sorted(plan['runs'], key=lambda r: (order.index(condition_of(r)), r['width'], r['init_seed']))
    curves = {}
    for r in runs:
        run_dir = next((root/r['run_id'] for root in args.runs if (root/r['run_id']).exists()), None)
        curves[r['run_id']] = load_rows(run_dir) if run_dir else []
    args.out.mkdir(parents=True, exist_ok=True)
    summary = [descriptors(r, curves[r['run_id']], status.get(r['run_id'], 'planned')) if curves[r['run_id']] else
               dict(run_id=r['run_id'], condition=condition_of(r), width=r['width'], init_seed=r['init_seed'],
                    status=status.get(r['run_id'], 'planned')) for r in runs]
    write_csv(args.out/'summary.csv', summary)
    write_csv(args.out/'matched.csv', [m for r in runs if curves[r['run_id']] for m in matched(r, curves[r['run_id']])])

    overview(runs, curves, status, plan['common']['steps'], plan['common']['p'], args.out)
    add = [r for r in runs if condition_of(r) == 'add']
    margin = lambda split, k: (lambda z: z['objective'][split][k])
    per_width(add, curves, status, args.out, 'margins_add', 'add (wd=1): signed margins by width',
              [('train', margin('train', 'margin_min')), ('audit', margin('audit', 'margin_median'))],
              ['minimum margin (logit)', 'median margin (logit)'],
              note='Margin = correct logit minus best incorrect logit; linear axis; > 0 means correct.')
    upd = lambda k: (lambda z: z['optimizer_update'][k] if z['optimizer_update'] else None)
    per_width(add, curves, status, args.out, 'updates_add', 'add (wd=1): actual last-step AdamW displacement by width',
              [('adaptive', upd('adaptive_norm')), ('decay', upd('decay_norm')), ('cosine', upd('cosine'))],
              ['norm', 'norm', 'cos(adaptive, decay)'], yscale='linear',
              note='Measured at each logged step for that single update (not an interval average).')
    p = plan['common']['p']
    per_width(add, curves, status, args.out, 'norm_add', 'add (wd=1): parameter norm by width (descriptive)',
              [('raw', lambda z: z['norm']), ('width-normalized', None)],
              ['L2 norm of all weights', f'norm / sqrt(3*{p}*width)'], n_params=lambda w: 3*p*w,
              note='Descriptive quantities only; a norm is not a complexity measure.')
    loss_matched(add, curves, status, args.out, 'loss_matched_add', 'add (wd=1): held-out behavior at comparable train loss')
    ctrl = [r for r in runs if r['width'] == 128]
    loss_matched(ctrl, curves, status, args.out, 'loss_matched_w128_controls',
                 'width 128: add (wd=1), add0 (wd=0) and shuf (wd=1) at comparable train loss', by_condition=True)
    for s in summary:
        print(s['run_id'], s['status'], s.get('last_logged_step'), 'fit', s.get('first_fit'), 'gen', s.get('first_gen'),
              'final audit', s.get('final_audit'))


if __name__ == '__main__':
    main()
