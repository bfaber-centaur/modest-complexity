#!/usr/bin/env python3
"""Status table, source CSV and unsmoothed learning-curve figures for planned runs.

Reads the plan, every lane registry and whatever metrics exist (complete, running
or capped). Incomplete trajectories end at their last logged step and are marked;
nothing is interpolated or extended.

uv run --locked --group analysis python report_map.py --runs runs/e1 --out research/figures/NAME
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WIDTHS = [16, 32, 64, 128, 256, 512]
WIDTH_COLORS = dict(zip(WIDTHS, ['#9e3d22', '#d36b1f', '#c9a227', '#3a9a5b', '#2a7bb6', '#5a3f9e']))
SEED_STYLES = ['-', '--', ':', '-.']
SPLITS = [('train', 'train split'), ('audit', 'held-out audit split')]
CONDITIONS = {'add': ('add', 1), 'add0': ('add', 0), 'shuf': ('random', 1)}


def latest_status(registries):
    status = {}
    for path in registries:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                event = json.loads(line)
                status[event['run_id']] = event  # files are append-only; last event wins
    return status


def load_rows(run_dir):
    path = run_dir/'metrics.jsonl'
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # a partially written last line of a running job
            break
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', type=Path, default=Path('research/plan.json'))
    ap.add_argument('--runs', type=Path, nargs='+', required=True, help='Roots holding <run_id>/metrics.jsonl')
    ap.add_argument('--registries', type=Path, nargs='*', help='Default: research/*/registry.jsonl')
    ap.add_argument('--condition', choices=sorted(CONDITIONS), default='add')
    ap.add_argument('--ids', nargs='*', help='Restrict to these run IDs')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    registries = args.registries if args.registries is not None else sorted(Path('research').glob('*/registry.jsonl'))
    status = latest_status(registries)
    task, wd = CONDITIONS[args.condition]
    runs = [r for r in plan['runs'] if r['task'] == task and r['wd'] == wd and (not args.ids or r['run_id'] in args.ids)]
    args.out.mkdir(parents=True, exist_ok=True)

    table, curves = [], {}
    for r in runs:
        run_dir = next((root/r['run_id'] for root in args.runs if (root/r['run_id']).exists()), None)
        rows = load_rows(run_dir) if run_dir else []
        curves[r['run_id']] = rows
        event = status.get(r['run_id'], {})
        state = event.get('status', 'planned')
        last = rows[-1] if rows else {}
        table.append(dict(run_id=r['run_id'], width=r['width'], init_seed=r['init_seed'], status=state,
                          last_logged_step=last.get('step'), planned_steps=plan['common']['steps'],
                          reason=event.get('reason') or '', session_label=event.get('session_label', ''),
                          train_acc=last.get('train'), audit_acc=last.get('audit'),
                          train_ce=last.get('objective', {}).get('train', {}).get('cross_entropy'),
                          audit_ce=last.get('objective', {}).get('audit', {}).get('cross_entropy'),
                          audit_margin_median=last.get('objective', {}).get('audit', {}).get('margin_median'),
                          seconds_at_last_log=last.get('seconds')))

    with (args.out/'status.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    fields = ['run_id', 'width', 'init_seed', 'step', 'train', 'cal', 'audit', 'norm', 'seconds']
    obj = [f'{s}_{k}' for s in ('train', 'cal', 'audit') for k in ('cross_entropy', 'margin_min', 'margin_p05', 'margin_median')]
    with (args.out/'curves.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(fields+obj)
        for r in runs:
            for row in curves[r['run_id']]:
                writer.writerow([r['run_id'], r['width'], r['init_seed']]+[row[k] for k in fields[3:]]+
                                [row['objective'][s][k] for s in ('train', 'cal', 'audit')
                                 for k in ('cross_entropy', 'margin_min', 'margin_p05', 'margin_median')])

    widths = sorted({r['width'] for r in runs})
    seeds = sorted({r['init_seed'] for r in runs})
    states = {t['run_id']: t for t in table}
    for metric, ylabel in (('accuracy', 'accuracy'), ('loss', 'cross-entropy (nats, log scale)')):
        fig, axes = plt.subplots(2, len(widths), figsize=(3.2*len(widths)+.6, 5.6), sharex=True, sharey=True,
                                 squeeze=False, layout='constrained')
        for col, width in enumerate(widths):
            for row_i, (split, split_label) in enumerate(SPLITS):
                ax = axes[row_i, col]
                for r in (r for r in runs if r['width'] == width):
                    pts = [z for z in curves[r['run_id']] if z['step'] > 0]
                    if not pts:
                        continue
                    xs = [z['step'] for z in pts]
                    ys = [z[split] if metric == 'accuracy' else z['objective'][split]['cross_entropy'] for z in pts]
                    st = states[r['run_id']]
                    label = f"init {r['init_seed']}" + ('' if st['status'] == 'complete' else
                                                        f" ({st['status']} @ step {st['last_logged_step']})")
                    ax.plot(xs, ys, SEED_STYLES[seeds.index(r['init_seed']) % 4], color=WIDTH_COLORS.get(width, 'k'),
                            lw=1.3, label=label)
                    if st['status'] != 'complete':
                        ax.plot(xs[-1], ys[-1], 'x', color='k', ms=6, mew=1.5)
                if metric == 'accuracy':
                    ax.axhline(1/plan['common']['p'], color='0.55', lw=.8, ls=(0, (1, 2)))
                    ax.set_ylim(-.02, 1.02)
                else:
                    ax.set_yscale('log')
                ax.set_xscale('log'); ax.set_xlim(100, plan['common']['steps']*1.05)
                ax.grid(True, which='major', color='0.9', lw=.6)
                if row_i == 0:
                    ax.set_title(f'width {width}')
                if row_i == 1:
                    ax.set_xlabel('update step (log scale)')
                if col == 0:
                    ax.set_ylabel(f'{split_label}\n{ylabel}')
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=7, loc='best', frameon=False)
        note = ('Unsmoothed, one line per run. x marks the last logged step of an unfinished run. '
                'Step 0 omitted on the log axis (in curves.csv).')
        if metric == 'accuracy':
            note += ' Dotted line: chance 1/31.'
        fig.suptitle(f"{args.condition} (wd={wd}), plan {plan['expedition']}: {metric} by split", fontsize=11)
        fig.text(.01, -.01, note, fontsize=7, ha='left', va='top', color='0.3')
        for ext in ('png', 'svg'):
            fig.savefig(args.out/f'{metric}.{ext}', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(table, indent=1))


if __name__ == '__main__':
    main()
