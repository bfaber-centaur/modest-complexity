#!/usr/bin/env python3
"""R4 figures from the published run files (runs/<ID>/metrics.jsonl) and the
registry. Every planned run appears; a missing, failed or capped run is named
on its row rather than dropped.

uv run --locked python research/regimes/routes/figures.py [--followup]
"""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CONDITIONS = ['rho1', 'rho31of32', 'rho15of16', 'rho7of8', 'rho3of4', 'rho1of2', 'shortcut_only']
LABELS = {'rho1': 'ρ = 1', 'rho31of32': 'ρ = 31/32', 'rho15of16': 'ρ = 15/16', 'rho7of8': 'ρ = 7/8',
          'rho3of4': 'ρ = 3/4', 'rho1of2': 'ρ = 1/2 (rule-only)', 'shortcut_only': 'shortcut-only\n(label = s)'}
SEEDS = [54200, 54201, 54202]
STYLE = {54200: '-', 54201: '--', 54202: ':'}
C = dict(train='black', exc='tab:purple', reliable='tab:green', independent='tab:gray', reversed='tab:red',
         s='tab:orange', rule='tab:blue', irr='tab:gray')
PANELS = [  # (title, [(metric, color, label)], y-limits or None)
    ('train accuracy', [('train_acc', C['train'], 'all'), ('train_acc_exceptions', C['exc'], 'exception bases')], (-.03, 1.03)),
    ('held-out accuracy (target y)', [('acc_reliable', C['reliable'], 'reliable (s = y)'),
                                      ('acc_independent', C['independent'], 'independent'),
                                      ('acc_reversed', C['reversed'], 'reversed (s = 1 − y)')], (-.03, 1.03)),
    ('prediction flip rate, held-out\n(once reliable acc. = 1: flip s = 1 − reversed acc., flip rule = reversed acc.)', [('flip_s', C['s'], 'flip s'), ('flip_rule', C['rule'], 'flip a rule bit'),
                                        ('flip_irr', C['irr'], 'flip an irrelevant bit')], (-.03, 1.03)),
    ('logit-margin change, held-out', [('delta_s', C['s'], 'signed, flip s'), ('delta_rule', C['rule'], 'signed, flip a rule bit'),
                                       ('absdelta_irr', C['irr'], '|Δ|, flip an irrelevant bit')], None),
]


def load(prefix, conditions, seeds):
    """Every attempt of every planned (condition, seed). The attempt shown is the
    last complete one (else the last attempt); the others stay in 'attempts' and
    in the CSV, and the figure names them."""
    registry = [json.loads(l) for l in (HERE/'registry.jsonl').read_text().splitlines() if l.strip()]
    status = {e['run_id']: e['status'] for e in registry}  # last event wins
    planned = {r['run_id'] for r in json.loads((HERE/'plan.json').read_text())['runs']}
    runs = {}
    for cond in conditions:
        for seed in seeds:
            attempts = []
            for a in range(1, 10):
                rid = f'{prefix}{cond}-s{seed}-a{a}'
                if rid not in planned:
                    continue
                path = HERE/'runs'/rid/'metrics.jsonl'
                rows = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
                attempts.append(dict(run_id=rid, status=status.get(rid, 'not run'), rows=rows))
            done = [x for x in attempts if x['status'] == 'complete']
            shown = (done or attempts or [dict(run_id=f'{prefix}{cond}-s{seed}-a1', status='not planned', rows=[])])[-1]
            runs[cond, seed] = dict(shown, attempts=attempts,
                                    note=[f"{x['run_id'][-2:]} {x['status']}" for x in attempts if x is not shown])
    return runs


def write_csv(runs, path, last_only=False):
    keys = ['run_id', 'condition', 'init_seed', 'status', 'step', 'train_loss', *(m for _, ms, _ in PANELS for m, _, _ in ms),
            'train_acc_conforming', 'absdelta_s', 'absdelta_rule', 'param_norm', 'mean_update_norm']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, keys, extrasaction='ignore'); w.writeheader()
        for (cond, seed), r in runs.items():
            for x in r['attempts'] or [r]:
                if last_only and x['run_id'] != r['run_id']:
                    continue
                for row in (x['rows'][-1:] if last_only else x['rows']) or [{}]:
                    w.writerow(dict(run_id=x['run_id'], condition=cond, init_seed=seed, status=x['status'], **row))


def over_training(runs, conditions, updates, out, title):
    fig, axes = plt.subplots(len(conditions), len(PANELS), figsize=(17, 2.15*len(conditions)+1.2),
                             sharex=True, squeeze=False)
    dmax = max([abs(row[m]) for r in runs.values() for row in r['rows'] for m in ('delta_s', 'delta_rule', 'absdelta_irr')] + [1])
    for i, cond in enumerate(conditions):
        for j, (ptitle, metrics, ylim) in enumerate(PANELS):
            ax = axes[i, j]
            for seed in SEEDS:
                rows = runs[cond, seed]['rows']
                for m, color, label in metrics:
                    pts = [(r['step'], r[m]) for r in rows if r.get(m) is not None]
                    if pts:
                        ax.plot(*zip(*pts), STYLE[seed], color=color, lw=1.1, alpha=.85,
                                label=label if seed == SEEDS[0] else None)
            if j in (1, 2):
                ax.axhline(.5 if j == 1 else 0, color='k', lw=.6, ls=(0, (1, 3)))
            if j == 3:
                ax.axhline(0, color='k', lw=.6, ls=(0, (1, 3)))
            ax.set_ylim(ylim if ylim else (-.05*dmax, 1.05*dmax))
            ax.set_xscale('symlog', linthresh=1, linscale=.3); ax.set_xlim(0, updates)
            ax.grid(alpha=.25, lw=.5)
            if i == 0:
                ax.set_title(ptitle, fontsize=10)
            if i == len(conditions) - 1:
                ax.set_xlabel('full-batch updates (symlog; step 0 at left edge)', fontsize=8)
            if j == 0:
                ax.set_ylabel(LABELS[cond], fontsize=10)
            bad = [f"s{seed}: {runs[cond, seed]['status']}" for seed in SEEDS if runs[cond, seed]['status'] != 'complete']
            if bad and j == 0:
                ax.text(.02, .5, '\n'.join(bad), transform=ax.transAxes, color='red', fontsize=8)
            ax.tick_params(labelsize=7)
    for j in range(len(PANELS)):
        # Explicit handles, because row 1 (rho = 1) has no exception-base trace to label.
        # Placed where row 1's curves leave the panel empty (checked on the exported PNG).
        handles = [plt.Line2D([], [], color=c, lw=1.1, label=label) for _, c, label in PANELS[j][1]]
        axes[0, j].legend(handles=handles, fontsize=7, loc='upper left' if j == 3 else 'upper right',
                          bbox_to_anchor=None if j == 3 else (1, .93), framealpha=.85)
    seed_handles = [plt.Line2D([], [], color='k', ls=STYLE[s], lw=1.1, label=f'init seed {s}') for s in SEEDS]
    fig.legend(handles=seed_handles, loc='upper right', ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title, x=.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .975))
    for ext in ('png', 'svg'):
        fig.savefig(out.with_suffix('.'+ext), dpi=130)
    plt.close(fig)


def at_end(runs, conditions, out, title):
    finals = {k: (r['rows'][-1] if r['status'] == 'complete' and r['rows'] else None) for k, r in runs.items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = {c: i for i, c in enumerate(conditions)}
    jitter = {54200: -.12, 54201: 0, 54202: .12}
    for (cond, seed), f in finals.items():
        if f is None:
            for ax in axes:
                ax.text(x[cond], .5, '×', color='red', ha='center', transform=ax.get_xaxis_transform())
            continue
        xx = x[cond] + jitter[seed]
        axes[0].plot(xx, f['acc_reversed'], 'o', color=C['reversed'], ms=5)
        if f['train_acc_exceptions'] is not None:
            axes[0].plot(xx, f['train_acc_exceptions'], 's', color=C['exc'], ms=5, mfc='none')
        axes[1].plot(xx, f['delta_s'], 'o', color=C['s'], ms=5)
        axes[1].plot(xx, f['delta_rule'], 'D', color=C['rule'], ms=5)
        share = f['delta_rule'] / (abs(f['delta_s']) + abs(f['delta_rule']))
        axes[2].plot(xx, share, 'o', color='k', ms=5)
    axes[0].plot([], [], 'o', color=C['reversed'], label='held-out reversed accuracy')
    axes[0].plot([], [], 's', color=C['exc'], mfc='none', label='train accuracy, exception bases')
    axes[1].plot([], [], 'o', color=C['s'], label='signed Δ margin, flip s')
    axes[1].plot([], [], 'D', color=C['rule'], label='signed Δ margin, flip a rule bit')
    axes[0].set_ylim(-.03, 1.03); axes[0].axhline(.5, color='k', lw=.6, ls=(0, (1, 3)))
    axes[1].axhline(0, color='k', lw=.6, ls=(0, (1, 3)))
    axes[2].set_ylim(-.05, 1.05); axes[2].axhline(.5, color='k', lw=.6, ls=(0, (1, 3)))
    axes[2].set_title('rule share = Δrule / (|Δs| + |Δrule|)', fontsize=10)
    axes[0].set_title('accuracy at the last step', fontsize=10); axes[1].set_title('logit-margin change at the last step', fontsize=10)
    for ax in axes:
        ax.set_xticks(range(len(conditions)), [LABELS[c].replace('\n', ' ') for c in conditions], rotation=30, ha='right', fontsize=8)
        ax.grid(alpha=.25, lw=.5)
    axes[0].legend(fontsize=7, loc='center right'); axes[1].legend(fontsize=7, loc='center right')
    fig.suptitle(title, x=.01, ha='left', fontsize=11)
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out.with_suffix('.'+ext), dpi=130)
    plt.close(fig)


def rule_share(row):
    # Undefined while both changes are tiny (early training); drawn as a gap there.
    den = abs(row['delta_s']) + abs(row['delta_rule'])
    return row['delta_rule'] / den if den > .5 else float('nan')


def followup(runs, conditions, updates, out, title):
    fig, axes = plt.subplots(len(conditions), 4, figsize=(17, 2.6*len(conditions)+1.2), sharex=True, squeeze=False)
    dmax = max([abs(row[m]) for r in runs.values() for row in r['rows'] for m in ('delta_s', 'delta_rule')] + [1])
    for i, cond in enumerate(conditions):
        ax = axes[i]; twin = ax[3].twinx()
        for seed in SEEDS:
            rows, ls = runs[cond, seed]['rows'], STYLE[seed]
            first = seed == SEEDS[0]
            col = lambda m: ([r['step'] for r in rows if r.get(m) is not None], [r[m] for r in rows if r.get(m) is not None])
            for m, c, label in (('train_acc', C['train'], 'train'), ('acc_reliable', C['reliable'], 'held-out reliable'),
                                ('acc_reversed', C['reversed'], 'held-out reversed')):
                ax[0].plot(*col(m), ls, color=c, lw=1.1, label=label if first else None)
            for m, c, label in (('delta_s', C['s'], 'signed Δ margin, flip s'), ('delta_rule', C['rule'], 'signed Δ margin, flip a rule bit')):
                ax[1].plot(*col(m), ls, color=c, lw=1.1, label=label if first else None)
            ax[2].plot([r['step'] for r in rows], [rule_share(r) for r in rows], ls, color='k', lw=1.1,
                       label='Δrule / (|Δs| + |Δrule|)' if first else None)
            ax[3].plot(*col('mean_update_norm'), ls, color='tab:purple', lw=1.1, label='‖Δparams‖ per update' if first else None)
            ax[3].plot(*col('adam_sqrt_vhat_median'), ls, color='tab:brown', lw=1.1, label='median √v̂' if first else None)
            twin.plot(*col('train_frac_saturated'), ls, color='tab:gray', lw=1.1, label='saturated train rows (right)' if first else None)
            r = runs[cond, seed]
            if r['status'] != 'complete' or r['note']:
                msg = f"s{seed}: {r['run_id'][-2:]} {r['status']} shown" + ''.join(f'; {n}' for n in r['note'])
                ax[0].text(.35, .36 - .09*SEEDS.index(seed), msg, transform=ax[0].transAxes, color='red', fontsize=7)
        ax[0].set_ylim(-.03, 1.03); ax[0].axhline(.5, color='k', lw=.6, ls=(0, (1, 3)))
        ax[1].set_ylim(-.05*dmax, 1.05*dmax); ax[1].axhline(0, color='k', lw=.6, ls=(0, (1, 3)))
        ax[2].set_ylim(-.05, 1.05); ax[2].axhline(.5, color='k', lw=.6, ls=(0, (1, 3)))
        ax[3].set_yscale('log'); ax[3].axhline(1e-8, color='tab:brown', lw=.8, ls=(0, (1, 3)))
        ax[3].text(.02, 1e-8, 'Adam eps', color='tab:brown', fontsize=7, va='bottom', transform=ax[3].get_yaxis_transform())
        twin.set_ylim(-.03, 1.03); twin.tick_params(labelsize=7, colors='tab:gray')
        ax[0].set_ylabel(LABELS[cond], fontsize=10)
        for j, a in enumerate(ax):
            a.set_xscale('symlog', linthresh=1, linscale=.3); a.set_xlim(0, updates)
            a.axvline(20000, color='k', lw=.8, ls='--', alpha=.6)
            a.grid(alpha=.25, lw=.5); a.tick_params(labelsize=7)
            if i == len(conditions) - 1:
                a.set_xlabel('full-batch updates (symlog; dashed line = initial T)', fontsize=8)
        if i == 0:
            for a, t in zip(ax, ('accuracy', 'logit-margin change, held-out', 'rule share of the two signed changes', 'numerics (log scale)')):
                a.set_title(t, fontsize=10)
            for a in ax[:3]:
                a.legend(fontsize=7, loc='upper left', framealpha=.85)
            h1, l1 = ax[3].get_legend_handles_labels(); h2, l2 = twin.get_legend_handles_labels()
            ax[3].legend(h1 + h2, l1 + l2, fontsize=7, loc='lower left', framealpha=.85)
    seed_handles = [plt.Line2D([], [], color='k', ls=STYLE[s], lw=1.1, label=f'init seed {s}') for s in SEEDS]
    fig.legend(handles=seed_handles, loc='upper right', ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title, x=.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .965))
    for ext in ('png', 'svg'):
        fig.savefig(out.with_suffix('.'+ext), dpi=130)
    plt.close(fig)


def bitwise_check(runs):
    """Every follow-up attempt's step-20,000 checkpoint against the initial run's
    final.npz, and every interrupted attempt's metrics against the shown attempt
    on their shared steps (same seed and code, so they must agree exactly)."""
    import numpy as np
    import sys
    sys.path.insert(0, str(HERE.parent/'common'))
    import nets
    out = {}
    for (cond, seed), r in runs.items():
        a_path = HERE/'runs'/f'r1-route-{cond}-s{seed}-a1'/'final.npz'
        for x in r['attempts']:
            b_path = HERE/'runs'/x['run_id']/'step20000.npz'
            if not (a_path.exists() and b_path.exists()):
                out[x['run_id'] + ' step20000'] = 'missing'; continue
            (a, sa, _), (b, sb, _) = nets.load(a_path), nets.load(b_path)
            same = (sa['t'] == sb['t'] and all(np.array_equal(a[k], b[k]) and np.array_equal(sa['m'][k], sb['m'][k])
                                                and np.array_equal(sa['v'][k], sb['v'][k]) for k in a))
            out[x['run_id'] + ' step20000 vs initial final.npz'] = 'bitwise equal' if same else 'DIFFERENT'
            if x['run_id'] != r['run_id'] and x['rows']:
                shown = {row['step']: row for row in r['rows']}
                shared = [row for row in x['rows'] if row['step'] in shown]
                ok = all(shown[row['step']] == row for row in shared)
                out[f"{x['run_id']} metrics vs {r['run_id']}"] = (
                    f"identical on all {len(shared)} shared logged steps (to step {shared[-1]['step']})" if ok else 'DIFFERENT')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--followup', action='store_true', help='render the follow-up figure instead of the initial comparison')
    args = ap.parse_args()
    figs = HERE/'figures'; figs.mkdir(exist_ok=True)
    if not args.followup:
        runs = load('r1-route-', CONDITIONS, SEEDS)
        write_csv(runs, figs/'routes_over_training.csv')
        over_training(runs, CONDITIONS, 20000, figs/'routes_over_training',
                      'R4 initial comparison: 3-bit parity rule vs shortcut bit s, by training reliability ρ of s '
                      '(10-128-2 ReLU, full-batch Adam, 256 training inputs, 3 paired init seeds; held-out = 256 unseen bases)')
        at_end(runs, CONDITIONS, figs/'routes_at_T',
               'R4 initial comparison at T = 20,000 updates (each point one init seed)')
        write_csv(runs, figs/'routes_at_T.csv', last_only=True)
    else:
        long_conditions = ['rho1', 'rho31of32', 'rho15of16']
        runs = load('r1-route-long-', long_conditions, SEEDS)
        write_csv(runs, figs/'followup_long.csv')
        followup(runs, long_conditions, 200000, figs/'followup_long',
                 'R4 follow-up (adaptive, separate from the initial comparison): the same recipe and seeds trained 10× longer '
                 '(200,000 updates)')
        check = bitwise_check(runs)
        (figs/'followup_bitwise_check.json').write_text(json.dumps(check, indent=1) + '\n')
        print(check)


if __name__ == '__main__':
    main()
