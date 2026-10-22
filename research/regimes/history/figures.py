#!/usr/bin/env python3
"""R3 figures and tables from local run directories (or the published copies).

Figure 1, recovery_curves: A and B accuracy aligned at the A->B and B->A switches.
Figure 2, survival_summary: per run, end-of-B A through the original head, frozen-
feature readouts against their references, relearning cost against fresh-A, and B
retained after the return phase.

uv run --locked python research/regimes/history/figures.py --runs runs/regimes
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SEEDS, DS, T_A, T_R = (53200, 53201, 53202), (100, 1000, 10000), 2000, 4000
ROWS = [('rel', 'Brel', 'related B (x0^x1^x3)'), ('unrel', 'Bunrel', 'unrelated B (x4^x5^x6)'),
        ('contA', None, 'continued-A (control)'), ('conflict', None, 'SEPARATE EXPERIMENT: notA through the shared A head')]
BLUE, ORANGE, GREY = '#2a78d6', '#eb6834', '#8a8a8a'
CRIT, GAP = .95, .08


def read_run(root, rid):
    d = Path(root)/rid
    if not (d/'runtime.json').exists():
        return None
    plain = d/'metrics.jsonl'  # published copies are gzipped
    stream = plain.open() if plain.exists() else gzip.open(d/'metrics.jsonl.gz', 'rt')
    with stream:
        rows = [json.loads(line) for line in stream]
    return dict(rows=rows, snaps=json.loads((d/'summary.json').read_text())['snapshots'])


def first(rows, phase, key, below=False):
    """First logged step of `phase` at which `key` meets the criterion, else None (censored)."""
    for r in rows:
        if r['phase'] == phase and (r[key] <= 1-CRIT if below else r[key] >= CRIT):
            return r['step']
    return None


def series(rows, phase, key):
    rr = [r for r in rows if r['phase'] == phase]
    return np.array([r['step'] for r in rr]), np.array([r[key] for r in rr])


def f(s):  # common log1p scale for every segment; step 0 sits exactly at the switch
    return np.log10(1+np.asarray(s, float))/4


def phase_end(snaps, phase):
    return next(s for s in snaps if s['label'] == 'phase_end' and s['phase'] == phase)


def snap_at(snaps, t):
    return next(s for s in snaps if s['t'] == t)


def statuses(registry):
    out = {}
    for line in registry.read_text().splitlines():
        e = json.loads(line)
        if e['status'] != 'running':
            out[e['run_id']] = e['status']
    return out


def recovery_figure(root, runs, status, out):
    widths = [f(T_A)+GAP+f(d)+GAP+f(T_R) for d in DS]
    fig, axes = plt.subplots(len(ROWS), len(DS), figsize=(16, 14), gridspec_kw=dict(width_ratios=widths, hspace=.62, wspace=.08))
    table = []
    for i, (arm, btask, title) in enumerate(ROWS):
        for j, d in enumerate(DS):
            ax = axes[i, j]; xB = f(T_A)+GAP; xR = xB+f(d)+GAP
            for k, s in enumerate(SEEDS):
                rid, aid = f'r1-hist-{arm}-d{d}-s{s}-a1', f'r1-hist-A-s{s}-a1'
                A, run = runs.get(aid), runs.get(rid)
                if A is not None:
                    st, acc = series(A['rows'], 1, 'A_test_acc')
                    ax.plot(f(st), acc, color=BLUE, lw=1.1)
                    ax.plot(f(st), series(A['rows'], 1, 'A_train_acc')[1], color=BLUE, lw=.8, ls='--')
                    ax.plot(xR+f(st), acc, color=GREY, lw=1, zorder=1)  # fresh-A reference in the return segment
                d0 = runs.get(f'r1-hist-d0-s{s}-a1')
                if d0 is not None:
                    st, acc = series(d0['rows'], 1, 'A_test_acc')
                    ax.plot(xR+f(st), acc, color=GREY, lw=1, ls=':', zorder=1)
                if run is None:
                    ax.text(xR+.3, .3-.08*k, f'{rid}: {status.get(rid, "missing")}', fontsize=7, color='red')
                    continue
                for ph, x0 in ((1, xB), (2, xR)):
                    for key, color in [('A', BLUE)] + ([(btask, ORANGE)] if btask else []):
                        st, acc = series(run['rows'], ph, f'{key}_test_acc')
                        ax.plot(x0+f(st), acc, color=color, lw=1.1)
                        ax.plot(x0+f(st), series(run['rows'], ph, f'{key}_train_acc')[1], color=color, lw=.8, ls='--')
                        for (stp, a), tr in zip(zip(st, acc), series(run['rows'], ph, f'{key}_train_acc')[1]):
                            table.append(dict(run_id=rid, arm=arm, d=d, init_seed=s, phase=ph, step=int(stp),
                                              head_task=key, test_acc=a, train_acc=tr))
                t = first(run['rows'], 2, 'A_test_acc')
                if t is None:
                    ax.plot(xR+f(T_R), CRIT, 'x', color='red', ms=8)  # censored relearning
                else:
                    ax.plot(xR+f(t), 1.04, 'v', color=BLUE, ms=5, clip_on=False)
            for x in (xB, xR):
                ax.axvline(x-GAP/2, color='k', lw=.6)
            ax.axhline(.5, color=GREY, lw=.6, ls=':'); ax.axhline(CRIT, color=GREY, lw=.5, ls='--')
            ticks, labels = [], []
            for x0, steps in ((0, (0, 10, 100, 1000)), (xB, [v for v in (0, 10, 100, 1000, 10000) if v <= d]),
                              (xR, (0, 10, 100, 1000))):
                ticks += [x0+f(v) for v in steps]; labels += [str(v) for v in steps]
            ax.set_xticks(ticks, labels, fontsize=6.5, rotation=90); ax.set_ylim(-.03, 1.07); ax.set_xlim(-.02, xR+f(T_R)+.02)
            if j == 0:
                ax.set_ylabel('accuracy')
            else:
                ax.set_yticklabels([])
            seg = 'notA' if arm == 'conflict' else ('A' if arm == 'contA' else btask)
            ax.set_title(f'{title}\nd = {d}', fontsize=9, loc='left')
            ax.text(f(T_A)/2, -.3, 'phase A', ha='center', fontsize=8, transform=ax.get_xaxis_transform())
            ax.text(xB+f(d)/2, -.3, f'phase B: {seg}', ha='center', fontsize=8, transform=ax.get_xaxis_transform())
            ax.text(xR+f(T_R)/2, -.3, 'return to A', ha='center', fontsize=8, transform=ax.get_xaxis_transform())
    handles = [plt.Line2D([], [], color=BLUE, lw=1.1, label='head A on A, held-out'),
               plt.Line2D([], [], color=BLUE, lw=.8, ls='--', label='head A on A, train'),
               plt.Line2D([], [], color=ORANGE, lw=1.1, label='head B on its B task, held-out'),
               plt.Line2D([], [], color=ORANGE, lw=.8, ls='--', label='head B on its B task, train'),
               plt.Line2D([], [], color=GREY, lw=1, label='fresh-A held-out (same seed, from init)'),
               plt.Line2D([], [], color=GREY, lw=1, ls=':', label='d = 0 return held-out'),
               plt.Line2D([], [], color=BLUE, marker='v', ls='', label='relearned: held-out A >= 0.95'),
               plt.Line2D([], [], color='red', marker='x', ls='', label='censored (not relearned by 4000)')]
    fig.legend(handles=handles, loc='upper center', ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(.5, .995))
    fig.suptitle('R3: A, then B for d updates, then A again; 3 init seeds as individual traces; x = steps since each switch (log1p scale, '
                 'same scale in every segment)', y=.955, fontsize=10)
    fig.text(.5, .005, 'Optimizer reset at every switch. With separate heads and wd = 0 the A head is unchanged during B, so the blue drop '
             'in phase B is trunk drift. In the conflict row (one shared head, no task identifier) notA accuracy = 1 - the blue curve. '
             'Dotted: chance 0.5; dashed grey: criterion 0.95.', ha='center', fontsize=8, wrap=True)
    for ext in ('png', 'svg'):
        fig.savefig(out/f'recovery_curves.{ext}', dpi=130, bbox_inches='tight')
    plt.close(fig)
    with (out/'recovery_curves.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)


def summary_rows(runs, status):
    rows = []
    for s in SEEDS:
        A = runs.get(f'r1-hist-A-s{s}-a1')
        fresh = first(A['rows'], 1, 'A_test_acc') if A else None
        ref = dict(init=snap_at(A['snaps'], 0)['readout']['2000']['test_acc'] if A else None,
                   ckA=phase_end(A['snaps'], 1)['readout']['2000']['test_acc'] if A else None)
        for arm, btask, _ in ROWS[:4] + [('d0', None, '')]:
            for d in ((0,) if arm == 'd0' else DS):
                rid = f'r1-hist-d0-s{s}-a1' if arm == 'd0' else f'r1-hist-{arm}-d{d}-s{s}-a1'
                run = runs.get(rid)
                row = dict(run_id=rid, arm=arm, d=d, init_seed=s, status=status.get(rid, 'missing'),
                           fresh_A_updates_to_criterion=fresh, readout_init=ref['init'], readout_ckA=ref['ckA'])
                if run is None:
                    rows.append(row); continue
                ph_b = 1 if arm != 'd0' else None
                endB = phase_end(run['snaps'], 1) if ph_b else snap_at(run['snaps'], 0)
                endR = phase_end(run['snaps'], 2 if ph_b else 1)
                relearn = first(run['rows'], 2 if ph_b else 1, 'A_test_acc')
                scratch = runs.get(f'r1-hist-scratch-{btask}-s{s}-a1') if btask else None
                row.update(
                    endB_A_test=endB['metrics']['A_test_acc'], endB_A_train=endB['metrics']['A_train_acc'],
                    endB_readout_test=endB['readout']['2000']['test_acc'], endB_readout_train=endB['readout']['2000']['train_acc'],
                    endB_readout200_test=endB['readout']['200']['test_acc'],
                    endB_null_test=endB['readout_null']['2000']['test_acc'], endB_null_train=endB['readout_null']['2000']['train_acc'],
                    readout_scratchB_same_d=snap_at(scratch['snaps'], d)['readout']['2000']['test_acc'] if scratch else None,
                    endB_A_pred_changed=endB['change_vs_start']['A']['pred_changed'],
                    endB_A_margin_rms_change=endB['change_vs_start']['A']['margin_rms_change'],
                    endB_A_margin_rms_ckA=endB['change_vs_start']['A']['margin_rms_before'],
                    endB_hidden_cka=endB['change_vs_start']['hidden_linear_cka'],
                    endB_trunk_rel_drift=endB['change_vs_start']['trunk_rel_drift'],
                    B_updates_to_criterion=(first(run['rows'], 1, f'{btask}_test_acc') if btask else
                                            first(run['rows'], 1, 'A_test_acc', below=True) if arm == 'conflict' else None),
                    scratch_B_updates_to_criterion=first(scratch['rows'], 1, f'{btask}_test_acc') if scratch else None,
                    endB_B_test=endB['metrics'][f'{btask}_test_acc'] if btask else None,
                    relearn_updates=relearn, relearn_censored=relearn is None,
                    relearn_examples=None if relearn is None else 256*relearn,
                    savings_ratio=None if relearn is None or not fresh else relearn/fresh,
                    endR_A_test=endR['metrics']['A_test_acc'],
                    endR_B_test=endR['metrics'][f'{btask}_test_acc'] if btask else None,
                    endR_A_pred_changed_vs_ckA=endR['change_vs_start']['A']['pred_changed'],
                    return_B_pred_changed=endR['change_vs_prev_phase_end']['B']['pred_changed'] if btask else None)
                rows.append(row)
    return rows


def summary_figure(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), gridspec_kw=dict(wspace=.22, hspace=.62))
    axes = axes.ravel()
    arms = [('rel', 'related B'), ('unrel', 'unrelated B'), ('contA', 'continued-A'), ('conflict', 'conflict*')]
    xs = {(a, d): i*4.5+j*1.2 for i, (a, _) in enumerate(arms) for j, d in enumerate(DS)}
    jit = {s: (k-1)*.18 for k, s in enumerate(SEEDS)}

    def put(ax, key, color, marker, label, filled=True):
        for r in rows:
            if (r['arm'], r['d']) in xs and r.get(key) is not None:
                ax.plot(xs[r['arm'], r['d']]+jit[r['init_seed']], r[key], marker, color=color, ms=5,
                        mfc=color if filled else 'none', label=label)
                label = None

    def frame(ax, title, ylabel):
        ax.set_xticks([xs[k] for k in xs], [f'{d}' for _, d in xs], fontsize=7)
        for i, (_, name) in enumerate(arms):
            ax.text(i*4.5+1.2, 1.01, name, ha='center', va='bottom', fontsize=8, transform=ax.get_xaxis_transform())
            if i:
                ax.axvline(i*4.5-.85, color=GREY, lw=.5)
        ax.set_title(title, fontsize=9, loc='left', pad=16); ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel('B duration d (updates)', fontsize=8)
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)

    ax = axes[0]
    put(ax, 'endB_A_test', BLUE, 'o', 'original A head, held-out')
    put(ax, 'endB_readout_test', '#1baf7a', 's', 'new readout on frozen end-of-B trunk, held-out')
    put(ax, 'readout_scratchB_same_d', ORANGE, 'D', 'same readout, trunk trained on B only (no A history)', filled=False)
    put(ax, 'readout_ckA', GREY, '_', 'same readout, end-of-A trunk')
    put(ax, 'readout_init', GREY, '.', 'same readout, untrained trunk')
    put(ax, 'endB_null_test', 'k', 'x', 'shuffled-label null, held-out')
    ax.axhline(.5, color=GREY, lw=.6, ls=':'); ax.set_ylim(-.03, 1.03)
    frame(ax, 'a. A at the end of phase B', 'held-out A accuracy')

    ax = axes[1]
    put(ax, 'relearn_updates', BLUE, 'o', 'updates to regain held-out A >= 0.95')
    put(ax, 'fresh_A_updates_to_criterion', GREY, '_', 'fresh-A from the same init')
    for r in rows:
        if (r['arm'], r['d']) in xs and r.get('relearn_censored'):
            ax.plot(xs[r['arm'], r['d']]+jit[r['init_seed']], T_R, 'x', color='red', ms=7)
    ax.set_yscale('symlog', linthresh=10); ax.set_ylim(-.5, 6000)
    frame(ax, 'b. relearning cost (0 = never below criterion)', 'updates (x256 = examples), symlog')

    ax = axes[2]
    put(ax, 'endB_A_pred_changed', BLUE, 'o', 'A-head predictions changed vs end of A (all 1024 inputs)')
    put(ax, 'endB_trunk_rel_drift', 'k', '^', 'relative trunk drift', filled=False)
    put(ax, 'endB_hidden_cka', '#6250d6', 's', 'hidden linear CKA vs end of A', filled=False)
    ax.set_ylim(-.03, 1.03)
    frame(ax, 'c. functional and weight change during B', 'fraction / ratio')

    ax = axes[3]
    put(ax, 'endB_B_test', ORANGE, 'o', 'B held-out at end of B')
    put(ax, 'endR_B_test', ORANGE, 's', 'B held-out after the 4000-update return', filled=False)
    ax.axhline(.5, color=GREY, lw=.6, ls=':'); ax.set_ylim(-.03, 1.03)
    for i in (2, 3):
        ax.text(i*4.5+1.2, .5, 'no B task', ha='center', fontsize=8, color=GREY)
    frame(ax, 'd. B before and after returning to A', 'held-out B accuracy')
    fig.text(.5, .0, '3 init seeds per condition, offset horizontally (left to right: 53200, 53201, 53202). Readouts: fresh linear softmax on '
             'train-standardized hidden units, 2000 Adam updates at lr 1e-2 on the 256 training inputs, evaluated on the 768 held-out inputs. '
             '*conflict = separate experiment (notA through the shared head); there notA requires the same features as A, so the readout '
             'cannot separate retention from relearning.', ha='center', fontsize=7.5, wrap=True)
    for ext in ('png', 'svg'):
        fig.savefig(out/f'survival_summary.{ext}', dpi=130, bbox_inches='tight')
    plt.close(fig)


def followup_figure(runs, status, out):
    """M3 follow-up, kept separate from the initial comparison: return-phase updates to criterion
    with head A reset to its init draw, on different trunks."""
    kinds = [('fresh', 'untrained trunk = fresh-A', GREY, '_'),
             ('scratch', 'B-only trunk (same d), head A at init', ORANGE, 'D'),
             ('reset', 'end-of-B trunk after A, head A reset to init', '#1baf7a', 's'),
             ('kept', 'end-of-B trunk after A, trained head A kept (M2)', BLUE, 'o'),
             ('ckA', 'end-of-A trunk, head A reset to init', 'k', '^')]
    groups = [(arm, b, d) for arm, b in (('rel', 'Brel'), ('unrel', 'Bunrel')) for d in (1000, 10000)]
    rows = []
    for s in SEEDS:
        fresh = first(runs[f'r1-hist-A-s{s}-a1']['rows'], 1, 'A_test_acc')
        for arm, b, d in groups:
            ids = dict(fresh=f'r1-hist-A-s{s}-a1', scratch=f'r1-hist-fu-reset-scratch-{b}-d{d}-s{s}-a1',
                       reset=f'r1-hist-fu-reset-{arm}-d{d}-s{s}-a1', kept=f'r1-hist-{arm}-d{d}-s{s}-a1',
                       ckA=f'r1-hist-fu-reset-ckA-s{s}-a1')
            for kind, rid in ids.items():
                run = runs.get(rid)
                phase = 2 if kind == 'kept' else 1
                t = first(run['rows'], phase, 'A_test_acc') if run else None
                rows.append(dict(group=f'{arm}-d{d}', init_seed=s, trunk_and_head=kind, run_id=rid,
                                 status=status.get(rid, 'missing'), updates_to_criterion=t,
                                 censored=run is not None and t is None, ratio_to_fresh_A=None if t is None else t/fresh))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    jit = {s: (k-1)*.07 for k, s in enumerate(SEEDS)}
    for gi, (arm, b, d) in enumerate(groups):
        for ki, (kind, label, color, marker) in enumerate(kinds):
            for r in rows:
                if r['group'] == f'{arm}-d{d}' and r['trunk_and_head'] == kind:
                    x = gi + (ki-2)*.17 + jit[r['init_seed']]
                    y = T_R if r['censored'] else r['updates_to_criterion']
                    ax.plot(x, y, 'x' if r['censored'] else marker, color='red' if r['censored'] else color,
                            mfc='none' if marker in 'Ds^' else color, ms=6,
                            label=label if (gi, r['init_seed']) == (0, SEEDS[0]) else None)
    ax.set_xticks(range(len(groups)), [f'{"related" if a == "rel" else "unrelated"} B, d = {d}' for a, _, d in groups], fontsize=8)
    ax.set_yscale('log'); ax.set_ylim(20, 2000)
    ax.set_ylabel('return-phase updates to held-out A >= 0.95 (log; x256 = examples)', fontsize=8)
    ax.set_title('Follow-up (adaptive, M3): relearning A with head A reset to its initial draw, by trunk; '
                 '3 init seeds offset left to right', fontsize=9, loc='left')
    ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.09), ncol=2, frameon=False)
    for ext in ('png', 'svg'):
        fig.savefig(out/f'followup_head_reset.{ext}', dpi=130, bbox_inches='tight')
    plt.close(fig)
    with (out/'followup_head_reset.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, default=Path('runs/regimes'))
    ap.add_argument('--out', type=Path, default=HERE/'figures')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    status = statuses(HERE/'registry.jsonl')
    ids = [r['run_id'] for r in json.loads((HERE/'plan.json').read_text())['runs'] if r['run_id'] in status]
    runs = {rid: read_run(args.runs, rid) for rid in ids}
    runs = {k: v for k, v in runs.items() if v is not None}
    recovery_figure(args.runs, runs, status, args.out)
    rows = summary_rows(runs, status)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (args.out/'survival_summary.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary_figure(rows, args.out)
    if any(k.startswith('r1-hist-fu-') for k in runs):
        followup_figure(runs, status, args.out)


if __name__ == '__main__':
    main()
