#!/usr/bin/env python3
"""Decay switched off mid-decline (endpoint intervention, QUESTION.md Q2).

Each intervention run resumed a published e3 wd 1 checkpoint of e3-me-filt-w128-i<seed>-a1 (update s = 300 or
1,000; weights, Adam moments and update count carried over) and continued at wd 0 to 20,000 updates. Asserted
per run: the resume file is the source's published checkpoint (SHA-256 against config and source manifest), the
source is a wd 1 filtered-start run, the run's wd is 0 and its other settings and data equal the source's, the
resumed update-s metrics row equals the source's except wall-clock fields and the optimizer-update diagnostic
(which a resumed row does not log), and its update-s checkpoint arrays equal the source's.

Comparators are the e3 wd 1 run (same state at s, decay kept on) and the e3 wd 0 run from the same start.

Readings (declared in Q2 before any run), on own held-out accuracy A (cal and audit, 662 pairs) over the window
W = 1,000 updates after the switch:
  d_off = A_off(s+W) - A_1(s),  d_1 = A_1(s+W) - A_1(s),  d_0 = A_0(s+W) - A_0(s),
  f = (d_off - d_0) / (d_1 - d_0);  f <= 1/3: (i) decay must stay on;  f >= 2/3: (ii) self-sustaining;
  otherwise partial;  recovery (iii) flagged if A_off reaches A_1(s) + 0.05 at any logged update after s.
Secondary: A and common accuracy at 20,000, first update after s below 0.10 (censored), train CE, norm, and the
shared e3 measures at checkpoints (decay.py definitions).

uv run --locked --group analysis python research/e4/dynamics/offswitch.py RUN_DIR ... --wd1 DIR ... --wd0 DIR ... --out DIR
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
import decay  # noqa: E402
from decay import COMMON, P, ladder_sets, save, tr30, tr50, write_csv  # noqa: E402

ROOT = decay.ROOT
W = 1000
CUT_I, CUT_II, RECOVER = 1/3, 2/3, .05
WALL = ('seconds', 'elapsed_after_diagnostics', 'optimizer_update')
COLORS = {'wd 1': decay.WD_COLORS[1.0], 'wd 0': decay.WD_COLORS[0.0], 300: '#e7298a', 1000: '#66a61e'}


def logged(run_dir, step):
    for line in (run_dir/'metrics.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['step'] == step:
            return {k: v for k, v in r.items() if k not in WALL}
    raise KeyError(f'{run_dir.name}: no metrics row at {step}')


def load_switch(run_dir, by_id):
    cfg = json.loads((run_dir/'config.json').read_text())
    resume = ROOT/cfg['resume']
    src = by_id[resume.parent.name]
    s = int(resume.stem.split('_')[1])
    assert src['wd'] == 1.0 and float(cfg['wd']) == 0.0, f'{run_dir.name}: not a wd 1 -> 0 switch'
    assert cfg['resume_sha256'] == decay.sha256(resume) == src['manifest'][resume.name]['sha256'], run_dir.name
    for k in ('lr', 'hidden', 'p', 'steps', 'log_every', 'task'):
        assert cfg[k] == json.loads((src['dir']/'config.json').read_text())[k], f'{run_dir.name}: {k} differs from source'
    with np.load(run_dir/'data.npz') as z:
        d = {k: z[k] for k in ('a', 'b', 'yadd', 'train', 'cal', 'audit')}
    assert all(np.array_equal(d[k], src['d'][k]) for k in d), f'{run_dir.name}: data differ from source'
    assert logged(run_dir, s) == logged(src['dir'], s), f'{run_dir.name}: resumed row differs from source'
    with np.load(run_dir/f'checkpoint_{s:06d}.npz') as x, np.load(resume) as y:
        assert all(np.array_equal(x[k], y[k]) for k in y.files), f'{run_dir.name}: update-{s} state differs from source'
    manifest = json.loads((run_dir/'manifest.json').read_text())['files']
    return dict(dir=run_dir, run_id=cfg['run_id'], wd=0.0, lr=cfg['lr'], steps=cfg['steps'], start=src['start'],
                init_seed=src['init_seed'], d=d, w0=None, manifest=manifest, commit=cfg['runtime']['git_commit'],
                source=src['run_id'], switch=s, arm=s)


def rows_for(run, sets):
    dr, cr = decay.dense(run), decay.checkpoint_rows(run, sets)
    s = run.get('switch')
    for r in dr + cr:  # exposure accrues at wd 1 up to the switch, then stays constant
        r.update(arm=run['arm'], exposure=decay.exposure(min(r['step'], s), run['lr'], 1.0) if s else r['exposure'])
    return dr, cr


def readings(run, dense_rows):
    A = {(r['arm'], r['init_seed']): {} for r in dense_rows}
    for r in dense_rows:
        A[(r['arm'], r['init_seed'])][r['step']] = r
    s, seed = run['switch'], run['init_seed']
    off, one, zero = A[(s, seed)], A[('wd 1', seed)], A[('wd 0', seed)]
    a = lambda m, t: m[t]['own_heldout']  # noqa: E731
    d_off, d_1, d_0 = a(off, s + W) - a(one, s), a(one, s + W) - a(one, s), a(zero, s + W) - a(zero, s)
    f = (d_off - d_0)/(d_1 - d_0)
    after = [t for t in sorted(off) if t > s]
    peak_after = max(after, key=lambda t: a(off, t))
    below = [t for t in after if a(off, t) < decay.THETA_FIXED]
    last = after[-1]
    return dict(run_id=run['run_id'], init_seed=seed, switch=s, source=run['source'],
                A1_s=a(one, s), Aoff_sW=a(off, s + W), A1_sW=a(one, s + W), A0_s=a(zero, s), A0_sW=a(zero, s + W),
                d_off=d_off, d_1=d_1, d_0=d_0, f=f,
                reading='(i) decay must stay on' if f <= CUT_I else '(ii) self-sustaining' if f >= CUT_II else 'partial',
                max_after=a(off, peak_after), max_after_step=peak_after,
                recovery=a(off, peak_after) >= a(one, s) + RECOVER,
                T_010_after=below[0] if below else f'>{last}',
                own_end=a(off, last), own_end_wd1=a(one, last), own_end_wd0=a(zero, last),
                train_acc_min_after=min(off[t]['train_acc'] for t in after), train_ce_end=off[last]['train_ce'],
                norm_s=off[s]['norm'], norm_end=off[last]['norm'])


PANELS = [('own_heldout', 'held-out accuracy', 'dense'), ('train_ce', 'train cross-entropy', 'dense'),
          ('P_cS', 'P(a+b over S)', 'ck'), ('margin_N_sigma', 'median margin of a+b over N (σ)', 'ck'),
          ('norm', 'weight norm', 'dense')]


def style(arm):
    return dict(color=COLORS[arm], label=arm if isinstance(arm, str) else f'wd 1, then 0 from update {arm}')


def figure(seeds, dense_rows, ck_rows, out):
    arms = ['wd 0', 'wd 1', 300, 1000]
    fig, axes = plt.subplots(len(PANELS), len(seeds), figsize=(4.2*len(seeds), 2.4*len(PANELS)), sharex=True, sharey='row',
                             layout='constrained', squeeze=False)
    for j, seed in enumerate(seeds):
        for arm in arms:
            dr = [r for r in dense_rows if r['arm'] == arm and r['init_seed'] == seed and r['step'] > 0]
            cr = [r for r in ck_rows if r['arm'] == arm and r['init_seed'] == seed and r['step'] > 0]
            if not dr:
                continue
            for i, (key, _, src) in enumerate(PANELS):
                rs = dr if src == 'dense' else cr
                axes[i, j].plot([r['step'] for r in rs], [r[key] for r in rs], lw=.9,
                                marker=None if src == 'dense' else 'o', ms=3, **style(arm))
            axes[0, j].plot([r['step'] for r in cr], [r['common_acc'] for r in cr], lw=0, marker='x', ms=4, color=COLORS[arm])
        for s in (300, 1000):
            for ax in axes[:, j]:
                ax.axvline(s, color=COLORS[s], lw=.7, ls=':')
        axes[0, j].set_title(f'filtered start, init {seed}', fontsize=9)
        axes[0, j].axhline(decay.THETA_FIXED, color='0.5', lw=.7, ls=(0, (4, 3)))
        axes[0, j].axhline(1/P, color='0.7', lw=.7, ls=':')
        axes[2, j].axhline(.5, color='0.5', lw=.7, ls=(0, (4, 3)))
        axes[3, j].axhline(0, color='0.5', lw=.7, ls=(0, (4, 3)))
    for i, (key, title, _) in enumerate(PANELS):
        axes[i, 0].set_ylabel(title, fontsize=8)
        for ax in axes[i]:
            ax.set_xscale('log'); ax.grid(True, color='0.9', lw=.6)
            if key == 'train_ce':
                ax.set_yscale('log')
    for ax in axes[-1]:
        ax.set_xlabel('update since the edit (log scale)')
    axes[0, 0].legend(fontsize=7, frameon=False, loc='lower left')
    save(fig, out, 'offswitch', 'Endpoint intervention (QUESTION.md Q2): the e3 wd 1 runs from the three filtered start files, '
         'resumed at update 300 or 1,000 (dotted lines) with weights, Adam moments and update count carried over, and '
         'continued at wd 0 to 20,000. Comparators: the same wd 1 runs (decay kept on) and the wd 0 runs from update 0. '
         'Accuracy panel: lines = own held-out accuracy (662 cal+audit pairs, every logged update); x = the 483 common pairs '
         'at published checkpoints; dashed = 0.10, dotted grey = chance 1/31. P(a+b over S) and the σ-margin are on the '
         'common pairs at published checkpoints. Source: offswitch_dense.csv, offswitch_checkpoints.csv.')
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2*len(seeds), 3.0), sharey=True, layout='constrained', squeeze=False)
    for j, seed in enumerate(seeds):
        ax = axes[0, j]
        for arm in arms:
            dr = [r for r in dense_rows if r['arm'] == arm and r['init_seed'] == seed and r['step'] <= 2500]
            ax.plot([r['step'] for r in dr], [r['own_heldout'] for r in dr], lw=.9, **style(arm))
        for s in (300, 1000):
            ax.axvspan(s, s + W, color=COLORS[s], alpha=.08, lw=0)
        ax.set_title(f'filtered start, init {seed}', fontsize=9); ax.grid(True, color='0.9', lw=.6)
        ax.set_xlabel('update since the edit'); ax.axhline(decay.THETA_FIXED, color='0.5', lw=.7, ls=(0, (4, 3)))
    axes[0, 0].set_ylabel('own held-out accuracy', fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc='lower left')
    save(fig, out, 'offswitch_window', 'The reading windows (shaded: the 1,000 updates after each switch, Q2) on a linear '
         'update axis to 2,500. Own held-out accuracy at every logged update. Source: offswitch_dense.csv.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', type=Path, nargs='+', help='intervention run directories')
    ap.add_argument('--wd1', type=Path, nargs='+', required=True, help='e3 wd 1 filtered-start runs (the resume sources)')
    ap.add_argument('--wd0', type=Path, nargs='+', required=True, help='e3 wd 0 filtered-start runs')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    refs = [dict(decay.load(d), arm='wd 1') for d in args.wd1] + [dict(decay.load(d), arm='wd 0') for d in args.wd0]
    assert all(r['wd'] == (1.0 if r['arm'] == 'wd 1' else 0.0) for r in refs)
    by_id = {r['run_id']: r for r in refs}
    runs = [load_switch(d, by_id) for d in args.runs]
    sets = ladder_sets(refs[0]['d']['a'], refs[0]['d']['b'], tr30, tr50, COMMON, P)
    dense_rows, ck_rows = [], []
    for run in refs + runs:
        dr, cr = rows_for(run, sets)
        dense_rows += dr; ck_rows += cr
    table = [readings(run, dense_rows) for run in sorted(runs, key=lambda r: (r['switch'], r['init_seed']))]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'offswitch_dense.csv', dense_rows)
    write_csv(args.out/'offswitch_checkpoints.csv', ck_rows)
    write_csv(args.out/'offswitch_readings.csv', table)
    figure(sorted({r['init_seed'] for r in runs}), dense_rows, ck_rows, args.out)
    for t in table:
        print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in t.items()))


if __name__ == '__main__':
    main()
