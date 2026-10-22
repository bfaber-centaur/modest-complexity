#!/usr/bin/env python3
"""Filtered start files continued at several weight-decay strengths: early rise, decline and decay exposure.

Inputs are continuation runs from the e3 mechanisms filter start files (arm B): every run must start
from one of starts/e3-me-filt-w128-i3000{0,1,2}-a1.npz (hash checked against the start file), with
fresh Adam moments (--init-weights), lr 1e-3, data seed 31000 and the frozen e1 30% training set.

Per run (definitions declared in QUESTION.md Q1, before measuring):
  dense, every logged update (metrics.jsonl): own held-out accuracy (cal and audit pooled, 662 pairs),
    train accuracy and cross-entropy, weight norm, decay exposure E = -t log(1 - lr wd);
  published checkpoints: accuracy and cross-entropy on the 483 common pairs, P(a+b over N),
    P(a+b over S), P(S over N) and the median sigma-margin of a+b over N (shared e3 definitions, D6;
    S, D, N from the e1 30% and e1f2 50% training sets, research/e3/audit/pairwise.py);
  summary: peak (max own held-out accuracy over updates 0-1,000), first update after the peak below
    theta for theta = the run's update-0 own held-out accuracy and theta = 0.10 (censored at the last
    update if never), re-crossings above theta, and endpoints;
  decision (QUESTION.md Q1 addendum, fixed before the new runs): for each run at 0 < wd < 1 and each
    threshold, r = observed crossing update / the update at which it would reach the exposure at which
    the wd 1 run from the same start crossed; within a factor 1.5 = "exposure clock", above = "slower"
    (censored runs: slower, with r a lower bound), below = "faster";
  matched (descriptive, chosen after viewing the first figure, not part of the decision rule): own
    held-out accuracy, train cross-entropy and norm of each decayed run at the logged update nearest
    E = 0.1, 0.3, 0.6, 1, 1.5, 2, beside the wd 0 run from the same start at that same update.

uv run --locked --group analysis python research/e4/dynamics/decay.py RUN_DIR ... --out DIR
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for d in ('research/e3/audit', 'research/e2/audit', 'research/size-time'):
    sys.path.insert(0, str(ROOT/d))
from pairwise import ladder_sets, preference  # noqa: E402
from trajectories import save  # noqa: E402
from margins_check import COMMON, tr30, tr50  # noqa: E402  (frozen e1 30% and e1f2 50% training sets)

P = 31
STARTS = ROOT/'research/e3/mechanisms/starts'
THETA_FIXED = .10
MATCH_E = (.1, .3, .6, 1., 1.5, 2.)
WD_COLORS = {0.0: '#555555', 0.1: '#1b9e77', 0.3: '#d95f02', 1.0: '#7570b3'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(run_dir):
    cfg = json.loads((run_dir/'config.json').read_text())
    start = ROOT/cfg['init_weights']
    assert start.parent == STARTS and start.name.startswith('e3-me-filt-'), f'{run_dir.name}: not a filtered start'
    assert sha256(start) == cfg['init_weights_sha256'], f'{run_dir.name}: start file hash'
    assert cfg['lr'] == 1e-3 and cfg['seeds']['data_seed'] == 31000 and cfg['hidden'] == 128
    with np.load(run_dir/'data.npz') as z:
        d = {k: z[k] for k in ('a', 'b', 'yadd', 'train', 'cal', 'audit')}
    assert np.array_equal(d['train'], tr30) and not set(COMMON.tolist()) & set(tr30.tolist())
    with np.load(start) as z:
        w0 = [z[k] for k in 'UVW']
    manifest = json.loads((run_dir/'manifest.json').read_text())['files']
    return dict(dir=run_dir, run_id=cfg['run_id'], wd=float(cfg['wd']), lr=cfg['lr'], steps=cfg['steps'],
                start=start.name, init_seed=cfg['seeds']['init_seed'], d=d, w0=w0, manifest=manifest,
                commit=cfg['runtime']['git_commit'])


def exposure(step, lr, wd):
    return -step*np.log1p(-lr*wd)


def dense(run):
    n_cal, n_audit = len(run['d']['cal']), len(run['d']['audit'])
    rows = []
    for line in (run['dir']/'metrics.jsonl').read_text().splitlines():
        r = json.loads(line)
        rows.append(dict(run_id=run['run_id'], wd=run['wd'], init_seed=run['init_seed'], step=r['step'],
                         exposure=exposure(r['step'], run['lr'], run['wd']),
                         own_heldout=(r['cal']*n_cal + r['audit']*n_audit)/(n_cal + n_audit),
                         train_acc=r['train'], train_ce=r['objective']['train']['cross_entropy'], norm=r['norm']))
    return rows


def checkpoint_rows(run, sets):
    S, D, N, y = sets
    a, b = run['d']['a'][COMMON], run['d']['b'][COMMON]
    truth = np.eye(P, dtype=bool)[y]; ix = np.arange(len(y))
    rows = []
    for path in sorted(run['dir'].glob('checkpoint_*.npz'), key=lambda q: int(q.stem.split('_')[1])):
        assert sha256(path) == run['manifest'][path.name]['sha256'], path
        with np.load(path) as ck:
            step = int(ck['step']); w = [ck[k] for k in 'UVW']
        if step == 0:
            assert all(np.array_equal(x, y0) for x, y0 in zip(w, run['w0'])), f"{run['run_id']}: update 0 differs from the start file"
        z = np.maximum(w[0][a] + w[1][b], 0) @ w[2]
        sigma = np.where(truth, np.nan, z); sigma = np.nanstd(sigma, 1)
        x = z - z.max(1, keepdims=True)
        rows.append(dict(run_id=run['run_id'], wd=run['wd'], init_seed=run['init_seed'], step=step,
                         exposure=exposure(step, run['lr'], run['wd']),
                         common_acc=float(np.mean(z.argmax(1) == y)),
                         common_ce=float(np.mean(np.log(np.exp(x).sum(1)) - x[ix, y])),
                         **{k: float(np.nanmean(preference(z, A, B)[0]))
                            for k, A, B in (('P_cN', truth, N), ('P_cS', truth, S), ('P_SN', S, N))},
                         margin_N_sigma=float(np.median((z[ix, y] - np.where(N, z, -np.inf).max(1))/sigma)),
                         norm=float(np.sqrt(sum(np.sum(v**2) for v in w)))))
    return rows


def summary(run, rows, cks):
    acc = {r['step']: r['own_heldout'] for r in rows}
    steps = sorted(acc)
    early = [s for s in steps if s <= 1000]
    peak = max(early, key=lambda s: acc[s])
    out = dict(run_id=run['run_id'], wd=run['wd'], init_seed=run['init_seed'], start=run['start'], commit=run['commit'][:7],
               last_step=steps[-1], own_update0=acc[0], peak=acc[peak], peak_step=peak)
    for name, theta in (('update0', acc[0]), ('fixed010', THETA_FIXED)):
        after = [s for s in steps if s > peak]
        below = [s for s in after if acc[s] < theta]
        t = below[0] if below else None
        out[f'T_{name}'] = t if t is not None else f'>{steps[-1]}'
        out[f'E_at_T_{name}'] = exposure(t, run['lr'], run['wd']) if t is not None else ''
        out[f'recross_{name}'] = sum(1 for s0, s1 in zip(after, after[1:]) if t is not None and s0 >= t and acc[s0] < theta <= acc[s1])
    last = rows[-1]; ck_last = max(cks, key=lambda r: r['step'])
    out.update(own_end=last['own_heldout'], common_end=ck_last['common_acc'], common_end_step=ck_last['step'],
               train_acc_end=last['train_acc'], train_ce_end=last['train_ce'], norm_0=rows[0]['norm'], norm_end=last['norm'],
               exposure_end=last['exposure'], P_cS_end=ck_last['P_cS'], P_SN_end=ck_last['P_SN'],
               margin_end=ck_last['margin_N_sigma'])
    return out


def decision(summ, lr=1e-3, band=1.5):
    """QUESTION.md Q1 addendum rule: r = observed T / T predicted from the wd 1 run of the same start at equal exposure."""
    ref = {s['init_seed']: s for s in summ if s['wd'] == 1.0}
    out = []
    for s in summ:
        if s['wd'] in (0.0, 1.0) or s['init_seed'] not in ref:
            continue
        per = -np.log1p(-lr*s['wd'])
        for name in ('update0', 'fixed010'):
            e_ref = ref[s['init_seed']][f'E_at_T_{name}']
            pred = e_ref/per
            t = s[f'T_{name}']
            censored = not isinstance(t, int)
            r = (s['last_step'] if censored else t)/pred
            verdict = ('slower (censored)' if censored else 'slower' if r > band else 'faster' if r < 1/band else 'exposure clock')
            out.append(dict(run_id=s['run_id'], wd=s['wd'], init_seed=s['init_seed'], threshold=name, E_ref=e_ref,
                            T_predicted=pred, T_observed=t, r=(f'>{r:.3f}' if censored else r), verdict=verdict))
    return out


def matched(runs, dense_rows):
    """Each decayed run at the logged update nearest each exposure in MATCH_E, beside its wd 0 partner at that update."""
    by = {}
    for r in dense_rows:
        by.setdefault((r['wd'], r['init_seed']), {})[r['step']] = r
    out = []
    for run in sorted(runs, key=lambda r: (r['init_seed'], r['wd'])):
        if run['wd'] == 0:
            continue
        mine, zero = by[(run['wd'], run['init_seed'])], by.get((0.0, run['init_seed']), {})
        for e in MATCH_E:
            step = min(mine, key=lambda t: abs(mine[t]['exposure'] - e))
            if abs(mine[step]['exposure'] - e) > .01*e:
                continue
            r, z = mine[step], zero.get(step)
            out.append(dict(run_id=run['run_id'], wd=run['wd'], init_seed=run['init_seed'], E=e, step=step,
                            exposure=r['exposure'], own_heldout=r['own_heldout'], train_ce=r['train_ce'], norm=r['norm'],
                            wd0_own_heldout=z['own_heldout'] if z else '', wd0_norm=z['norm'] if z else ''))
    return out


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


PANELS = [('own_heldout', 'held-out accuracy', 'dense'), ('train_ce', 'train cross-entropy', 'dense'),
          ('P_cS', 'P(a+b over S)', 'ck'), ('margin_N_sigma', 'median margin of a+b over N (σ)', 'ck'),
          ('norm', 'weight norm', 'dense')]


def figure(runs, dense_rows, ck_rows, summ, out, axis):
    seeds = sorted({r['init_seed'] for r in runs})
    fig, axes = plt.subplots(len(PANELS), len(seeds), figsize=(4.2*len(seeds), 2.4*len(PANELS)), sharex=True, sharey='row',
                             layout='constrained', squeeze=False)
    xkey = 'step' if axis == 'update' else 'exposure'
    for j, seed in enumerate(seeds):
        for run in sorted((r for r in runs if r['init_seed'] == seed), key=lambda r: r['wd']):
            if axis == 'exposure' and run['wd'] == 0:
                continue
            color = WD_COLORS.get(run['wd'], 'k'); label = f"wd {run['wd']:g}"
            dr = [r for r in dense_rows if r['run_id'] == run['run_id'] and r[xkey] > 0]
            cr = [r for r in ck_rows if r['run_id'] == run['run_id'] and r[xkey] > 0]
            for i, (key, _, src) in enumerate(PANELS):
                ax = axes[i, j]
                if src == 'dense':
                    ax.plot([r[xkey] for r in dr], [r[key] for r in dr], color=color, lw=.9, label=label)
                else:
                    ax.plot([r[xkey] for r in cr], [r[key] for r in cr], color=color, lw=.9, marker='o', ms=3, label=label)
                if key == 'own_heldout':
                    ax.plot([r[xkey] for r in cr], [r['common_acc'] for r in cr], color=color, lw=0, marker='x', ms=4)
                    s = next(x for x in summ if x['run_id'] == run['run_id'])
                    for name, mk in (('update0', 'v'), ('fixed010', 's')):
                        t = s[f'T_{name}']
                        if isinstance(t, int):
                            r = next(r for r in dr if r['step'] == t)
                            ax.plot([r[xkey]], [r['own_heldout']], color=color, marker=mk, ms=6, mfc='white', zorder=5)
        axes[0, j].set_title(f'filtered start, init {seed}', fontsize=9)
        axes[0, j].axhline(THETA_FIXED, color='0.5', lw=.7, ls=(0, (4, 3)))
        axes[0, j].axhline(1/P, color='0.7', lw=.7, ls=':')
        axes[3, j].axhline(0, color='0.5', lw=.7, ls=(0, (4, 3)))
        axes[2, j].axhline(.5, color='0.5', lw=.7, ls=(0, (4, 3)))
    for i, (key, title, _) in enumerate(PANELS):
        axes[i, 0].set_ylabel(title, fontsize=8)
        for ax in axes[i]:
            ax.set_xscale('log'); ax.grid(True, color='0.9', lw=.6)
            if key == 'train_ce':
                ax.set_yscale('log')
    for ax in axes[-1]:
        ax.set_xlabel('update (log scale)' if axis == 'update' else 'decay exposure  −t·log(1 − lr·wd)  (log scale)')
    axes[0, 0].legend(fontsize=7, frameon=False, loc='lower left')
    note = ('Continuations of the three e3 filtered start files (arm B) with fresh Adam moments, lr 1e-3, data seed 31000, '
            '299 training pairs. Accuracy panel: lines = own held-out accuracy (662 cal+audit pairs, every logged update); '
            'x = accuracy on the 483 common pairs at published checkpoints; open triangle = first update after the peak '
            'below the run\'s update-0 accuracy, open square = first update below 0.10 (dashed); dotted = chance 1/31. '
            'Other panels: train cross-entropy and weight norm every logged update; P(a+b over S) and the σ-margin on the '
            'common pairs at published checkpoints. Update 0 is not shown on the log axis. ')
    if axis == 'exposure':
        note += ('Exposure aligns only the explicit multiplicative shrinkage (1 − lr·wd) per update; it does not equate the '
                 'rest of the AdamW dynamics. wd 0 has zero exposure and is not shown. ')
    save(fig, out, f'decay_{axis}', note + 'Source: decay_dense.csv, decay_checkpoints.csv, decay_summary.csv.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    runs = [load(d) for d in args.runs]
    sets = ladder_sets(runs[0]['d']['a'], runs[0]['d']['b'], tr30, tr50, COMMON, P)
    assert np.array_equal(sets[3], runs[0]['d']['yadd'][COMMON])
    dense_rows, ck_rows, summ = [], [], []
    for run in sorted(runs, key=lambda r: (r['init_seed'], r['wd'])):
        dr, cr = dense(run), checkpoint_rows(run, sets)
        dense_rows += dr; ck_rows += cr; summ.append(summary(run, dr, cr))
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'decay_dense.csv', dense_rows)
    write_csv(args.out/'decay_checkpoints.csv', ck_rows)
    write_csv(args.out/'decay_summary.csv', summ)
    rule = decision(summ)
    if rule:
        write_csv(args.out/'decay_decision.csv', rule)
    match = matched(runs, dense_rows)
    if match:
        write_csv(args.out/'decay_matched.csv', match)
    figure(runs, dense_rows, ck_rows, summ, args.out, 'update')
    if any(r['wd'] > 0 for r in runs):
        figure(runs, dense_rows, ck_rows, summ, args.out, 'exposure')
    for s in summ + rule:
        print(' '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in s.items()))


if __name__ == '__main__':
    main()
