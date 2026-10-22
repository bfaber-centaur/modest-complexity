#!/usr/bin/env python3
"""Nested training-size ladders, one per data seed, each scored on its own never-trained pairs.

Runs are grouped by data seed. Within a ladder every run's training set must nest between the
smallest and the largest (asserted, with transpose closure and identical pair/label arrays), and
all runs are evaluated on the pairs outside the ladder's largest training set ("eval pairs").
Ladders have different eval pairs and are shown separately.

Label sets per ladder, as in e2 dynamics (research/e2/dynamics/margins.py): for an eval pair
(a, b), S = labels of training pairs (a, b') or (a', b) at the smallest size, D = such labels added
only by the largest size, N = the remaining wrong labels (never seen with a or b). The correct
label is in none of them (asserted). Per pair, sigma = standard deviation of the 30 wrong-label
logits (e2 audit cycle 4, margins_check.py). Measures, medians or means over eval pairs:
  margin_N_sigma   (a+b logit - max over N) / sigma, median;
  gap_SN_sigma     (max over S - max over N) / sigma, median;
  rank_absent      a+b's normalized rank among wrong labels outside the run's own operand-label
                   sets (fraction above it; 0 = top, 0.5 = no preference), mean over pairs that
                   have such labels (research/audit/heldout_rank.py definition);
  pref_SN          fraction of (S label, N label) pairs where the S label has the higher logit,
                   mean over pairs with both sets nonempty (0.5 = no preference; scale-free).
Pairs with an empty set are counted in sizes.json, never silently dropped from accuracy.

uv run --locked --group analysis python research/e3/dynamics/ladder.py RUN_DIR ... --out DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/size-time'))
from operand_labels import operand_label_sets  # noqa: E402
from trajectories import SEED_STYLES, save, write_csv  # noqa: E402

SHARE_COLORS = {.3: '#7a7a7a', .4: '#d39a1f', .45: '#b03a8c', .5: '#2a5fb6'}
FINAL_KEYS = ('acc', 'margin_N_sigma', 'gap_SN_sigma', 'rank_absent', 'pref_SN')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(run_dir):
    cfg = json.loads((run_dir/'config.json').read_text())
    with np.load(run_dir/'data.npz') as d:
        data = {k: d[k] for k in ('a', 'b', 'y', 'yadd', 'train', 'cal', 'audit')}
    assert np.array_equal(data['y'], data['yadd']), f'{run_dir.name}: not an addition run'
    return dict(dir=run_dir, run_id=cfg['run_id'], data_seed=cfg['seeds']['data_seed'], init_seed=cfg['seeds']['init_seed'],
                share=cfg.get('train_fraction', .3), width=cfg['hidden'], data=data, train=set(data['train'].tolist()))


def transpose_closed(ix, a, b, p):
    return set((a[ix]*p + b[ix]).tolist()) == set((b[ix]*p + a[ix]).tolist())


def ladder_sets(runs):
    """Eval pairs and S, D, N masks for one data seed; asserts nesting, disjointness, closure."""
    d0 = runs[0]['data']; a, b, y = d0['a'], d0['b'], d0['y']; p = int(y.max()) + 1
    for r in runs:
        assert all(np.array_equal(r['data'][k], d0[k]) for k in ('a', 'b', 'y')), r['run_id']
        assert transpose_closed(r['data']['train'], a, b, p), r['run_id']
    sizes = sorted({len(r['train']) for r in runs})
    by_size = {len(r['train']): r['train'] for r in runs}
    for r in runs:  # one training set per size, and the sets nest in size order
        assert r['train'] == by_size[len(r['train'])], f"{r['run_id']}: two training sets of the same size"
    for small, large in zip(sizes, sizes[1:]):
        assert by_size[small] < by_size[large], 'training sets do not nest'
    smallest, largest = by_size[sizes[0]], by_size[sizes[-1]]
    ev = np.array(sorted(set(range(p*p)) - largest))
    assert transpose_closed(ev, a, b, p) and not set(ev.tolist()) & largest
    t_small, t_large = np.array(sorted(smallest)), np.array(sorted(largest))
    S = operand_label_sets(a, b, y, t_small, p)[ev]
    L = operand_label_sets(a, b, y, t_large, p)[ev]
    truth = np.eye(p, dtype=bool)[y[ev]]
    assert not (S & ~L).any() and not (L & truth).any()
    D, N = L & ~S, ~L & ~truth
    assert (S.astype(int) + D + N + truth == 1).all()
    return dict(a=a, b=b, y=y, p=p, ev=ev, S=S, D=D, N=N, sizes=sizes)


def score(run, sets):
    """One row per checkpoint for one run on its ladder's eval pairs."""
    a, b, y, p, ev, S, N = (sets[k] for k in ('a', 'b', 'y', 'p', 'ev', 'S', 'N'))
    manifest = json.loads((run['dir']/'manifest.json').read_text())['files']
    logged = {r['step']: r for r in map(json.loads, (run['dir']/'metrics.jsonl').read_text().splitlines())}
    truth = np.eye(p, dtype=bool)[y[ev]]
    own_out = ~operand_label_sets(a, b, y, run['data']['train'], p)[ev] & ~truth
    rows_ix = np.arange(len(ev))
    both = S.any(1) & N.any(1); has_out = own_out.any(1)
    rows = []
    for path in sorted(run['dir'].glob('checkpoint_*.npz'), key=lambda q: int(q.stem.split('_')[1])):
        assert sha256(path) == manifest[path.name]['sha256'], path
        with np.load(path) as ck:
            step = int(ck['step'])
            z = np.maximum(ck['U'][a] + ck['V'][b], 0) @ ck['W']
        audit = run['data']['audit']
        assert abs(np.mean(z[audit].argmax(1) == y[audit]) - logged[step]['audit']) < 1e-12, (path, 'audit accuracy')
        z = z[ev]; true = z[rows_ix, y[ev]]
        sigma = np.where(truth, np.nan, z)
        sigma = np.nanstd(sigma, 1)
        best = lambda m: np.where(m, z, -np.inf).max(1)
        mN, gSN = true - best(N), best(S) - best(N)
        above = ((z > true[:, None]) & own_out).sum(1)
        # pref_SN: per pair, the share of (s, n) label pairs with z_s > z_n
        greater = (z[:, :, None] > z[:, None, :]) & S[:, :, None] & N[:, None, :]
        pref = greater.sum((1, 2)) / np.maximum(S.sum(1)*N.sum(1), 1)
        lse = np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1)
        rows.append(dict(data_seed=run['data_seed'], share=run['share'], n_train=len(run['train']), init_seed=run['init_seed'],
                         run_id=run['run_id'], step=step, train_ce=logged[step]['objective']['train']['cross_entropy'],
                         acc=float(np.mean(z.argmax(1) == y[ev])), ce=float(np.mean(lse - true)),
                         median_sigma=float(np.median(sigma)),
                         margin_N_sigma=float(np.median((mN/sigma)[N.any(1)])),
                         gap_SN_sigma=float(np.median((gSN/sigma)[both])),
                         rank_absent=float(np.mean(above[has_out]/own_out.sum(1)[has_out])),
                         top_absent=float(np.mean(above == 0)), pref_SN=float(np.mean(pref[both]))))
    return rows


def final_figure(rows, ladders, out, ref_seed):
    """Accuracy and normalized measures at the last step against actual training-pair count."""
    keys = [('acc', 'held-out accuracy (eval pairs)'), ('margin_N_sigma', 'median margin of a+b over N, in sigma'),
            ('rank_absent', 'a+b rank among own absent labels (0 = top)'), ('pref_SN', 'S-over-N preference (0.5 = none)')]
    fig, axes = plt.subplots(len(keys), len(ladders), figsize=(4.2*len(ladders), 3.1*len(keys)), sharex=True, sharey='row',
                             layout='constrained', squeeze=False)
    last = max(r['step'] for r in rows)
    for j, (seed, sets) in enumerate(ladders.items()):
        final = [r for r in rows if r['data_seed'] == seed and r['step'] == last]
        inits = sorted({r['init_seed'] for r in final})
        for i, (key, label) in enumerate(keys):
            ax = axes[i, j]
            for k, init in enumerate(inits):
                pts = sorted((r['n_train'], r[key], r['share']) for r in final if r['init_seed'] == init)
                ax.plot([q[0] for q in pts], [q[1] for q in pts], color='0.55', lw=.8, ls=SEED_STYLES[k % 4], zorder=1,
                        label=f'init {init}')
                ax.scatter([q[0] for q in pts], [q[1] for q in pts], c=[SHARE_COLORS.get(q[2], 'k') for q in pts],
                           marker='os^D'[k % 4], s=30, zorder=2)
            ax.grid(True, color='0.9', lw=.6)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
        title = f"data seed {seed}{' (e2)' if seed == ref_seed else ''}: {len(sets['ev'])} eval pairs"
        axes[0, j].set_title(title, fontsize=9)
        axes[0, j].axhline(1/31, color='0.4', lw=.8, ls=(0, (4, 3)))
        axes[2, j].axhline(.5, color='0.4', lw=.8, ls=(0, (4, 3))); axes[3, j].axhline(.5, color='0.4', lw=.8, ls=(0, (4, 3)))
        axes[1, j].axhline(0, color='0.4', lw=.8)
        axes[-1, j].set_xlabel('actual training pairs')
        axes[0, j].legend(fontsize=7, frameon=False, loc='upper left')
    axes[0, 0].set_ylim(-.02, 1.02)
    fig.suptitle(f'Nested training-size ladders, one per data seed: values at step {last:,}', fontsize=11)
    save(fig, out, 'ladders_final', 'Marker color = nominal setting (grey 30%, amber 40%, magenta 45%, blue 50%); lines join one '
         'init seed across sizes. Each ladder is scored on its own eval pairs (outside its largest training set), so ladders '
         'are not pooled. Dashed: chance 1/31 (accuracy) or no preference 0.5. sigma = std of the pair\'s 30 wrong-label logits. '
         'Source: ladders.csv.')


def trajectory_figure(rows, ladders, out):
    keys = [('acc', 'held-out accuracy'), ('margin_N_sigma', 'margin of a+b over N (sigma)'), ('pref_SN', 'S-over-N preference')]
    fig, axes = plt.subplots(len(ladders), len(keys), figsize=(4.4*len(keys), 3*len(ladders)), sharex=True, sharey='col',
                             layout='constrained', squeeze=False)
    for i, seed in enumerate(ladders):
        runs = dict.fromkeys(r['run_id'] for r in rows if r['data_seed'] == seed)
        inits = sorted({r['init_seed'] for r in rows if r['data_seed'] == seed})
        for run in runs:
            rs = [r for r in rows if r['run_id'] == run and r['step'] > 0]
            style = dict(color=SHARE_COLORS.get(rs[0]['share'], 'k'), lw=1, ls=SEED_STYLES[inits.index(rs[0]['init_seed']) % 4])
            for j, (key, _) in enumerate(keys):
                axes[i, j].plot([r['step'] for r in rs], [r[key] for r in rs],
                                label=f"{rs[0]['share']:.0%} ({rs[0]['n_train']}), init {rs[0]['init_seed']}", **style)
        axes[i, 0].set_ylabel(f'data seed {seed}', fontsize=9)
        axes[i, 0].legend(fontsize=6, frameon=False, loc='upper left', ncol=2)
    for j, (_, title) in enumerate(keys):
        axes[0, j].set_title(title, fontsize=9)
        for ax in axes[:, j]:
            ax.set_xscale('log'); ax.grid(True, color='0.9', lw=.6)
        axes[-1, j].set_xlabel('update step (log scale)')
    save(fig, out, 'ladders_trajectories', 'Published checkpoints, step 0 omitted. Color = nominal setting, line style = init '
         'seed; legend gives actual training pairs. Each row is one ladder on its own eval pairs. Source: ladders.csv.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reference-seed', type=int, default=31000, help='Data seed labelled as the e2 ladder')
    args = ap.parse_args()
    runs = [load(d) for d in args.runs]
    seeds = sorted({r['data_seed'] for r in runs})
    ladders = {s: ladder_sets([r for r in runs if r['data_seed'] == s]) for s in seeds}
    rows = [row for r in runs for row in score(r, ladders[r['data_seed']])]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'ladders.csv', rows)
    last = max(r['step'] for r in rows)
    sizes = {str(s): dict(eval_pairs=len(v['ev']), train_counts=v['sizes'],
                          **{f'size_{k}': dict(mean=float(v[k].sum(1).mean()), min=int(v[k].sum(1).min()),
                                               empty=int((v[k].sum(1) == 0).sum())) for k in 'SDN'})
             for s, v in ladders.items()}
    (args.out/'sizes.json').write_text(json.dumps(sizes, indent=1) + '\n')
    final_figure(rows, ladders, args.out, args.reference_seed)
    trajectory_figure(rows, ladders, args.out)
    for r in sorted((r for r in rows if r['step'] == last), key=lambda r: (r['data_seed'], r['n_train'], r['init_seed'])):
        print(r['data_seed'], r['n_train'], r['init_seed'], ' '.join(f'{k}={r[k]:.3f}' for k in FINAL_KEYS))
    print(json.dumps(sizes))


if __name__ == '__main__':
    main()
