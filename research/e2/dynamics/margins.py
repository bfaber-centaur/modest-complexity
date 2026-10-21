#!/usr/bin/env python3
"""Margin of the correct label over the strongest wrong label in three label sets, on
the 483 pairs that neither the 30% nor the 50% training set contains.

The 30% training set is a subset of the 50% one (research/e2/coordinator/common_set.json),
so the 30 wrong labels of a common pair (a, b) split into sets that are the same for
every run:
  S  labels of 30% training pairs (a, b') or (a', b);
  D  labels of such pairs that only the 50% training set contains;
  N  the rest, never seen with a or b in either training set.
A run's own operand-associated set is S (30% run), S | D (50% run), or S plus part
of D for a share in between, whose training set also nests. For addition the
correct label is in none of them (asserted). "Beats X" means no label in X has a higher
logit (true when X is empty; empty-set counts are in sizes.json); with no preference it
would hold with probability 1/(|X|+1), averaged over pairs ("ref" lines). Own forward
pass on published checkpoints, each checked against its e1 manifest hash; accuracy is
checked against the run's own logged audit accuracy and the coordinator's
common_set_scores.csv.

uv run --locked --group analysis python research/e2/dynamics/margins.py \
    --group "30% train" RUN_DIR ... --group "50% train" RUN_DIR ... --out DIR
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
sys.path.insert(0, str(ROOT/'research/size-time'))
from operand_labels import operand_label_sets  # noqa: E402
from trajectories import SEED_STYLES, WIDTH_COLORS, save, write_csv  # noqa: E402

SUPPLEMENT = ROOT/'research/e2/dynamics/e1-checkpoints'
COMMON = ROOT/'research/e2/data/common-never-trained-30-50.npz'
COMMON_SHA256 = '1535a21ab192ae95d37a5abb5eb783e74f51bd18a0500fbe0218d910914434d3'
SPLITS = {'30%': ROOT/'research/data/e1-split.npz', '50%': ROOT/'research/data/e1f2-split-tf50.npz'}
COORDINATOR_SCORES = ROOT/'research/e2/coordinator/common_set_scores.csv'
COLORS = ['#7a7a7a', '#2a5fb6', '#b03a8c', '#3a9a5b', '#d36b1f', '#c9a227', '#5a3f9e', '#9e3d22']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha256(x):
    """Same digest as experiment.array_sha256 for one array: dtype, shape, then bytes."""
    x = np.ascontiguousarray(x)
    return hashlib.sha256(f'{x.dtype.str}{x.shape}'.encode() + x.tobytes()).hexdigest()


def partition():
    """Common pairs and their S, D, N masks (rows = common pairs, columns = labels)."""
    s30, s50 = (np.load(SPLITS[k]) for k in ('30%', '50%'))
    a, b, y = s30['a'], s30['b'], s30['yadd']
    assert all(np.array_equal(s30[k], s50[k]) for k in ('a', 'b', 'yadd'))
    p = int(y.max()) + 1
    t30, t50 = s30['train'], s50['train']
    assert set(t30) <= set(t50)
    for t in (t30, t50):  # transposes share a split, so the positional sets equal the either-position ones
        assert set((a[t]*p + b[t]).tolist()) == set((b[t]*p + a[t]).tolist())
    common = np.load(COMMON)['index']
    assert array_sha256(common) == COMMON_SHA256, 'common set hash'
    assert not set(common) & set(t50)
    L30 = operand_label_sets(a, b, y, t30, p)[common]
    L50 = operand_label_sets(a, b, y, t50, p)[common]
    assert not (L30 & ~L50).any()
    truth = np.eye(p, dtype=bool)[y[common]]
    assert not (L50 & truth).any(), 'an addition label co-occurs with an operand of its own pair'
    S, D = L30, L50 & ~L30
    N = ~L50 & ~truth
    assert (S.astype(int) + D + N + truth == 1).all(), 'S, D, N and the correct label must partition the labels'
    return dict(a=a, b=b, y=y, p=p, common=common, S=S, D=D, N=N, train={'30%': set(t30), '50%': set(t50)})


def checkpoints(run_dir):
    paths = [*run_dir.glob('checkpoint_*.npz'), *(SUPPLEMENT/run_dir.name).glob('checkpoint_*.npz')]
    return sorted(paths, key=lambda q: int(q.stem.split('_')[1]))


def score(run_dir, label, part):
    cfg = json.loads((run_dir/'config.json').read_text())
    manifest = json.loads((run_dir/'manifest.json').read_text())['files']
    logged = {r['step']: r for r in map(json.loads, (run_dir/'metrics.jsonl').read_text().splitlines())}
    with np.load(run_dir/'data.npz') as d:
        assert np.array_equal(d['y'], part['y']) and np.array_equal(d['a'], part['a'])
        train, audit = d['train'], d['audit']
    assert part['train']['30%'] <= set(train.tolist()) <= part['train']['50%'], 'training set does not nest'
    share = f"{cfg.get('train_fraction', .3):.0%}"
    a, b, y, c = part['a'], part['b'], part['y'], part['common']
    S, D, N = part['S'], part['D'], part['N']
    own_in = operand_label_sets(a, b, y, train, part['p'])[c]  # S at 30%, S | D at 50%, in between otherwise
    assert not (S & ~own_in).any() and not (own_in & ~(S | D)).any()
    own_out = ~own_in & ~np.eye(part['p'], dtype=bool)[y[c]]
    rows, per_pair = [], {}
    for path in checkpoints(run_dir):
        assert sha256(path) == manifest[path.name]['sha256'], path
        with np.load(path) as ck:
            step = int(ck['step'])
            z = np.maximum(ck['U'][a] + ck['V'][b], 0) @ ck['W']
        assert abs(np.mean(z[audit].argmax(1) == y[audit]) - logged[step]['audit']) < 1e-12, (path, 'audit accuracy')
        zc = z[c]; true = zc[np.arange(len(c)), y[c]]
        margin = lambda m: true - np.where(m, zc, -np.inf).max(1)  # +inf when the set is empty
        mS, mD, mN, mout, min_ = margin(S), margin(D), margin(N), margin(own_out), margin(own_in)
        correct = zc.argmax(1) == y[c]
        assert np.array_equal(correct, (mS > 0) & (mD > 0) & (mN > 0))
        lse = np.log(np.exp(zc - zc.max(1, keepdims=True)).sum(1)) + zc.max(1)
        med = lambda m: float(np.median(m[np.isfinite(m)])) if np.isfinite(m).any() else None
        rows.append(dict(group=label, run_id=run_dir.name, share=share, width=cfg['hidden'],
                         init_seed=cfg['seeds']['init_seed'], step=step,
                         train_ce=logged[step]['objective']['train']['cross_entropy'], common_acc=float(correct.mean()),
                         common_ce=float(np.mean(lse - true)), beats_S=float(np.mean(mS >= 0)),
                         beats_D=float(np.mean(mD >= 0)), beats_N=float(np.mean(mN >= 0)),
                         beats_own_out=float(np.mean(mout >= 0)),
                         partial=float(np.mean((mout >= 0) & (min_ < 0))),
                         median_margin_S=med(mS), median_margin_D=med(mD), median_margin_N=med(mN),
                         median_gap_S_over_N=med(mN - mS), median_gap_D_over_N=med(mN - mD)))
        per_pair[step] = dict(correct=correct, beats_out=mout >= 0)
    return rows, per_pair


def transitions(run_id, per_pair):
    """Between consecutive published checkpoints: do wrong pairs that already beat every label
    outside the run's own set become correct more often than wrong pairs that do not?"""
    out, steps = [], sorted(per_pair)
    for t0, t1 in zip(steps, steps[1:]):
        wrong = ~per_pair[t0]['correct']
        beat = per_pair[t0]['beats_out'] & wrong; other = ~per_pair[t0]['beats_out'] & wrong
        now = per_pair[t1]['correct']
        rate = lambda m: float(now[m].mean()) if m.any() else None
        out.append(dict(run_id=run_id, from_step=t0, to_step=t1, wrong_at_from=int(wrong.sum()),
                        wrong_and_beats_out=int(beat.sum()), correct_next_if_beats_out=rate(beat),
                        correct_next_otherwise=rate(other), newly_correct=int((now & wrong).sum())))
    return out


def figure(rows, labels, refs, out, name, title, note):
    fig, axes = plt.subplots(2, 5, figsize=(23, 8), sharex=True, layout='constrained')
    seeds = sorted({r['init_seed'] for r in rows})
    panels = [('common_acc', 'accuracy (argmax = correct label)', 1/31, 'chance 1/31'),
              ('beats_N', 'beats every N label (never seen with a or b)', refs['N'], 'no preference'),
              ('beats_S', 'beats every S label (seen with a or b at 30%)', refs['S'], 'no preference'),
              ('beats_D', 'beats every D label (seen with a or b only at 50%)', refs['D'], 'no preference'),
              ('median_gap_S_over_N', 'median gap: strongest S label minus strongest N label', 0, None),
              ('partial', 'beats every label outside its own set,\nloses to one inside', None, None),
              ('median_margin_N', 'median margin over strongest N label', 0, None),
              ('median_margin_S', 'median margin over strongest S label', 0, None),
              ('median_margin_D', 'median margin over strongest D label', 0, None),
              ('median_gap_D_over_N', 'median gap: strongest D label minus strongest N label', 0, None)]
    # Width groups ("width 128") keep the e1 map's width colors.
    colors = [WIDTH_COLORS[int(l.split()[1])] if l.startswith('width ') else c for c, l in zip(COLORS, labels)]
    for color, label in zip(colors, labels):
        for run in dict.fromkeys(r['run_id'] for r in rows if r['group'] == label):
            rs = [r for r in rows if r['run_id'] == run and r['step'] > 0]
            style = dict(color=color, lw=1.1, ls=SEED_STYLES[seeds.index(rs[0]['init_seed']) % 4], marker='o', ms=3)
            for ax, (key, *_) in zip(axes.flat, panels):
                pts = [(r['step'], r[key]) for r in rs if r[key] is not None]
                ax.plot(*zip(*pts), label=f"{label}, init {rs[0]['init_seed']}", **style)
    for ax, (key, ylabel, ref, ref_label) in zip(axes.flat, panels):
        ax.set_xscale('log'); ax.grid(True, color='0.9', lw=.6); ax.set_title(ylabel, fontsize=9)
        if key.startswith('median'):
            ax.axhline(0, color='0.4', lw=.8); ax.set_ylabel('logit difference')
        else:
            ax.set_ylim(-.02, 1.02); ax.set_ylabel('fraction of common pairs')
            if ref is not None:
                ax.axhline(ref, color='0.4', lw=.9, ls=(0, (4, 3)), label=f'{ref_label} {ref:.3f}')
                ax.legend(handles=ax.get_lines()[-1:], fontsize=7, frameon=False, loc='upper right')
    for ax in axes[1]:
        ax.set_xlabel('update step (log scale)')
    axes[0, 0].legend(fontsize=7, frameon=False, loc='center left')
    fig.suptitle(title, fontsize=11)
    save(fig, out, name, note)


def fit_figure(rows, labels, out, name):
    """The same medians against the run's own training cross-entropy instead of steps."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout='constrained')
    seeds = sorted({r['init_seed'] for r in rows})
    colors = [WIDTH_COLORS[int(l.split()[1])] if l.startswith('width ') else c for c, l in zip(COLORS, labels)]
    keys = [('median_gap_S_over_N', 'median gap: strongest S minus strongest N label'),
            ('median_margin_N', 'median margin over strongest N label'),
            ('median_margin_S', 'median margin over strongest S label')]
    for color, label in zip(colors, labels):
        for run in dict.fromkeys(r['run_id'] for r in rows if r['group'] == label):
            rs = [r for r in rows if r['run_id'] == run and r['step'] > 0]
            style = dict(color=color, lw=1.1, ls=SEED_STYLES[seeds.index(rs[0]['init_seed']) % 4], marker='o', ms=3)
            for ax, (key, _) in zip(axes, keys):
                ax.plot([r['train_ce'] for r in rs], [r[key] for r in rs], label=f"{label}, init {rs[0]['init_seed']}", **style)
    for ax, (_, title) in zip(axes, keys):
        ax.set_xscale('log'); ax.invert_xaxis(); ax.grid(True, color='0.9', lw=.6); ax.axhline(0, color='0.4', lw=.8)
        ax.set_title(title, fontsize=9); ax.set_ylabel('logit difference')
        ax.set_xlabel("run's own training cross-entropy (nats, log scale, decreasing to the right)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Common-pair margins against training fit', fontsize=11)
    save(fig, out, name, 'Published checkpoints only (markers), step 0 omitted. Training cross-entropy is the logged value at '
         'the checkpoint step, on each run\'s own training set (sizes differ between shares). Same pairs and label sets as '
         f'{name.replace("_fit", "")}.png.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', nargs='+', action='append', required=True, metavar='LABEL RUN_DIR')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--name', default='margins')
    ap.add_argument('--title', default='Correct-label margins on the 483 pairs neither share trained on')
    args = ap.parse_args()
    part = partition()
    sizes = {k: part[k].sum(1) for k in 'SDN'}
    refs = {k: float(np.mean(1/(v + 1))) for k, v in sizes.items()}
    rows, trans = [], []
    for label, *dirs in args.group:
        for d in map(Path, dirs):
            r, per_pair = score(d, label, part)
            rows += r; trans += transitions(d.name, per_pair)
    coordinator = {(r['run_id'], int(r['step'])): float(r['common_acc']) for r in csv.DictReader(COORDINATOR_SCORES.open())}
    checked = [r for r in rows if (r['run_id'], r['step']) in coordinator]
    assert all(abs(r['common_acc'] - coordinator[r['run_id'], r['step']]) < 1e-12 for r in checked)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/f'{args.name}.csv', rows)
    write_csv(args.out/f'{args.name}_transitions.csv', trans)
    summary = dict(common_pairs=len(part['common']), common_sha256=COMMON_SHA256, no_preference=refs,
                   checked_against_coordinator_scores=len(checked),
                   **{f'size_{k}': dict(mean=float(v.mean()), min=int(v.min()), max=int(v.max()), empty=int((v == 0).sum()))
                      for k, v in sizes.items()})
    (args.out/'sizes.json').write_text(json.dumps(summary, indent=1) + '\n')
    s = {k: summary[f'size_{k}'] for k in 'SDN'}
    note = (f"Published checkpoints only (markers), step 0 omitted; e1 width-128 runs i30001 and i30002 include the extra "
            f"checkpoints in research/e2/dynamics/e1-checkpoints/. Per common pair the 30 wrong labels are split into S "
            f"(mean {s['S']['mean']:.1f} labels), D ({s['D']['mean']:.1f}; empty for {s['D']['empty']} pairs) and N "
            f"({s['N']['mean']:.1f}; empty for {s['N']['empty']}), the same sets for every run; a run's own set is S at 30% "
            f"and S+D at 50%. Margin = correct-label logit minus the highest logit in the set. Dashed: fraction expected "
            f"if the correct label were ranked at random within the set. Source: {args.name}.csv.")
    figure(rows, [g[0] for g in args.group], refs, args.out, args.name, args.title, note)
    fit_figure(rows, [g[0] for g in args.group], args.out, f'{args.name}_fit')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
