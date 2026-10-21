#!/usr/bin/env python3
"""Weights-only edits that isolate each unit's frequency structure, scored by plain argmax.

Question and predictions: QUESTION.md, cycle 2. No edit looks at labels, pairs or
a + b - c. For each live unit j, k_j is its dominant input frequency (units.py).
- filter: U[:, j], V[:, j] (token axis) and W[j, :] (label axis) projected onto rfft
  bin k_j; the DC of U and V is kept (it sets the unit's gate), W's DC changes no argmax.
  Added after seeing the first result: variants (inputs only, output only, input DC removed)
  and a null that rotates each unit's kept component of U, V and W by independent random
  phases (same frequencies and energies; 8 draws).
- scale: W[j, :] multiplied by s for the units whose dominant input and output
  frequencies agree; s = inf keeps only those units.
Nulls: the same edit at step 0, the filter to a random non-dominant frequency per unit
(seed 36000), the scaling of random live-unit sets matched in count and total
|u_j||w_j| within 10%, and the e1 shuffled-label runs. Comparison added at the close of
cycle 2 (DECISIONS D10): the e1 no-decay runs, same split and a+b labels, weight decay 0. Scores: argmax accuracy against
a + b on the common never-trained set (research/e2/data) and on the run's own cal + audit;
train accuracy against the run's own training labels; RMS pair-centred logit change.

uv run --locked --group analysis python research/e2/mechanisms/unitfilter.py --out research/e2/mechanisms/figures/unitfilter \
    --runs research/coordinator/runs/e1f2-co-add-w128-i30000-tf50-a1:0,50000 ...
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/mechanisms'))
from freq import load_run, sha256, write_csv  # noqa: E402
from gauge import live_units  # noqa: E402
from units import concentration, unit_shares  # noqa: E402

SEED = 36000
DRAWS = 8
SCALES = (2, 4, 8, np.inf)
COMMON = ROOT/'research/e2/data/common-never-trained-30-50.npz'


def dominant(w):
    """Dominant input and output frequency of every unit (1 for units that compute nothing)."""
    live = live_units(w)
    with np.errstate(invalid='ignore'):
        s_in, s_out = (s[:, live] for s in unit_shares(w))
    k_in, k_out = np.ones(len(live), int), np.ones(len(live), int)
    k_in[live], k_out[live] = s_in.argmax(0) + 1, s_out.argmax(0) + 1
    return k_in, k_out, live


def filter_units(w, k, inputs=True, outputs=True, dc=True):
    """Each unit's U and V columns and/or W row reduced to frequency k[j] (plus, by default, the DC of U and V)."""
    p = w[0].shape[0]
    keep = np.zeros((p//2 + 1, len(k)), bool)
    keep[k, np.arange(len(k))] = True
    out_keep = keep.copy()
    keep[0] = dc
    u, v = (np.fft.irfft(np.fft.rfft(x, axis=0)*keep, n=p, axis=0) if inputs else x for x in w[:2])
    return [u, v, np.fft.irfft(np.fft.rfft(w[2], axis=1)*out_keep.T, n=p, axis=1) if outputs else w[2]]


def rotate_phases(w, k, rng):
    """Each unit's frequency-k[j] component of U, V and W rotated by an independent uniform phase."""
    p, n = w[0].shape[0], len(k)
    out = []
    for x, axis in ((w[0], 0), (w[1], 0), (w[2], 1)):
        rot = np.ones((p//2 + 1, n), complex)
        rot[k, np.arange(n)] = np.exp(2j*np.pi*rng.random(n))
        out.append(np.fft.irfft(np.fft.rfft(x, axis=axis)*(rot if axis == 0 else rot.T), n=p, axis=axis))
    return out


def other_frequency(k, rng):
    """A uniformly random frequency in 1..15 other than k, per unit."""
    return (k - 1 + rng.integers(1, 15, len(k))) % 15 + 1


def scale_units(w, units, s):
    out = w[2].copy()
    if np.isinf(s):
        rest = np.ones(len(out), bool)
        rest[units] = False
        out[rest] = 0
    else:
        out[units] *= s
    return [w[0], w[1], out]


def logits(w, d):
    return np.maximum(w[0][d['a']] + w[1][d['b']], 0) @ w[2]


def score(w, d, base):
    z = logits(w, d)
    change = (z - z.mean(1, keepdims=True)) - (base - base.mean(1, keepdims=True))
    acc = lambda ix, y: float(np.mean(z[ix].argmax(1) == y[ix]))
    return dict(common_acc=acc(d['common'], d['yadd']), own_heldout_acc=acc(np.concatenate([d['cal'], d['audit']]), d['yadd']),
                train_acc=acc(d['train'], d['y']), train_acc_vs_a_plus_b=acc(d['train'], d['yadd']),
                logit_change_rms=float(np.sqrt(np.mean(change**2))))


def kept(w, e):
    return dict(kept_input_energy=float(np.sum(e[0]**2 + e[1]**2)/np.sum(w[0]**2 + w[1]**2)),
                kept_output_energy=float(np.sum(e[2]**2)/np.sum(w[2]**2)))


def analyse(run, step, rng):
    w, d, meta = load_run(run, step)
    d['common'] = np.load(COMMON)['index']
    assert not set(d['common'].tolist()) & set(d['train'].tolist())
    base = logits(w, d)
    k_in, k_out, live = dominant(w)
    aligned = np.flatnonzero(live & (k_in == k_out))
    labels = 'shuffled' if '-shuf-' in meta['run_id'] else 'a+b'
    info = dict(run_id=meta['run_id'], step=step, train_pairs=len(d['train']), labels=labels,
                aligned_units=len(aligned), **concentration(w))
    row = lambda kind, draw, e, scale='', n=int(live.sum()): dict(info, kind=kind, scale=scale, draw=draw, n_units=n,
                                                                 **kept(w, e), **score(e, d, base))
    rows = [row('none', 0, w), row('filter dominant', 0, filter_units(w, k_in))]
    # Variants added after seeing the first result: which part of the filter matters.
    for kind, opts in (('filter dominant, inputs only', dict(outputs=False)), ('filter dominant, output only', dict(inputs=False)),
                       ('filter dominant, input DC removed', dict(dc=False))):
        rows.append(row(kind, 0, filter_units(w, k_in, **opts)))
    for draw in range(DRAWS):
        rows.append(row('filter random non-dominant', draw, filter_units(w, other_frequency(k_in, rng))))
    for draw in range(DRAWS):  # added after seeing the first result: a null matched on concentration and energy
        rows.append(row('filter dominant, phases scrambled', draw, rotate_phases(filter_units(w, k_in), k_in, rng)))
    size = np.sqrt(np.sum(w[0]**2, 0) + np.sum(w[1]**2, 0))*np.linalg.norm(w[2], axis=1)
    pool, target, sets = np.flatnonzero(live), size[aligned].sum(), []
    for _ in range(200*DRAWS):
        pick = rng.choice(pool, len(aligned), replace=False)
        if abs(size[pick].sum()/target - 1) <= .1:
            sets.append(pick)
            if len(sets) == DRAWS:
                break
    for s in SCALES:
        rows.append(row('scale aligned', 0, scale_units(w, aligned, s), s, len(aligned)))
        rows += [row('scale random set', i, scale_units(w, pick, s), s, len(aligned)) for i, pick in enumerate(sets)]
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True, help='RUN_DIR:STEP,STEP,...')
    ap.add_argument('--source-commit', help='Commit the published runs were read from (recorded only)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows, inputs = [], []
    for spec in args.runs:
        run, steps = spec.split(':')
        for step in map(int, steps.split(',')):
            r, meta = analyse(ROOT/run, step, rng)
            rows += r
            inputs.append({k: meta[k] for k in ('run_id', 'step', 'checkpoint_sha256', 'split_sha256', 'target_sha256',
                                                'max_abs_diff_vs_logged_metrics', 'manifest_hashes_verified')})
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'unitfilter.csv', rows)
    (args.out/'unitfilter_summary.json').write_text(json.dumps(dict(
        inputs=inputs, common_set_sha256=sha256(COMMON), seed=SEED, draws=DRAWS, scales=[str(s) for s in SCALES],
        read_from_commit=args.source_commit, session_label='e2-me-cloud-1', analysis_code_commit=git('rev-parse', 'HEAD'),
        analysis_tree_status=git('status', '--porcelain'), analysis_script_sha256=sha256(__file__),
        command=' '.join(['python', *sys.argv])), indent=1) + '\n')
    plot(rows, args.out/'unitfilter')


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    cls = lambda run: 'shuffled' if '-shuf-' in run else 'no decay' if '-add0-' in run else '50%' if 'tf50' in run else '30%'
    order = {'50%': 0, '30%': 1, 'no decay': 2, 'shuffled': 3}
    runs = sorted({r['run_id'] for r in rows}, key=lambda r: (order[cls(r)], r))
    color = {'50%': '#1f6fb4', '30%': '#c0392b', 'no decay': '#e08a1e', 'shuffled': '#7d3c98'}
    col = {run: color[cls(run)] for run in runs}
    short = {'50%': '50%', '30%': '30%', 'no decay': 'wd 0', 'shuffled': 'shuf.'}
    label = {run: short[cls(run)] + '\ni' + run.split('-i')[1][:5] for run in runs}
    final = {run: max(r['step'] for r in rows if r['run_id'] == run) for run in runs}
    first = {run: min(r['step'] for r in rows if r['run_id'] == run) for run in runs}
    sel = lambda run, step, kind, scale='': [r for r in rows if r['run_id'] == run and r['step'] == step
                                             and r['kind'] == kind and str(r['scale']) == str(scale)]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5), constrained_layout=True)
    for panel, key, title in ((ax[0, 0], 'common_acc', 'common set (483 pairs neither split trains on)'),
                              (ax[0, 1], 'own_heldout_acc', "run's own held-out split (cal + audit)"),
                              (ax[0, 2], 'train_acc', "training pairs, against the run's own training labels")):
        for i, run in enumerate(runs):
            for step, kind, x0, style in ((first[run], 'filter dominant', -.33, dict(mfc='none', mec=col[run], marker='o', ms=6)),
                                          (final[run], 'filter random non-dominant', -.11, dict(color='#999999', marker='.', ms=5)),
                                          (final[run], 'filter dominant, phases scrambled', .11,
                                           dict(color='#999999', marker='d', ms=3)),
                                          (final[run], 'none', .33, dict(color='k', marker='x', ms=7)),
                                          (final[run], 'filter dominant', .33, dict(color=col[run], marker='o', ms=7))):
                v = [r[key] for r in sel(run, step, kind)]
                panel.plot(i + x0 + (rng.uniform(-.06, .06, len(v)) if len(v) > 1 else 0), v, ls='none', **style)
        panel.axhline(1/31, color='k', ls=':', lw=1)
        panel.tick_params(axis='x', labelsize=8)
        panel.set(xticks=range(len(runs)), xticklabels=[label[r] for r in runs], ylim=(-.03, 1.03),
                  ylabel='argmax accuracy' + (' (a+b)' if key != 'train_acc' else ''), title=title)
    for run in runs:
        steps = sorted({r['step'] for r in rows if r['run_id'] == run})
        for kind, ls in (('filter dominant', '-'), ('none', '--')):
            ax[1, 0].plot([max(s, 100) for s in steps], [sel(run, s, kind)[0]['common_acc'] for s in steps],
                          color=col[run], ls=ls, marker='o', ms=3)
    ax[1, 0].axhline(1/31, color='k', ls=':', lw=1)
    ax[1, 0].set(xscale='log', ylim=(-.03, 1.03), xlabel='update step (step 0 drawn at 100)', ylabel='common-set accuracy (a+b)',
                 title='over training: filtered (solid) vs unedited (dashed)\n(dominant frequencies read at each checkpoint)')
    for run in [r for r in runs if cls(r) != 'shuffled']:
        for kind, style in (('filter random non-dominant', dict(color='#999999', marker='.', s=10)),
                            ('filter dominant, phases scrambled', dict(color='#999999', marker='d', s=6)),
                            ('filter dominant', dict(color=col[run], marker='o', s=22))):
            v = [r for r in rows if r['run_id'] == run and r['kind'] == kind]
            ax[1, 1].scatter([r['train_acc'] for r in v], [r['common_acc'] for r in v], **style)
    ax[1, 1].plot([0, 1], [0, 1], color='k', lw=.8)
    ax[1, 1].axhline(1/31, color='k', ls=':', lw=1)
    ax[1, 1].set(xlabel='train accuracy after the edit', ylabel='common-set accuracy (a+b)', xlim=(-.03, 1.03), ylim=(-.03, 1.03),
                 title='filter edits at every analysed checkpoint, a+b runs\n(held-out against retained training fit; line: equal)')
    xs = {2: 1, 4: 2, 8: 3, np.inf: 4}
    for run in [r for r in runs if cls(r) != '50%']:
        base = sel(run, final[run], 'none')[0]['common_acc']
        for kind, ls in (('scale aligned', '-'), ('scale random set', ':')):
            ax[1, 2].plot([0, *xs.values()], [base] + [np.median([r['common_acc'] for r in sel(run, final[run], kind, s)])
                                                      for s in xs], color=col[run], ls=ls, marker='o', ms=3)
    ax[1, 2].axhline(1/31, color='k', ls=':', lw=1)
    ax[1, 2].set(xticks=range(5), xticklabels=['1', '2', '4', '8', 'aligned\nonly'], ylim=(-.03, 1.03),
                 xlabel='output-row scale s', ylabel='common-set accuracy (a+b)',
                 title='30%, no-decay and shuffled-label runs, final step:\nupweight frequency-aligned units (solid)\n'
                       'or matched random sets (dotted, median)')
    keys = [Line2D([], [], color='k', marker='x', ls='none', label='unedited, final step'),
            Line2D([], [], color='#555555', marker='o', ls='none', label='each unit filtered to its dominant frequency, final step'),
            Line2D([], [], mfc='none', mec='#555555', marker='o', ls='none', label='the same filter at the first step (0, or 300)'),
            Line2D([], [], color='#999999', marker='.', ls='none', label='each unit filtered to a random other frequency (8 draws)'),
            Line2D([], [], color='#999999', marker='d', ls='none', ms=4,
                   label='dominant-frequency filter with phases scrambled (8 draws)'),
            Line2D([], [], color='k', ls=':', label='chance 1/31')]
    keys += [Line2D([], [], color=c, lw=6, label=f'{n} runs') for n, c in
             (('50% share', color['50%']), ('30% share', color['30%']), ('30% share, weight decay 0', color['no decay']),
              ('shuffled-label (30% split)', color['shuffled']))]
    fig.legend(handles=keys, loc='outside lower center', ncol=4, fontsize=9)
    fig.suptitle('Width 128, weights-only edits scored by plain argmax: 50% share (blue), 30% share (red), '
                 '30% share without weight decay (orange), shuffled training labels (purple)', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
