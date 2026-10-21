#!/usr/bin/env python3
"""Balanced-gauge normalization on analysis copies, with the rescaling invariance
check repeated on the normalized weights.

Convention: each hidden unit gets equal input and output norms,
||u_j||² + ||v_j||² = ||w_j||², via the positive rescaling
c_j = (||w_j||² / (||u_j||² + ||v_j||²))^(1/4). That is the minimum-L2 point of the
unit's rescaling orbit, so every positively rescaled copy of a model normalizes to
the same weights. It removes this one symmetry only: not hidden-unit permutations,
not other equivalent networks, and it says nothing about minimum program length.
It is applied to analysis copies; no training model is normalized. A unit with
exactly zero input or output weights computes nothing and is left unchanged, so its
input weights stay gauge-dependent (gauge_summary.json counts such units).

uv run --locked --group analysis python research/mechanisms/gauge.py \
    --run research/coordinator/runs/e1-co-add-w128-i30000-a1 --out research/mechanisms/figures/NAME
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freq import (RESCALE_SEED, SIGMAS, ROOT, codec, curve, evaluate, kendall_tau, load_run, ranking,  # noqa: E402
                  rescale, sha256, spectrum, synthetic, write_csv)
from experiment import array_sha256  # noqa: E402


def live_units(w):
    """Units with nonzero input and output weights; the others compute nothing."""
    return (np.sum(w[0]**2, 0) + np.sum(w[1]**2, 0) > 0) & (np.sum(w[2]**2, 1) > 0)


def balance(w):
    """Balanced copy of w and the gains used (gain 1 for units that compute nothing)."""
    a = np.sum(w[0]**2, 0) + np.sum(w[1]**2, 0)
    b = np.sum(w[2]**2, 1)
    c = np.ones_like(a)
    live = live_units(w)
    c[live] = (b[live]/a[live])**.25
    return rescale(w, c), c


def norm(w):
    return float(np.sqrt(sum(np.sum(x*x) for x in w)))


def sweep(w, d, draws, top):
    """Same random gains as freq.py's rescaling sweep (same seed and draw order)."""
    wb, _ = balance(w)
    base_logits = evaluate(w, d)[1]
    r0, rb = ranking(w), ranking(wb)
    scale = max(np.abs(x).max() for x in wb)
    rng = np.random.default_rng(RESCALE_SEED)
    rows = []
    for s in SIGMAS:
        for i in range(draws):
            wr = rescale(w, np.exp(s*rng.standard_normal(w[0].shape[1])))
            wrb, _ = balance(wr)
            rr, rrb = ranking(wr), ranking(wrb)
            raw, bal = codec(wr, d), codec(wrb, d)
            rows.append(dict(sigma=s, draw=i,
                             max_rel_weight_diff_after_balancing=float(max(np.abs(x - y).max()
                                                                          for x, y in zip(wrb, wb))/scale),
                             max_abs_logit_diff=float(np.abs(evaluate(wrb, d)[1] - base_logits).max()),
                             raw_topK_changed=bool(set(rr[:top]) != set(r0[:top])),
                             raw_kendall_tau=kendall_tau(r0, rr), raw_codec_bits=raw['codec_bits'],
                             balanced_ranking_equal=bool(np.array_equal(rrb, rb)),
                             balanced_kendall_tau=kendall_tau(rb, rrb), balanced_codec_bits=bal['codec_bits'],
                             balanced_codec_q=bal['codec_q'], balanced_codec_k=bal['codec_k']))
    return rows


def plot(w, wb, c, d, rows, top, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    live = live_units(w)
    ax[0, 0].hist(np.log2(c[live]), bins=30, color='#777777')
    ax[0, 0].axvline(0, color='k', lw=.8)
    ax[0, 0].set(xlabel='log2 c_j (gain that balances unit j; 0 = already balanced)', ylabel='hidden units',
                 title=f'distance of the checkpoint from the balanced gauge ({live.sum()} units)')
    e0, eb = spectrum(w).sum(1)[1:], spectrum(wb).sum(1)[1:]
    freqs = np.arange(1, len(e0) + 1)
    ax[0, 1].bar(freqs - .2, e0/e0.sum(), .4, color='#777777', label='checkpoint')
    ax[0, 1].bar(freqs + .2, eb/eb.sum(), .4, color='#3a9a5b', label='balanced copy')
    for e, dx, color in ((e0, -.2, '#777777'), (eb, .2, '#3a9a5b')):
        for i in np.argsort(-e, kind='stable')[:top]:
            ax[0, 1].annotate('*', (freqs[i] + dx, e[i]/e.sum()), ha='center', va='bottom', color=color)
    ax[0, 1].set(xlabel='frequency k', ylabel='share of non-DC input energy', xticks=freqs,
                 ylim=(0, 1.35*max(e0.max()/e0.sum(), eb.max()/eb.sum())), title=f'energy ranking; * marks each top {top}')
    ax[0, 1].legend(fontsize=8, loc='upper center', ncol=2)
    r0, rb = curve(w, d, ranking(w), ('retain',)), curve(wb, d, ranking(wb), ('retain',))
    for split, color in (('train', '#d36b1f'), ('audit', '#2a7bb6')):
        ax[0, 2].plot([r['K'] for r in r0], [r[f'{split}_accuracy'] for r in r0], 'o-', ms=3, color=color,
                      label=f'{split}, checkpoint ranking')
        ax[0, 2].plot([r['K'] for r in rb], [r[f'{split}_accuracy'] for r in rb], 'x--', ms=5, color=color,
                      label=f'{split}, balanced ranking')
    ax[0, 2].set(xlabel='K = top-K frequencies retained', ylabel='accuracy', xticks=np.arange(len(e0) + 1),
                 title='same function, retention under each ranking')
    ax[0, 2].legend(fontsize=7)
    sig = np.array([r['sigma'] for r in rows])
    jit = np.random.default_rng(0).uniform(-.012, .012, len(sig))
    ticks = dict(xscale='log', xticks=SIGMAS, xticklabels=[str(s) for s in SIGMAS], xlim=(.07, 1.6),
                 xlabel='rescaling log-SD sigma of the copy before balancing')
    ax[1, 0].scatter(sig + jit, [max(r['max_rel_weight_diff_after_balancing'], 1e-19) for r in rows], s=10,
                     color='#3a9a5b', label='balanced rescaled copy vs balanced checkpoint')
    ax[1, 0].axhline(np.finfo(float).eps, color='k', ls=':', lw=1, label='float64 machine epsilon')
    ax[1, 0].set(yscale='log', ylim=(1e-19, 1), ylabel='max |weight difference| / max |weight|',
                 title='normalization removes the rescaling', **ticks)
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].scatter(sig + jit, [r['raw_codec_bits'] for r in rows], s=10, color='#5a3f9e', label='rescaled copy (raw)')
    ax[1, 1].scatter(sig - jit, [r['balanced_codec_bits'] for r in rows], s=18, marker='x', color='#3a9a5b',
                     label='same copy, balanced')
    ax[1, 1].set(ylabel='codec bits (pruning + quantization)', title='compress() code length', **ticks)
    ax[1, 1].legend(fontsize=8)
    ax[1, 2].scatter(sig + jit, [r['raw_kendall_tau'] for r in rows], s=10, color='#5a3f9e',
                     label='raw: vs checkpoint ranking')
    ax[1, 2].scatter(sig - jit, [r['balanced_kendall_tau'] for r in rows], s=18, marker='x', color='#3a9a5b',
                     label='balanced: vs balanced checkpoint ranking')
    ax[1, 2].set(ylim=(-1.05, 1.05), ylabel='Kendall tau', title='frequency ranking', **ticks)
    ax[1, 2].legend(fontsize=8, loc='lower left')
    fig.suptitle(title, fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--run', type=Path, help='Published run directory with manifest.json')
    src.add_argument('--synthetic', action='store_true', help='Hand-built network on the frozen e1 split')
    ap.add_argument('--step', type=int, help='Checkpoint step (default: last published)')
    ap.add_argument('--source-commit', help='Commit the published run was read from (recorded only)')
    ap.add_argument('--label', help='Checkpoint description for titles')
    ap.add_argument('--top', type=int, default=4)
    ap.add_argument('--draws', type=int, default=16)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.synthetic:
        raw = np.load(ROOT/'research/data/e1-split.npz')
        d = dict(a=raw['a'], b=raw['b'], y=raw['yadd'], yadd=raw['yadd'], train=raw['train'], cal=raw['cal'],
                 audit=raw['audit'])
        w = synthetic(d)
        meta = dict(run_id='synthetic', purpose='synthetic code check: hand-built, not trained',
                    split_sha256=array_sha256(d['a'], d['b'], d['yadd'], d['train'], d['cal'], d['audit']))
    else:
        w, d, meta = load_run(args.run, args.step)
        meta['read_from_commit'] = args.source_commit
    args.out.mkdir(parents=True, exist_ok=True)
    wb, c = balance(w)
    logits, logits_b = evaluate(w, d)[1], evaluate(wb, d)[1]
    rows = sweep(w, d, args.draws, args.top)
    r0, rb = ranking(w), ranking(wb)
    eb = spectrum(wb).sum(1)[1:]
    live = live_units(w)
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    summary = dict(label=args.label or meta['purpose'], input=meta,
                   balance_gain_log2=dict(median_abs=float(np.median(np.abs(np.log2(c[live])))) if live.any() else 0.,
                                          min=float(np.log2(c).min()), max=float(np.log2(c).max()),
                                          dead_units=int((~live).sum())),
                   balanced_max_abs_logit_diff=float(np.abs(logits_b - logits).max()),
                   param_norm=dict(checkpoint=norm(w), balanced=norm(wb)),
                   ranking=dict(checkpoint=r0.tolist(), balanced=rb.tolist(), kendall_tau=kendall_tau(r0, rb),
                                topK_changed=bool(set(r0[:args.top]) != set(rb[:args.top]))),
                   balanced_energy_share=(eb/eb.sum()).tolist(),
                   codec=dict(checkpoint=codec(w, d), balanced=codec(wb, d)),
                   invariance=dict(draws=len(rows),
                                   max_rel_weight_diff=max(r['max_rel_weight_diff_after_balancing'] for r in rows),
                                   balanced_ranking_equal_all=all(r['balanced_ranking_equal'] for r in rows),
                                   balanced_codec_values=sorted({r['balanced_codec_bits'] for r in rows}),
                                   raw_codec_values=sorted({r['raw_codec_bits'] for r in rows})),
                   analysis_code_commit=git('rev-parse', 'HEAD'), analysis_tree_status=git('status', '--porcelain'),
                   analysis_script_sha256=sha256(__file__), command=' '.join(['python', *sys.argv]))
    (args.out/'gauge_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    write_csv(args.out/'gauge_sweep.csv', rows)
    write_csv(args.out/'gauge_retention.csv', curve(wb, d, rb))
    name = meta['run_id'] + (f" step {meta['step']}" if 'step' in meta else '')
    plot(w, wb, c, d, rows, args.top, f"{name} ({summary['label']}): balanced-gauge normalization of analysis copies",
         args.out/'gauge')
    print(json.dumps({k: summary[k] for k in ('balance_gain_log2', 'param_norm', 'codec', 'invariance')}))


if __name__ == '__main__':
    main()
