#!/usr/bin/env python3
"""Frequency retention/removal curves and a positive-rescaling invariance audit
for one checkpoint, computed on analysis copies (nothing here trains).

Frequencies are rfft bins of the input embeddings U and V along the token axis,
ranked by energy summed over hidden units: the historical `interventions`
definition in experiment.py, generalized from K=4 to every K. W is never
filtered. Selection is checkpoint-local: the ranking comes from the analysed
checkpoint itself, not from a later one.

uv run --locked --group analysis python research/mechanisms/freq.py \
    --run research/coordinator/runs/e1-co-add-w128-i30000-a1 --out research/mechanisms/figures/NAME
uv run --locked --group analysis python research/mechanisms/freq.py --synthetic --out research/mechanisms/figures/synthetic
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment import array_sha256, compress, forward, objective_metrics  # noqa: E402

SPLITS = ('train', 'cal', 'audit')
RANDOM_SEED = 33100   # random frequency subsets
RESCALE_SEED = 33200  # random rescalings
SIGMAS = (0.1, 0.25, 0.5, 1.0)


def spectrum(w):
    """Per-unit energy of U plus V in each rfft bin, shape (bins, hidden)."""
    return sum(np.abs(np.fft.rfft(x, axis=0))**2 for x in w[:2])


def ranking(w):
    """Nonzero frequencies by total input energy, highest first (stable ties)."""
    return np.argsort(-spectrum(w).sum(1)[1:], kind='stable') + 1


def filter_inputs(w, freqs, keep):
    """keep=True: DC plus `freqs` only; keep=False: `freqs` removed. W is unchanged."""
    out = []
    for x in w[:2]:
        f = np.fft.rfft(x, axis=0)
        g = np.zeros_like(f) if keep else f.copy()
        idx = np.asarray(freqs, int)
        if keep:
            g[0] = f[0]
            g[idx] = f[idx]
        else:
            g[idx] = 0
        out.append(np.fft.irfft(g, n=x.shape[0], axis=0))
    return out + [w[2]]


def rescale(w, c):
    """Positive gain c_j on unit j's inputs and 1/c_j on its outputs; relu(c z)/c = relu(z)."""
    return [w[0]*c, w[1]*c, w[2]/c[:, None]]


def evaluate(w, d):
    logits = forward(w, d['a'], d['b'])[0]
    row = {}
    for s in SPLITS:
        ix = d[s]
        row[f'{s}_accuracy'] = float(np.mean(logits[ix].argmax(1) == d['y'][ix]))
        row.update({f'{s}_{k}': v for k, v in objective_metrics(logits, d['y'], ix).items()})
    return row, logits


def curve(w, d, order, modes=('retain', 'remove')):
    energy = spectrum(w).sum(1)[1:]
    rows = []
    for mode in modes:
        for k in range(len(order) + 1):
            chosen = order[:k]
            metrics, _ = evaluate(filter_inputs(w, chosen, mode == 'retain'), d)
            rows.append(dict(mode=mode, K=k, frequencies=' '.join(map(str, chosen)),
                             energy_fraction=float(energy[chosen - 1].sum()/energy.sum()), **metrics))
    return rows


def random_subsets(w, d, n, seed):
    """Uniform random K-subsets of the nonzero frequencies: a reference, not energy-matched."""
    rng = np.random.default_rng(seed)
    energy = spectrum(w).sum(1)[1:]
    freqs = np.arange(1, len(energy) + 1)
    rows = []
    for mode in ('retain', 'remove'):
        for k in range(1, len(freqs)):
            for i in range(n):
                chosen = np.sort(rng.choice(freqs, k, replace=False))
                metrics, _ = evaluate(filter_inputs(w, chosen, mode == 'retain'), d)
                rows.append(dict(mode=mode, K=k, draw=i, frequencies=' '.join(map(str, chosen)),
                                 energy_fraction=float(energy[chosen - 1].sum()/energy.sum()), **metrics))
    return rows


def kendall_tau(r1, r2):
    """Kendall tau between two orderings of the same frequencies."""
    pos1, pos2 = np.empty(len(r1)), np.empty(len(r2))
    pos1[np.asarray(r1) - 1] = np.arange(len(r1))
    pos2[np.asarray(r2) - 1] = np.arange(len(r2))
    i, j = np.triu_indices(len(r1), 1)
    return float(np.mean(np.sign(pos1[i] - pos1[j])*np.sign(pos2[i] - pos2[j])))


def codec(w, d):
    c = compress(w, d['a'], d['b'], [d[s] for s in SPLITS])
    return dict(codec_bits=c['bits'], codec_k=c['k'], codec_q=c['q'])


def prune_order(w):
    # The unit score inside compress(); it is invariant to positive rescaling.
    score = (np.linalg.norm(w[0], axis=0) + np.linalg.norm(w[1], axis=0))*np.linalg.norm(w[2], axis=1)
    return np.argsort(-score, kind='stable')


def compare(w, wr, d, base_logits, top):
    """Function and proxy differences between w and a rescaled copy wr."""
    _, logits = evaluate(wr, d)
    r0, r1 = ranking(w), ranking(wr)
    return dict(max_abs_logit_diff=float(np.abs(logits - base_logits).max()),
                max_rel_logit_diff=float(np.abs(logits - base_logits).max()/np.abs(base_logits).max()),
                ranking=' '.join(map(str, r1)), topK_changed=bool(set(r0[:top]) != set(r1[:top])),
                kendall_tau=kendall_tau(r0, r1),
                prune_order_unchanged=bool(np.array_equal(prune_order(w), prune_order(wr))),
                param_norm=float(np.sqrt(sum(np.sum(x*x) for x in wr))), **codec(wr, d))


def targeted(w, top):
    """Smallest uniform gain on the units dominated by the rank-(top+1) frequency
    that moves it into the top `top` (geometric grid, ratio 1.01, up to 1000)."""
    order = ranking(w)
    target = int(order[top])
    units = spectrum(w)[1:].argmax(0) + 1 == target
    if not units.any():
        return dict(target=target, units=0, gain=None), None
    for gain in np.geomspace(1.01, 1000, 695):
        c = np.where(units, gain, 1.)
        if target in ranking(rescale(w, c))[:top]:
            return dict(target=target, units=int(units.sum()), gain=float(gain)), c
    return dict(target=target, units=int(units.sum()), gain=None), None


def synthetic(d, p=31, freqs=(3, 7, 11, 14), gains=(1., .9, .8, .7), units=32, seed=34000):
    """Hand-built network, not trained. Each frequency gets `units` ReLUs with random
    phases and the same embedding for a and b; that group's W rows are a ridge fit
    of its own unit-amplitude target cos(2πk(a+b-c)/p) over all p² pairs. The groups
    contribute about equally to the function; their input energy ranks by `gains`."""
    rng = np.random.default_rng(seed)
    tokens, classes = np.arange(p), np.arange(p)
    u_blocks, w_blocks = [], []
    for k, g in zip(freqs, gains):
        u = g*np.cos(2*np.pi*k*tokens[:, None]/p + rng.uniform(0, 2*np.pi, units))
        h = np.maximum(u[d['a']] + u[d['b']], 0)
        target = np.cos(2*np.pi*k*((d['a'] + d['b'])[:, None] - classes)/p)
        w_blocks.append(np.linalg.solve(h.T@h + 1e-6*np.eye(units), h.T@target))
        u_blocks.append(u)
    U = np.hstack(u_blocks)
    return [U, U.copy(), np.vstack(w_blocks)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified(run_dir, name):
    path = run_dir/name
    manifest = json.loads((run_dir/'manifest.json').read_text())
    if sha256(path) != manifest['files'][name]['sha256']:
        raise ValueError(f'{path} does not match its manifest hash')
    return path


def load_weights(run_dir, step):
    ck = np.load(verified(run_dir, f'checkpoint_{step:06d}.npz'))
    return [ck['U'], ck['V'], ck['W']]


def load_run(run_dir, step):
    """Load a published run, verifying manifest hashes and the plan's split hash."""
    manifest = json.loads((run_dir/'manifest.json').read_text())
    if step is None:
        step = max(int(p.stem.split('_')[1]) for p in run_dir.glob('checkpoint_*.npz'))
    ck_path, data_path = run_dir/f'checkpoint_{step:06d}.npz', verified(run_dir, 'data.npz')
    for name in ('config.json', 'metrics.jsonl'):
        verified(run_dir, name)
    raw = np.load(data_path)
    d = {k: raw[k] for k in ('a', 'b', 'y', 'yadd', 'train', 'cal', 'audit')}
    # The e1 plan's split, unless research/plan-followups.json gives this run its own.
    expected = json.loads((ROOT/'research/plan.json').read_text())['common']['split_sha256']
    followups = json.loads((ROOT/'research/plan-followups.json').read_text())['runs']
    expected = next((r['split_sha256'] for r in followups if r['run_id'] == manifest['run_id'] and 'split_sha256' in r),
                    expected)
    split = array_sha256(d['a'], d['b'], d['yadd'], d['train'], d['cal'], d['audit'])
    if split != expected or array_sha256(d['y']) != manifest['data_hashes']['target_sha256']:
        raise ValueError('split or target hash differs from the plan/manifest')
    w = load_weights(run_dir, step)
    logged = next(json.loads(line) for line in (run_dir/'metrics.jsonl').read_text().splitlines()
                  if json.loads(line)['step'] == step)
    metrics, _ = evaluate(w, d)
    replay = max(abs(metrics[f'{s}_accuracy'] - logged[s]) for s in SPLITS)
    replay = max([replay] + [abs(metrics[f'{s}_{k}'] - v) for s in SPLITS for k, v in logged['objective'][s].items()])
    config = json.loads((run_dir/'config.json').read_text())
    meta = dict(run_id=manifest['run_id'], step=step, purpose=config.get('purpose'),
                checkpoint=str(ck_path), checkpoint_sha256=sha256(ck_path), data_sha256=sha256(data_path),
                manifest_hashes_verified=True, split_sha256=split, target_sha256=array_sha256(d['y']),
                producing_code_commit=manifest['producing_code_commit'],
                max_abs_diff_vs_logged_metrics=replay)
    return w, d, meta


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_retention(rows, rand, energy, reference, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(train='#d36b1f', audit='#2a7bb6')
    light = dict(train='#f2c9a5', audit='#b5d3ea')
    p = 2*len(energy) + 1
    fig, ax = plt.subplots(4, 2, figsize=(10, 14), constrained_layout=True)
    freqs = np.arange(1, len(energy) + 1)
    share = energy/energy.sum()
    order = np.argsort(-energy, kind='stable')
    ax[0, 0].bar(freqs, share, color='#777777', label='analysed checkpoint')
    if reference is not None:
        step, ref = reference
        ax[0, 0].plot(freqs, ref/ref.sum(), '_', color='k', ms=14, mew=2, label=f'same run, step {step}')
    for rank, i in enumerate(order[:5]):
        ax[0, 0].annotate(f'#{rank + 1}', (freqs[i], share[i]), ha='center', va='bottom', fontsize=8)
    ax[0, 0].axhline(1/len(energy), color='k', ls=':', lw=1, label=f'uniform share 1/{len(energy)}')
    ax[0, 0].set(xlabel='frequency k', ylabel='share of non-DC input energy', xticks=freqs, ylim=(0, 1.35*share.max()),
                 title='input-embedding energy by frequency (#n = rank)')
    ax[0, 0].legend(fontsize=7, loc='upper center', ncol=3)
    ks = np.arange(len(energy) + 1)
    ax[0, 1].plot(ks, np.r_[0, np.cumsum(share[order])], 'o-', color='#777777', ms=3)
    ax[0, 1].set(xlabel='K (top-K by energy)', ylabel='energy fraction in top K', ylim=(0, 1.02),
                 title='cumulative energy of the selected frequencies')
    for col, mode in enumerate(('retain', 'remove')):
        sel = [r for r in rows if r['mode'] == mode]
        rr = [r for r in rand if r['mode'] == mode]
        for row_i, (metric, label) in enumerate([('accuracy', 'accuracy'), ('cross_entropy', 'cross-entropy (nats, log scale)'),
                                                 ('margin_median', 'signed margin (logit units)')]):
            a = ax[row_i + 1, col]
            for split, dx in (('train', -.12), ('audit', .12)):
                a.scatter([r['K'] + dx for r in rr], [r[f'{split}_{metric}'] for r in rr], s=5, color=light[split],
                          zorder=1, label=f'{split}, random K-subsets (not energy-matched)')
            for split in ('train', 'audit'):
                a.plot([r['K'] for r in sel], [r[f'{split}_{metric}'] for r in sel], 'o-', ms=3, color=colors[split],
                       label=f'{split}, top-K by energy' + (' (median)' if metric == 'margin_median' else ''))
                if metric == 'margin_median':
                    a.plot([r['K'] for r in sel], [r[f'{split}_margin_p05'] for r in sel], 's--', ms=2, lw=1,
                           color=colors[split], label=f'{split}, 5th percentile')
            if metric == 'accuracy':
                a.axhline(1/p, color='k', ls=':', lw=1, label=f'chance 1/{p}')
                a.set_ylim(-0.02, 1.02)
            elif metric == 'cross_entropy':
                a.set_yscale('log')
                a.axhline(np.log(p), color='k', ls=':', lw=1, label=f'uniform prediction, ln {p}')
            else:
                a.axhline(0, color='k', lw=.6)
            verb = 'retained (DC kept)' if mode == 'retain' else 'removed'
            a.set(xlabel=f'K = top-K frequencies {verb}', ylabel=label, xticks=ks)
            if row_i == 0:
                a.set_title(f'{mode} top-K: K=0 is ' + ('DC only' if mode == 'retain' else 'the original model'))
    handles = {}
    for a in ax[1:].ravel():
        for h, lab in zip(*a.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle(textwrap.fill(title, 110), fontsize=10)  # narrow figure: wrap long run ids
    for ext in ('png', 'svg'):
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=130)
    plt.close(fig)


def plot_rescaling(w, wt, d, sweep, info, top, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    e0, e1 = spectrum(w).sum(1)[1:], spectrum(wt).sum(1)[1:]
    freqs = np.arange(1, len(e0) + 1)
    ax[0, 0].bar(freqs - .2, e0/e0.sum(), .4, color='#777777', label='checkpoint')
    ax[0, 0].bar(freqs + .2, e1/e1.sum(), .4, color='#5a3f9e', label=f'rescaled: k={info["target"]} units x{info["gain"]:.3g}')
    for e, dx, color in ((e0, -.2, '#777777'), (e1, .2, '#5a3f9e')):
        for i in np.argsort(-e, kind='stable')[:top]:
            ax[0, 0].annotate('*', (freqs[i] + dx, e[i]/e.sum()), ha='center', va='bottom', color=color)
    ax[0, 0].set(xlabel='frequency k', ylabel='share of non-DC input energy', xticks=freqs,
                 ylim=(0, 1.35*max(e0.max()/e0.sum(), e1.max()/e1.sum())), title=f'energy ranking; * marks each top {top}')
    ax[0, 0].legend(fontsize=8, loc='upper center', ncol=2)
    r0, r1 = curve(w, d, ranking(w), ('retain',)), curve(wt, d, ranking(wt), ('retain',))
    for a, metric, label in ((ax[0, 1], 'cross_entropy', 'cross-entropy (nats, log scale)'), (ax[0, 2], 'accuracy', 'accuracy')):
        for split, color in (('train', '#d36b1f'), ('audit', '#2a7bb6')):
            a.plot([r['K'] for r in r0], [r[f'{split}_{metric}'] for r in r0], 'o-', ms=3, color=color,
                   label=f'{split}, checkpoint ranking')
            a.plot([r['K'] for r in r1], [r[f'{split}_{metric}'] for r in r1], 'x--', ms=5, color=color,
                   label=f'{split}, ranking after rescaling')
        a.set(xlabel='K = top-K frequencies retained', ylabel=label, xticks=np.arange(len(e0) + 1),
              title='same function, retention under each ranking')
        if metric == 'cross_entropy':
            a.set_yscale('log')
        a.legend(fontsize=7)
    sig = np.array([s['sigma'] for s in sweep])
    jitter = np.random.default_rng(0).uniform(-.012, .012, len(sig))
    ax[1, 0].scatter(sig + jitter, [max(s['max_rel_logit_diff'], 1e-19) for s in sweep], s=10, color='#5a3f9e',
                     label='random rescaling draws')
    ax[1, 0].axhline(np.finfo(float).eps, color='k', ls=':', lw=1, label='float64 machine epsilon')
    ax[1, 0].set(xscale='log', yscale='log', ylim=(1e-19, 1), xlabel='rescaling log-SD sigma (c_j = exp(sigma z_j))',
                 ylabel='max |logit change| / max |logit|', title='function check, all 961 pairs (zeros drawn at 1e-19)',
                 xticks=SIGMAS, xticklabels=[str(s) for s in SIGMAS])
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].scatter(sig + jitter, [s['codec_bits'] for s in sweep], s=10, color='#5a3f9e', label='random rescaling')
    for bits, q in sorted({(s['codec_bits'], s['codec_q']) for s in sweep}):
        ax[1, 1].annotate(f'q={q}', (SIGMAS[-1]*1.15, bits), va='center', fontsize=8)
    ax[1, 1].axhline(info['codec_bits_checkpoint'], color='#777777', ls=':', label='checkpoint')
    ax[1, 1].set(xscale='log', xlim=(.07, 1.6), xlabel='rescaling log-SD sigma', ylabel='codec bits (pruning + quantization)',
                 title='compress() code length', xticks=SIGMAS, xticklabels=[str(s) for s in SIGMAS])
    ax[1, 1].legend(fontsize=8)
    ax[1, 2].scatter(sig + jitter, [s['kendall_tau'] for s in sweep], s=10, color='#5a3f9e', label='Kendall tau vs checkpoint')
    frac = [np.mean([s['topK_changed'] for s in sweep if s['sigma'] == x]) for x in SIGMAS]
    ax[1, 2].plot(SIGMAS, frac, 'o-', color='#c9a227', label=f'fraction of draws with a different top-{top} set')
    ax[1, 2].set(xscale='log', ylim=(-1.05, 1.05), xlabel='rescaling log-SD sigma', ylabel='value',
                 title='frequency ranking', xticks=SIGMAS, xticklabels=[str(s) for s in SIGMAS])
    ax[1, 2].legend(fontsize=8)
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
    ap.add_argument('--reference-step', type=int, help='Overlay this step\'s energy shares (e.g. 0, initialization)')
    ap.add_argument('--source-commit', help='Commit the published run was read from (recorded only)')
    ap.add_argument('--label', help='Checkpoint description for titles, e.g. "late, non-generalizing"')
    ap.add_argument('--top', type=int, default=4, help='Top-K set compared under rescaling')
    ap.add_argument('--draws', type=int, default=16, help='Random subsets per K and rescalings per sigma')
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
    label = args.label or meta['purpose']
    name = f"{meta['run_id']}" + (f" step {meta['step']}" if 'step' in meta else '')
    args.out.mkdir(parents=True, exist_ok=True)

    order = ranking(w)
    rows = curve(w, d, order)
    rand = random_subsets(w, d, args.draws, RANDOM_SEED)
    base, base_logits = evaluate(w, d)
    info, c = targeted(w, args.top)
    rng = np.random.default_rng(RESCALE_SEED)
    sweep = [dict(sigma=s, draw=i, **compare(w, rescale(w, np.exp(s*rng.standard_normal(w[0].shape[1]))), d,
                                              base_logits, args.top))
             for s in SIGMAS for i in range(args.draws)]
    info.update(codec_bits_checkpoint=codec(w, d)['codec_bits'])
    if c is not None:
        wt = rescale(w, c)
        info.update(**compare(w, wt, d, base_logits, args.top))
    energy = spectrum(w).sum(1)[1:]
    reference = None
    if args.run and args.reference_step is not None:
        reference = (args.reference_step, spectrum(load_weights(args.run, args.reference_step)).sum(1)[1:])
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    summary = dict(label=label, input=meta, checkpoint_metrics=base, ranking=order.tolist(),
                   energy_share=(energy/energy.sum()).tolist(),
                   reference_energy_share=None if reference is None else dict(
                       step=reference[0], share=(reference[1]/reference[1].sum()).tolist()),
                   codec_checkpoint=codec(w, d),
                   param_norm=float(np.sqrt(sum(np.sum(x*x) for x in w))), targeted_rescaling=info,
                   analysis_code_commit=git('rev-parse', 'HEAD'), analysis_tree_status=git('status', '--porcelain'),
                   analysis_script_sha256=sha256(__file__), numpy=np.__version__,
                   seeds=dict(random_subsets=RANDOM_SEED, rescalings=RESCALE_SEED),
                   command=' '.join(['python', *sys.argv]))
    (args.out/'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    write_csv(args.out/'retention.csv', rows)
    write_csv(args.out/'random_subsets.csv', rand)
    write_csv(args.out/'rescaling_sweep.csv', sweep)
    plot_retention(rows, rand, energy, reference, f'{name} ({label}): frequency retention/removal, checkpoint-local ranking',
                   args.out/'retention')
    if c is not None:
        plot_rescaling(w, wt, d, sweep, info, args.top,
                       f'{name} ({label}): positive hidden-unit rescaling, logits unchanged', args.out/'rescaling')
    print(json.dumps(dict(out=str(args.out), ranking=order.tolist(), targeted=info.get('gain'),
                          codec=summary['codec_checkpoint']), indent=None))


if __name__ == '__main__':
    main()
