#!/usr/bin/env python3
"""Is the held-out preference for the true label among "absent" labels carried by
particular input frequencies, or spread like the training fit?

Readout: the audit's statistic (research/audit/heldout_rank.py), reimplemented. For
each held-out pair (calibration + audit splits), labels that never co-occur with
either operand in any training pair are "absent"; the true label is always one of
them. Report the true label's mean normalized rank among the other absent labels
(0 = above all of them, 0.5 = no preference) and the fraction of pairs where it
tops them.

Interventions on analysis copies of one checkpoint: top-K retention and removal
(freq.py definitions), uniform random K-subsets, and for removal a norm-matched
null. The null subtracts the removed perturbation Δ (U's and V's top-K components)
after a random orthogonal map of token space that fixes the constant vector, the
same map for U and V: same norm, rank and singular values as Δ, random directions.
Unlike the projection, it adds energy on average instead of removing it.

uv run --locked --group analysis python research/mechanisms/relational.py \
    --run research/coordinator/runs/e1-co-add-w512-i30000-a1 --out research/mechanisms/figures/NAME
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freq import RANDOM_SEED, ROOT, evaluate, filter_inputs, load_run, ranking, sha256, spectrum, write_csv  # noqa: E402

NULL_SEED = 33300


def absent_labels(d):
    """Held-out indices and absent[i, c]: label c never co-occurs with either operand
    of held-out pair i in any training pair."""
    p = int(d['yadd'].max()) + 1
    present = np.zeros((p, p), bool)
    tr = d['train']
    present[d['a'][tr], d['yadd'][tr]] = True
    present[d['b'][tr], d['yadd'][tr]] = True
    held = np.concatenate([d['cal'], d['audit']])
    absent = ~(present[d['a'][held]] | present[d['b'][held]])
    if not absent[np.arange(len(held)), d['yadd'][held]].all():
        raise ValueError('a held-out true label co-occurs with an operand in training')
    return held, absent


def relational(logits, d, held, absent):
    rows = np.arange(len(held))
    x, y = logits[held], d['yadd'][held]
    others = absent.copy()
    others[rows, y] = False
    above = ((x > x[rows, y][:, None]) & others).sum(1)
    return dict(true_rank_among_absent=float(np.mean(above/others.sum(1))),
                true_top_among_absent=float(np.mean(above == 0)),
                pred_is_present=float(np.mean(~absent[rows, x.argmax(1)])))


def rotation(p, rng):
    """Random orthogonal p x p map that fixes the constant vector."""
    basis = np.linalg.qr(np.eye(p) - 1/p)[0][:, :p - 1]  # orthonormal basis of zero-mean vectors
    q, r = np.linalg.qr(rng.standard_normal((p - 1, p - 1)))
    q *= np.sign(np.diag(r))
    return basis @ q @ basis.T + np.full((p, p), 1/p)


def readout(w, d, held, absent):
    metrics, logits = evaluate(w, d)
    return dict(train_accuracy=metrics['train_accuracy'], **relational(logits, d, held, absent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--step', type=int)
    ap.add_argument('--source-commit', help='Commit the published run was read from (recorded only)')
    ap.add_argument('--label', help='Checkpoint description for titles')
    ap.add_argument('--draws', type=int, default=16, help='Random subsets and null draws per K')
    ap.add_argument('--audit-results', type=Path, default=ROOT/'research/audit/results/preview-1-heldout-rank.jsonl')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    w, d, meta = load_run(args.run, args.step)
    meta['read_from_commit'] = args.source_commit
    held, absent = absent_labels(d)
    p = w[0].shape[0]
    order = ranking(w)
    energy = spectrum(w).sum(1)[1:]
    freqs = np.arange(1, len(energy) + 1)
    rows = []
    for mode in ('retain', 'remove'):
        for k in range(len(order) + 1):
            rows.append(dict(kind='top-K', mode=mode, K=k, draw=0, frequencies=' '.join(map(str, order[:k])),
                             **readout(filter_inputs(w, order[:k], mode == 'retain'), d, held, absent)))
    rng = np.random.default_rng(RANDOM_SEED)
    for mode in ('retain', 'remove'):
        for k in range(1, len(freqs)):
            for i in range(args.draws):
                chosen = np.sort(rng.choice(freqs, k, replace=False))
                rows.append(dict(kind='random subset', mode=mode, K=k, draw=i, frequencies=' '.join(map(str, chosen)),
                                 **readout(filter_inputs(w, chosen, mode == 'retain'), d, held, absent)))
    rng = np.random.default_rng(NULL_SEED)
    for k in range(1, len(freqs) + 1):
        removed = filter_inputs(w, order[:k], False)
        delta = [w[0] - removed[0], w[1] - removed[1]]
        for i in range(args.draws):
            r = rotation(p, rng)
            rows.append(dict(kind='norm-matched null', mode='remove', K=k, draw=i, frequencies='',
                             **readout([w[0] - r @ delta[0], w[1] - r @ delta[1], w[2]], d, held, absent)))
    base = readout(w, d, held, absent)
    audit = None
    if args.audit_results.exists():
        for line in args.audit_results.read_text().splitlines():
            r = json.loads(line)
            if r['run'] == meta['run_id']:
                audit = next((x for x in r['rows'] if x['step'] == meta['step']), None)
    chance_top = float(np.mean(1/absent.sum(1)))
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    summary = dict(label=args.label or meta['purpose'], input=meta, heldout_pairs=len(held),
                   chance_top_among_absent=chance_top, checkpoint=base, audit_lane_values=audit,
                   ranking=order.tolist(), seeds=dict(random_subsets=RANDOM_SEED, null=NULL_SEED),
                   analysis_code_commit=git('rev-parse', 'HEAD'), analysis_tree_status=git('status', '--porcelain'),
                   analysis_script_sha256=sha256(__file__), command=' '.join(['python', *sys.argv]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'relational_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    write_csv(args.out/'relational.csv', rows)
    name = meta['run_id'] + f" step {meta['step']}"
    plot(rows, base, chance_top, len(held), f"{name} ({summary['label']}): held-out true-label preference among absent labels",
         args.out/'relational')
    print(json.dumps(dict(checkpoint=base, audit=audit, chance_top=chance_top)))


def plot(rows, base, chance_top, n_held, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {'top-K': dict(color='#5a3f9e', marker='o', ms=3, ls='-', zorder=3),
             'random subset': dict(color='#c9a227', marker='.', ms=4, ls='none', alpha=.6, zorder=1),
             'norm-matched null': dict(color='#3a9a5b', marker='x', ms=4, ls='none', alpha=.6, zorder=2)}
    offset = {'top-K': 0, 'random subset': -.15, 'norm-matched null': .15}
    metrics = [('true_rank_among_absent', 'true label: mean normalized rank\namong absent labels (0.5 = no preference)'),
               ('true_top_among_absent', 'fraction of held-out pairs where the true\nlabel tops all absent labels'),
               ('train_accuracy', 'train accuracy (context)')]
    fig, ax = plt.subplots(3, 2, figsize=(11, 11), constrained_layout=True, sharey='row')
    for col, mode in enumerate(('retain', 'remove')):
        for row, (key, label) in enumerate(metrics):
            a = ax[row, col]
            for kind, s in style.items():
                sel = [r for r in rows if r['kind'] == kind and r['mode'] == mode]
                if sel:
                    a.plot([r['K'] + offset[kind] for r in sel], [r[key] for r in sel], label=kind, **s)
            a.axhline(base[key], color='k', lw=.8, ls='--', label='unmodified checkpoint')
            if key == 'true_rank_among_absent':
                a.axhline(.5, color='k', ls=':', lw=1, label='no preference (0.5)')
            elif key == 'true_top_among_absent':
                a.axhline(chance_top, color='k', ls=':', lw=1, label=f'no preference ({chance_top:.3f})')
            verb = 'retained (DC kept)' if mode == 'retain' else 'removed'
            a.set(xlabel=f'K = top-K frequencies {verb}', xticks=range(16), ylabel=label if col == 0 else None)
            if row == 0:
                a.set_title(f'{mode} top-K (held-out = calibration + audit, {n_held} pairs)')
    handles = {}
    for a in ax.ravel():
        for h, lab in zip(*a.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle(title, fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
