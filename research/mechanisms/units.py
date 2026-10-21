#!/usr/bin/env python3
"""Per-unit frequency concentration across the published checkpoints of several runs.

The aggregate spectrum (freq.py) sums energy over hidden units, so units that each
use one frequency but together cover all of them look flat. Per live unit j:
- input concentration: largest share of u_j and v_j's non-DC energy (rfft bins 1..15,
  token axis) in one frequency;
- output concentration: the same for w_j along the output-label axis;
- agreement: whether the unit's dominant input and output frequencies coincide.
Shares are ratios within a unit, so positive unit rescaling leaves them unchanged.
Dominant frequencies are read from each checkpoint itself (checkpoint-local).

uv run --locked --group analysis python research/mechanisms/units.py --out research/mechanisms/figures/NAME \
    --runs research/coordinator/runs/e1f2-co-add-w128-i30000-tf50-a1 ...
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freq import ROOT, evaluate, load_run, sha256, write_csv  # noqa: E402
from gauge import live_units  # noqa: E402


def unit_shares(w):
    """Per-unit frequency shares, input (U and V) and output (W), shape (15, hidden)."""
    e_in = sum(np.abs(np.fft.rfft(x, axis=0))**2 for x in w[:2])[1:]
    e_out = (np.abs(np.fft.rfft(w[2], axis=1))**2)[:, 1:].T
    return e_in/e_in.sum(0), e_out/e_out.sum(0)


def concentration(w):
    live = live_units(w)
    s_in, s_out = (s[:, live] for s in unit_shares(w))
    dom = s_in.argmax(0) + 1
    counts = np.bincount(dom, minlength=s_in.shape[0] + 1)[1:]
    return dict(live_units=int(live.sum()), input_concentration=float(s_in.max(0).mean()),
                output_concentration=float(s_out.max(0).mean()),
                in_out_agreement=float(np.mean(dom == s_out.argmax(0) + 1)),
                frequencies_used=int((counts > 0).sum()), units_per_frequency=' '.join(map(str, counts)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--source-commit', help='Commit the published runs were read from (recorded only)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rows, inputs = [], []
    for run in args.runs:
        manifest = json.loads((run/'manifest.json').read_text())
        steps = sorted(int(n.split('_')[1][:6]) for n, f in manifest['files'].items()
                       if n.startswith('checkpoint_') and f['published'])
        for step in steps:
            w, d, meta = load_run(run, step)
            metrics, _ = evaluate(w, d)
            inputs.append(dict(run_id=meta['run_id'], step=step, checkpoint_sha256=meta['checkpoint_sha256'],
                               split_sha256=meta['split_sha256'], replay=meta['max_abs_diff_vs_logged_metrics']))
            rows.append(dict(run_id=meta['run_id'], step=step, train_pairs=len(d['train']),
                             train_accuracy=metrics['train_accuracy'], audit_accuracy=metrics['audit_accuracy'],
                             **concentration(w)))
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'units.csv', rows)
    (args.out/'units_summary.json').write_text(json.dumps(dict(
        inputs=inputs, read_from_commit=args.source_commit, analysis_code_commit=git('rev-parse', 'HEAD'),
        analysis_tree_status=git('status', '--porcelain'), analysis_script_sha256=sha256(__file__),
        command=' '.join(['python', *sys.argv])), indent=1) + '\n')
    plot(rows, args.out/'units')
    for r in rows:
        if r['step'] == max(x['step'] for x in rows if x['run_id'] == r['run_id']):
            print(r)


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = sorted({r['run_id'] for r in rows})
    metrics = [('audit_accuracy', 'held-out audit accuracy'),
               ('input_concentration', 'mean per-unit input concentration\n(share in the dominant frequency)'),
               ('output_concentration', 'mean per-unit output (W) concentration'),
               ('in_out_agreement', 'fraction of units whose dominant input\nand output frequencies agree')]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True, sharex=True)
    for a, (key, label) in zip(ax.ravel(), metrics):
        for run in runs:
            sel = sorted((r for r in rows if r['run_id'] == run), key=lambda r: r['step'])
            share = sel[0]['train_pairs']
            color = '#1f6fb4' if share > 400 else '#c0392b'
            ls = {'0': '-', '1': '--', '2': ':'}[run.split('-i3000')[1][0]]
            a.plot([max(r['step'], 100) for r in sel], [r[key] for r in sel], color=color, ls=ls, marker='o', ms=3,
                   label=f"{run} ({share} train pairs)")
        a.set(xscale='log', ylabel=label, ylim=(0, 1.02), xlabel='update step (step 0 drawn at 100)')
    ax[0, 0].axhline(1/31, color='k', ls=':', lw=1, label='chance 1/31')
    ax[0, 1].axhline(1/15, color='k', ls=':', lw=1, label='uniform over 15 frequencies')
    handles = {}
    for a in ax.ravel():
        for h, lab in zip(*a.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=2, fontsize=8)
    fig.suptitle('Width 128: per-unit frequency structure, 50% training share (blue) vs e1 30% share (red); '
                 'line style = init seed 30000/30001/30002', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(path.with_suffix(f'.{ext}'), dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
