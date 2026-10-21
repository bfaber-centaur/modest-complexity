#!/usr/bin/env python3
"""How far do the weights move between checkpoints of one run, and in which input
frequencies? Relative Frobenius change of U, V and W between consecutive steps, and
the frequency shares (freq.py definition) of the change to U and V.

uv run --locked --group analysis python research/mechanisms/drift.py \
    --run research/coordinator/runs/e1f1-co-add-w512-i30000-s200k-a1 --steps 50000 100000 200000 --out DIR
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freq import ROOT, load_run, sha256, spectrum  # noqa: E402


def drift(w0, w1):
    rel = {m: float(np.linalg.norm(b - a)/np.linalg.norm(a)) for m, a, b in zip('UVW', w0, w1)}
    energy = spectrum([b - a for a, b in zip(w0[:2], w1[:2])]).sum(1)[1:]
    share = energy/energy.sum()
    order = np.argsort(-share, kind='stable') + 1
    return dict(relative_change=rel, change_energy_share=share.tolist(), change_ranking=order.tolist(),
                change_top4_share=float(np.sort(share)[::-1][:4].sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--steps', type=int, nargs='+', required=True)
    ap.add_argument('--source-commit', help='Commit the published run was read from (recorded only)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    loaded = [load_run(args.run, s) for s in args.steps]
    for _, _, meta in loaded:
        meta['read_from_commit'] = args.source_commit
    pairs = [dict(start=a[2]['step'], end=b[2]['step'], **drift(a[0], b[0])) for a, b in zip(loaded, loaded[1:])]
    if len(loaded) > 2:
        pairs.append(dict(start=loaded[0][2]['step'], end=loaded[-1][2]['step'], **drift(loaded[0][0], loaded[-1][0])))
    git = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = dict(inputs=[m for _, _, m in loaded], param_norm={m['step']: float(np.sqrt(sum(np.sum(x**2) for x in w)))
                                                             for w, _, m in loaded},
               uniform_top4_share=4/15, drift=pairs, analysis_code_commit=git('rev-parse', 'HEAD'),
               analysis_tree_status=git('status', '--porcelain'), analysis_script_sha256=sha256(__file__),
               command=' '.join(['python', *sys.argv]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'drift.json').write_text(json.dumps(out, indent=1) + '\n')
    for p in pairs:
        print(p['start'], p['end'], {k: round(v, 4) for k, v in p['relative_change'].items()},
              'change top-4 share', round(p['change_top4_share'], 3), 'ranking', p['change_ranking'][:4])


if __name__ == '__main__':
    main()
