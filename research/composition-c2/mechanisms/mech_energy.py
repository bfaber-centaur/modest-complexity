#!/usr/bin/env python3
"""Post-hoc descriptive check for the follow-up (ADAPTIVE.md): how much of the history residual
r = h_i - c_i(x) lies in the centroid span U versus the dimension-matched random subspaces of
c2-mech-residual-a1 (the same draws: same seed and order), and how far the split states leave the
ReLU state set (negative coordinates; natural states and centroids have none).

uv run --locked python research/composition-c2/mechanisms/mech_energy.py --out DIR
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

import mech_core as mc
import mech_residual as mr
import c2rnn
import c2task
import nets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    T = c2task.load(c2task.ORIGINAL)
    R = c2task.rows(T['P'], c2task.programs(3))
    held = np.isin(mc.lex(R['ops']), mc.lex(np.array(T['held'])))
    rows = []
    for ck in mc.checkpoints([10000]):
        p = mc.load_verified(ck)
        rng = np.random.default_rng([mr.SEED, list(mc.CONDITIONS).index(ck['condition']), ck['seed']])
        H = c2rnn.forward(p, R['ops'], s=R['s'])[0]
        for i in (2, 3):  # same call order as mech_residual.analyze, so the random draws match
            x = R['states'][:, i-1]
            c, U = mr.centroid_basis(H[i][~held], x[~held])
            Qs = mr.random_bases(rng, H[i].shape[1])
            r = H[i] - c[x]
            e = np.sum(r*r, 1)
            states = dict(natural=H[i], centroid=c[x], inside_U=c[x] + mr.proj(U, r), outside_U=c[x] + r - mr.proj(U, r))
            for name, m in (('held', held), ('all', np.ones_like(held))):
                frac_U = np.sum(mr.proj(U, r[m])**2, 1) / e[m]
                frac_Q = [np.mean(np.sum(mr.proj(Q, r[m])**2, 1) / e[m]) for Q in Qs]
                row = dict(condition=ck['condition'], seed=ck['seed'], run_id=ck['run_id'], i=i, rows=name,
                           residual_energy_in_U=float(frac_U.mean()), residual_energy_in_random_mean=float(np.mean(frac_Q)),
                           residual_energy_in_random_max=float(np.max(frac_Q)), expected_random=7/H[i].shape[1])
                for s, h in states.items():
                    row[f'neg_coord_frac_{s}'] = float(np.mean(h[m] < 0))
                rows.append(row)
        print(ck['run_id'], flush=True)
    with (args.out/'energy.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.out/'checks.json').write_text(json.dumps(dict(provenance=nets.provenance(args.out, mc.__file__, mr.__file__, __file__)), indent=1, default=str))


if __name__ == '__main__':
    main()
