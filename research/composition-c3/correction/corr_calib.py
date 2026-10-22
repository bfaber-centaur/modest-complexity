#!/usr/bin/env python3
"""Training-only calibration of the map (QUESTION.md): grid k x lr on the grouped split's fit
prefixes, scored by arm B's validation squared error to c(x), averaged over the six bases.
Map init seed 900200 (D7 calibration seed), not the evaluated draw 71200. No final answers and
no eval prefixes are computed. Also times map updates on this host (D8).

uv run --locked python research/composition-c3/correction/corr_calib.py --out DIR
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np

import corr_core as cc
import c3correct
import nets

GRID_K, GRID_LR = (16, 32, 64), (1e-3, 1e-2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    _, _, pre = cc.task()
    sp = cc.split(pre)
    rows, records, seconds, updates = [], [], 0., 0
    for b in cc.bases():
        p, rec = cc.load_base(b)
        H = cc.states(p, pre)
        Hf, xf, Hv, xv = H[sp['fit']], pre['x'][sp['fit']], H[sp['val']], pre['x'][sp['val']]
        C = cc.centroids(Hf, xf)
        for k in GRID_K:
            for lr in GRID_LR:
                t0 = time.perf_counter()
                qB, fB = cc.fit_map(Hf, C[xf], k, lr, cc.CALIB_SEED)
                qC, fC = cc.fit_map(Hf, Hf, k, lr, cc.CALIB_SEED)
                seconds += time.perf_counter() - t0; updates += 2*cc.UPDATES
                rows.append(dict(condition=b['condition'], seed=b['seed'], run_id=b['run_id'], k=k, lr=lr,
                                 B_fit_mse=fB['loss_last'], B_val_mse=cc.map_mse(qB, Hv, C[xv]),
                                 B_dead_output_units=fB['dead_output_units'],
                                 C_fit_mse=fC['loss_last'], C_val_mse=cc.map_mse(qC, Hv, Hv),
                                 C_dead_output_units=fC['dead_output_units'],
                                 val_identity_mse=float(np.mean((Hv - C[xv]) ** 2))))
        rec['params_sha256_after'] = c3correct.params_sha256(p)
        assert rec['params_sha256_after'] == rec['params_sha256'], rec['run_id']
        records.append(rec)
        print(b['run_id'], 'calibrated', flush=True)
    with (args.out/'calib.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Selection rule: smallest mean B validation error; cells within 1% of it tie, and a tie goes
    # to the smaller k, then the smaller lr.
    score = {(k, lr): float(np.mean([r['B_val_mse'] for r in rows if r['k'] == k and r['lr'] == lr]))
             for k in GRID_K for lr in GRID_LR}
    best = min(score.values())
    chosen = min((c for c, v in score.items() if v <= 1.01*best), key=lambda c: (c[0], c[1]))
    ms = 1000*seconds/updates
    for c, v in sorted(score.items()):
        print(f'k={c[0]:3d} lr={c[1]:g} mean B val mse {v:.5f}', '<- chosen' if c == chosen else '')
    print(f'map update: {ms:.3f} ms on this host ({updates} updates)')
    (args.out/'checks.json').write_text(json.dumps(dict(
        grid_k=GRID_K, grid_lr=GRID_LR, init_seed=cc.CALIB_SEED, split_seed=cc.SPLIT_SEED, updates=cc.UPDATES,
        split={k: v.tolist() for k, v in sp.items()},
        mean_B_val_mse={f'k={k},lr={lr}': v for (k, lr), v in score.items()},
        chosen=dict(k=chosen[0], lr=chosen[1]), ms_per_map_update=ms, bases=records,
        provenance=nets.provenance(args.out, cc.__file__, __file__)), indent=1, default=str))


if __name__ == '__main__':
    main()
