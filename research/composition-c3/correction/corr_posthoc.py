#!/usr/bin/env python3
"""Post-hoc reference (ADAPTIVE.md, requested by the coordinator after the first figure; no fitting):
the per-step-readout bases' own step-2 readout V2 applied to h_2 of the grouped split's fit, val
and eval prefixes. Every one of these prefixes is a training state of the base (D2), and V2 was
trained on all of them, so this is not a held-out decoder; it shows only that the frozen h_2 of
these prefixes is linearly read by a readout trained on all 128.

uv run --locked python research/composition-c3/correction/corr_posthoc.py --out DIR
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

import corr_core as cc
import c3correct
import nets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    _, _, pre = cc.task()
    sp = cc.split(pre)
    rows, records = [], []
    for b in cc.bases():
        if b['condition'] != 'per-step readout':
            continue
        p, rec = cc.load_base(b)
        H = cc.states(p, pre)
        pred = (H @ p['V2'] + p['bv2']).argmax(1)
        for g, idx in (('all', np.arange(len(H))), *sp.items()):
            rows.append(dict(condition=b['condition'], seed=b['seed'], run_id=b['run_id'], readout='V2', prefixes=g,
                             n=len(idx), symbol_acc=float(np.mean(pred[idx] == pre['x'][idx]))))
        rec['params_sha256_after'] = c3correct.params_sha256(p)
        assert rec['params_sha256_after'] == rec['params_sha256']
        records.append(rec)
        print(b['run_id'], [(r['prefixes'], r['symbol_acc']) for r in rows if r['run_id'] == b['run_id']], flush=True)
    with (args.out/'v2.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.out/'checks.json').write_text(json.dumps(dict(post_hoc=True, split_seed=cc.SPLIT_SEED, bases=records,
                                                        provenance=nets.provenance(args.out, cc.__file__, __file__)), indent=1, default=str))


if __name__ == '__main__':
    main()
