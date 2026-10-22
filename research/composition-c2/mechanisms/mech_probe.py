#!/usr/bin/env python3
"""D5 decoder-across-position table (c2probe.decode_table, unchanged, probe seed 64300) for the
nine recovered checkpoints at 10,000 updates. Descriptive: probes are fit on training-program h_i
and scored on held-out-program h_j; held-out h_0..h_2 are training prefixes (r1 C1.1).

uv run --locked python research/composition-c2/mechanisms/mech_probe.py --out DIR
"""
import argparse
import csv
import json
from pathlib import Path

import mech_core as mc
import c2probe
import c2task
import nets

PROBE_SEED = 64300


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, nargs='+', default=[10000])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    T = c2task.load(c2task.ORIGINAL)
    rows = []
    for ck in mc.checkpoints(args.steps):
        d = c2probe.decode_table(mc.load_verified(ck), T, PROBE_SEED)
        for i in d['positions']:
            for j in d['positions']:
                rows.append(dict(condition=ck['condition'], seed=ck['seed'], run_id=ck['run_id'], step=ck['step'],
                                 fit_position=i, score_position=j, train_fit=d['train_fit'][i], heldout_acc=d['table'][i][j]))
        print(ck['run_id'], ck['step'], [round(x, 2) for x in d['table'][3]], flush=True)
    with (args.out/'decode_table.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.out/'checks.json').write_text(json.dumps(dict(probe_seed=PROBE_SEED, provenance=nets.provenance(args.out, mc.__file__, __file__)), indent=1, default=str))


if __name__ == '__main__':
    main()
