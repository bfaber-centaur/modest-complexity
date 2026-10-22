#!/usr/bin/env python3
"""M1: ordinary evaluation of every recovered checkpoint, checked against r1's published records.

For each of the 27 recovered checkpoints (3 conditions x 3 seeds x steps 0/1000/10000):
verify the file's SHA-256 against RECOVERY.json, recompute r1's metrics line with c2train.evaluate
(c2rnn forward pass) and require exact equality with the published metrics.jsonl line at that step:
per-head train/held-out accuracy and cross-entropy, and the 128-character held_final_pred. The
labels are first checked against an independent loop over r1's execute().

uv run --locked python research/composition-c2/mechanisms/mech_eval.py --out DIR
"""
import argparse
import csv
import json
from pathlib import Path

import mech_core as mc
import c2task
import c2train
import nets


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    T = c2task.load(c2task.ORIGINAL)
    data = dict(train=c2task.rows(T['P'], T['train']), held=c2task.rows(T['P'], T['held']))
    for name, d in data.items():  # oracle: r1.execute row by row
        loop = [c2task.r1.execute(T['P'], int(s), tuple(map(int, q))) for s, q in zip(d['s'], d['ops'])]
        assert (d['states'] == loop).all(), f'{name} labels differ from r1.execute'
    nbr, _ = c2task.neighbour_answerable(T['P'], T['train'], T['held'])
    assert int((~nbr).sum()) == 42

    rows, mismatches = [], 0
    for ck in mc.checkpoints():
        p = mc.load_verified(ck)
        got, _ = c2train.evaluate(p, data, nbr)
        published = [json.loads(line) for line in ck['metrics'].read_text().splitlines()]
        pub = next(r for r in published if r['step'] == ck['step'])
        same = all(got[k] == pub[k] for k in ('train', 'held', 'held_final_pred'))
        mismatches += not same
        rows.append(dict(condition=ck['condition'], seed=ck['seed'], run_id=ck['run_id'], step=ck['step'],
                         ckpt_sha256=ck['sha256'], matches_published=same,
                         **{f'{split}_{h}_acc': got[split][h]['acc'] for split in ('train', 'held') for h in c2train.R1_HEADS},
                         held_N_acc=got['held_subset']['N'], held_notN_acc=got['held_subset']['notN'],
                         held_final_pred=got['held_final_pred']))
        print(ck['run_id'], ck['step'], 'match' if same else 'MISMATCH', f"held final {got['held']['final']['acc']:.3f}", flush=True)
    with (args.out/'eval.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.out/'checks.json').write_text(json.dumps(dict(
        checkpoints=len(rows), mismatches=mismatches, labels_equal_r1_execute=True, n_notN=42,
        provenance=nets.provenance(args.out, mc.__file__, __file__)), indent=1, default=str))
    if mismatches:
        raise SystemExit(f'{mismatches} checkpoints do not reproduce their published metrics')


if __name__ == '__main__':
    main()
