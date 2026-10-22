#!/usr/bin/env python3
"""Decoder-across-position table (DECISIONS D5, c2probe.decode_table unchanged, probe seed
64300) and the readout-across-position table, for every cell and seed at the given steps.

Readout table: the model's own readout copy k (k = 1..3; one shared readout in tied-readout
cells) applied to h_j (j = 1..3), accuracy against the true symbol at j, on training-program
and held-out-program rows. Programs are length 3 from the start state, so every untied copy is
used at its own step (no step map).

Writes out/decode_table.csv and out/readout_table.csv (one row per entry).
uv run --locked python research/composition-c2/sharing/share_decode.py --steps 0 10000 [--cells TT TU]
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import share_runs as S  # noqa: E402
import c2probe  # noqa: E402
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import nets  # noqa: E402

PROBE_SEED = 64300


def readout_table(p, T):
    rows = []
    for split, progs in (('train', T['train']), ('held', T['held'])):
        d = c2task.rows(T['P'], progs)
        H = c2rnn.forward(p, d['ops'], s=d['s'])[0]
        for k in (1, 2, 3):
            V, bv = c2rnn._keys(p, c2rnn.READOUT, k)
            for j in (1, 2, 3):
                acc = float(np.mean((H[j] @ p[V] + p[bv]).argmax(1) == d['states'][:, j-1]))
                rows.append(dict(split=split, readout=k, position=j, acc=acc))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', type=int, nargs='+', default=[0, 10000])
    ap.add_argument('--cells', nargs='+', default=list(S.CELLS))
    args = ap.parse_args()
    T, _ = S.task()
    out = Path(__file__).resolve().parent/'out'; out.mkdir(exist_ok=True)
    dec, ro = [], []
    for cell in args.cells:
        for seed in S.SEEDS:
            for step in args.steps:
                path = S.checkpoint(cell, seed, step)
                p = nets.load(path)[0]
                assert c2rnn.scheme(p) == dict(transition=S.TIE[cell][0] == 'tied', readout=S.TIE[cell][1] == 'tied')
                key = dict(cell=cell, seed=seed, step=step, run_id=S.run_id(cell, seed), ckpt_sha256=nets.sha256_file(path))
                tab = c2probe.decode_table(p, T, PROBE_SEED)
                for i in tab['positions']:
                    for j in tab['positions']:
                        dec.append(dict(key, probe_seed=PROBE_SEED, fit_position=i, score_position=j,
                                        train_fit=tab['train_fit'][i], held_acc=tab['table'][i][j]))
                ro += [dict(key, **r) for r in readout_table(p, T)]
                print(cell, seed, step, np.round(np.array(tab['table']), 2).tolist())
    for name, rows in (('decode_table.csv', dec), ('readout_table.csv', ro)):
        with (out/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
