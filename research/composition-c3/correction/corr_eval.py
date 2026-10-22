#!/usr/bin/env python3
"""Arms A-E on the six frozen bases, one split per run (QUESTION.md).

  --split main     fit on all 128 depth-2 prefixes; evaluate the 128 held-out-program rows (seen
                   prefix, new op 3) and the 384 training-program rows.
  --split grouped  fit on the 64 fit prefixes of the grouped split; report fit, val and eval
                   prefixes' held-out-program and training-program rows separately. The eval
                   prefixes are adapter-held-out prefixes, not globally unseen states.

Outputs accuracy.csv (final answer), states.csv (symbol and centroid error of the corrected h_2),
fits.csv (map and decoder fit) and checks.json (hashes before/after, published reproductions).

uv run --locked python research/composition-c3/correction/corr_eval.py --out DIR --split main --k K --lr LR [--bases f2]
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

import corr_core as cc
import c3correct
import nets


def published(c2, f2_bases=()):
    """A's held-out accuracy (c2 mechanisms eval, matched to r1) and c2's class-mean (dose 0) accuracy;
    for the follow-up bases, A's held-out accuracy from each run's logged step-10000 row."""
    out = {}
    for b in f2_bases:
        row = [json.loads(line) for line in (b['config'].parent/'metrics.jsonl').open()][-1]
        assert row['step'] == cc.STEP
        out[(b['run_id'], 'A', 'held')] = row['held']['final']['acc']
    for r in csv.DictReader((c2/'c2-mech-eval-a1/eval.csv').open()):
        if r['step'] == str(cc.STEP):
            out[(r['run_id'], 'A', 'held')] = float(r['held_final_acc'])
    for r in csv.DictReader((c2/'c2-mech-residual-a1/residual.csv').open()):
        if r['kind'] == 'dose' and r['i'] == '2' and r['L'] == '3' and float(r['param']) == 0. and r['subset'] in ('held', 'train'):
            out[(r['run_id'], 'E', r['subset'])] = float(r['accuracy'])
    return out


def write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--split', choices=('main', 'grouped'), required=True)
    ap.add_argument('--k', type=int, required=True)
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--bases', choices=('original', 'f2'), default='original')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    groups = ('fit',) if args.split == 'main' else ('fit', 'val', 'eval')
    bases = cc.bases(args.bases)
    ref = published(cc.REPO/'research/composition-c2/mechanisms/runs', bases if args.bases == 'f2' else ())
    acc_rows, state_rows, fit_rows, records, repro, prefixes = [], [], [], [], [], {}
    for b in bases:
        _, R, pre = cc.task(b['task_id'])  # the split follows each task's own prefix symbols
        sp = dict(fit=np.arange(len(pre['x']))) if args.split == 'main' else cc.split(pre)
        prefixes[b['task_id']] = {k: v.tolist() for k, v in sp.items()}
        p, rec = cc.load_base(b)
        key = dict(condition=b['condition'], task_id=b['task_id'], seed=b['seed'], run_id=b['run_id'], split=args.split)
        H, x = cc.states(p, pre), pre['x']
        mods, fits = cc.fit_arms(H[sp['fit']], x[sp['fit']], args.k, args.lr)
        rec['params_sha256_after_fit'] = c3correct.params_sha256(p)
        C, scale = mods['C'], cc.centroid_scale(mods['C'])
        for arm, f in fits.items():
            fit_rows.append(dict(key, arm=arm, **{k: f.get(k) for k in
                                 ('k', 'lr', 'seed', 'updates', 'loss_first', 'loss_last', 'dead_output_units', 'probe_seed', 'train_acc')}))
        for g in groups:
            idx = sp[g]
            in_g = np.isin(R['prefix'], idx)
            for arm in cc.ARMS:
                for subset, flag in (('held', True), ('train', False)):
                    sel = np.flatnonzero(in_g & (R['held'] == flag))
                    a = cc.accuracy(p, R, sel, cc.correction(arm, mods, R['states'][sel, cc.DEPTH-1]))
                    acc_rows.append(dict(key, arm=arm, prefixes=g, rows=subset, n=len(sel), accuracy=a))
                    want = ref.get((b['run_id'], arm, subset)) if args.split == 'main' else None
                    if want is not None:
                        repro.append(dict(run_id=b['run_id'], arm=arm, rows=subset, here=a, c2=want, equal=abs(a - want) < 1e-12))
                fn = cc.correction(arm, mods, x[idx])
                Hc = H[idx] if fn is None else fn(H[idx])
                state_rows.append(dict(key, arm=arm, prefixes=g, n=len(idx),
                                       symbol_acc=float(np.mean(cc.nearest(C, Hc) == x[idx])),
                                       centroid_err=float(np.mean(((Hc - C[x[idx]]) ** 2).sum(1))) / scale,
                                       decoder_acc=float(np.mean(cc.decode(mods['D'], H[idx]) == x[idx])) if arm == 'D' else ''))
        rec['params_sha256_after'] = c3correct.params_sha256(p)
        rec['hash_unchanged'] = rec['params_sha256'] == rec['params_sha256_after_fit'] == rec['params_sha256_after']
        assert rec['hash_unchanged'], b['run_id']
        records.append(rec)
        heldA = [r['accuracy'] for r in acc_rows if r['run_id'] == b['run_id'] and r['rows'] == 'held' and r['prefixes'] == groups[-1]]
        print(b['run_id'], args.split, groups[-1], 'held A-E', [round(v, 3) for v in heldA], flush=True)
    write(args.out/'accuracy.csv', acc_rows); write(args.out/'states.csv', state_rows); write(args.out/'fits.csv', fit_rows)
    (args.out/'checks.json').write_text(json.dumps(dict(
        split=args.split, bases_set=args.bases, k=args.k, lr=args.lr, map_seed=cc.MAP_SEED, split_seed=cc.SPLIT_SEED,
        probe_seed=cc.PROBE_SEED, updates=cc.UPDATES, prefixes_by_task=prefixes,
        reproductions=repro, all_reproduced=all(r['equal'] for r in repro) if repro else None, bases=records,
        provenance=nets.provenance(args.out, cc.__file__, __file__)), indent=1, default=str))


if __name__ == '__main__':
    main()
