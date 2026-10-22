#!/usr/bin/env python3
"""M2-M4 (QUESTION.md): whole-hidden-state interchange, semantic-centroid patches, random and
magnitude-matched controls, and descriptive geometry, on every recovered checkpoint.

For donor depth i and recipient depth j (0..3), every (donor prefix, suffix) combination is run
once; the 262,144 ordered (base row, donor row) pairs of length-3 rows per cell are scored by
indexing those combinations (the patched answer depends on the base only through its suffix).

Outputs (one row per checkpoint and cell unless noted):
  cells.csv     S (all pairs), consistency C, centroid S_cent, control target-hits, readout variants
  pairs.csv     S and base-retained by pair subset/category, donor history and recombined program
  geometry.csv  history spread per depth and cross-depth centroid offsets (descriptive)
  checks.json   self-swap, patched-vs-recombined equivalence and diagonal-constancy checks

uv run --locked python research/composition-c2/mechanisms/mech_interchange.py --out DIR [--steps 0 1000 10000]
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

import mech_core as mc
import c2rnn
import c2task
import nets

CONTROL_SEED = 64301
DEPTHS = range(mc.LENGTH + 1)


def sym_at(R, depth):
    return R['s'] if depth == 0 else R['states'][:, depth-1]


def analyze(p, T, rng):
    P = T['P']
    R = c2task.rows(P, c2task.programs(3))
    held = np.isin(mc.lex(R['ops']), mc.lex(np.array(T['held'])))
    pred0 = c2rnn.forward(p, R['ops'], s=R['s'])[1][-1].argmax(1)
    correct = pred0 == R['states'][:, -1]
    n = len(R['s'])
    base, donor = np.divmod(np.arange(n*n), n)  # all ordered (base row, donor row) pairs
    untied_readout = 'V' not in p

    pid = {i: mc.lex(R['ops'][:, :i]) * mc.N_SYM + R['s'] for i in DEPTHS}
    sid = {j: mc.lex(R['ops'][:, j:]) for j in DEPTHS}
    pre = {i: mc.prefixes(i) for i in DEPTHS}
    H = {i: mc.state(p, pre[i]['s'], pre[i]['ops']) for i in DEPTHS}
    psym = {i: mc.apply_ops(P, pre[i]['s'], pre[i]['ops']) for i in DEPTHS}
    train_rows = ~held
    Htrain = {i: H[i][pid[i][train_rows]] for i in DEPTHS}  # row-weighted training states
    ytrain = {i: sym_at(R, i)[train_rows] for i in DEPTHS}
    cent = {i: np.stack([Htrain[i][ytrain[i] == x].mean(0) for x in range(mc.N_SYM)]) for i in DEPTHS}
    held_prefix3 = held  # prefixes(3) is the row order

    cells, pairs, checks = [], [], dict(self_swap_state_bitwise=True, self_swap_pred_equal=True, recombined_max_abs_diff=0.,
                                        recombined_bitwise=True, diagonal_pred_identical=True)
    by_length = {}
    for i in DEPTHS:
        for j in DEPTHS:
            L = i + mc.LENGTH - j
            sfx = mc.suffixes(mc.LENGTH - j)
            nd, ns = len(H[i]), len(sfx)
            h = np.repeat(H[i], ns, axis=0)
            ops = np.tile(sfx, (nd, 1))
            dsym = np.repeat(psym[i], ns)
            target = mc.apply_ops(P, dsym, ops)
            hf = mc.run_suffix(p, h, ops, j)
            pred = mc.readout(p, hf).argmax(1)
            succ = pred == target

            # equivalence: the patched final state equals the plain forward pass on prefix ++ suffix
            ref = mc.recombined_state(p, np.repeat(pre[i]['s'], ns), np.repeat(pre[i]['ops'], ns, axis=0), ops)
            diff = float(np.abs(ref - hf).max())
            checks['recombined_max_abs_diff'] = max(checks['recombined_max_abs_diff'], diff)
            checks['recombined_bitwise'] &= bool(np.array_equal(ref, hf))
            # diagonal constancy: same (start, program) multiset for equal i - j
            key = np.concatenate([np.repeat(pre[i]['s'], ns)[:, None], np.repeat(pre[i]['ops'], ns, axis=0), ops], 1)
            order = np.lexsort(key.T[::-1])
            if L in by_length:
                k0, p0 = by_length[L]
                checks['diagonal_pred_identical'] &= bool(np.array_equal(k0, key[order]) and np.array_equal(p0, pred[order]))
            else:
                by_length[L] = (key[order], pred[order])

            # controls (R1 coordinate permutation of the donor state, R2 per-unit marginal resample)
            perm = rng.random(h.shape).argsort(1)
            h_perm = np.take_along_axis(h, perm, 1)
            h_marg = Htrain[i][rng.integers(len(Htrain[i]), size=h.shape), np.arange(h.shape[1])]
            hit_perm = mc.readout(p, mc.run_suffix(p, h_perm, ops, j)).argmax(1) == target
            hit_marg = mc.readout(p, mc.run_suffix(p, h_marg, ops, j)).argmax(1) == target

            # semantic centroids c_i(x), fit on training rows
            hc = np.repeat(cent[i], ns, axis=0)
            ops_c = np.tile(sfx, (mc.N_SYM, 1))
            target_c = mc.apply_ops(P, np.repeat(np.arange(mc.N_SYM), ns), ops_c)
            succ_cent = mc.readout(p, mc.run_suffix(p, hc, ops_c, j)).argmax(1) == target_c

            # consistency across histories with the same symbol at depth i
            pm = pred.reshape(nd, ns)
            C = mc.agreement(pm, psym[i]) if i > 0 else float('nan')
            cross = float('nan')
            if i == 3:
                cross = mc.cross_agreement(pm[held_prefix3], psym[i][held_prefix3], pm[~held_prefix3], psym[i][~held_prefix3])

            row = dict(i=i, j=j, L=L, S=float(succ.mean()), C=C, C_held_vs_train=cross,
                       S_cent=float(succ_cent.mean()), hit_perm=float(hit_perm.mean()), hit_marg=float(hit_marg.mean()))
            for k in (1, 2):  # per-step readout only: the same patched states read with V1, V2 (descriptive)
                row[f'S_V{k}'] = float((mc.readout(p, hf, k).argmax(1) == target).mean()) if untied_readout else float('nan')
            cells.append(row)

            # pair-level view
            kk = pid[i][donor] * ns + sid[j][base]
            ok, pr = succ[kk], pred[kk]
            retained = pr == R['states'][base, -1]
            same_state = sym_at(R, i)[donor] == sym_at(R, j)[base]
            same_prefix = (pid[i][donor] == pid[j][base]) if i == j else np.zeros_like(same_state)
            selfp = (donor == base) if i == j else np.zeros_like(same_state)
            if i == j:  # self-swap of every row: unchanged final state and prediction
                checks['self_swap_state_bitwise'] &= bool(np.array_equal(hf[pid[i]*ns + sid[j]], H[3]))
                checks['self_swap_pred_equal'] &= bool(np.array_equal(pr[selfp], pred0[base[selfp]]))
            both = correct[base] & correct[donor]
            cats = dict(all=np.ones_like(ok), self=selfp, same_prefix=same_prefix,
                        same_state_diff_history=same_state & ~same_prefix, diff_state=~same_state)

            def add(subset, category, mask, donor_hist='any', recombined='any'):
                m = mask.sum()
                pairs.append(dict(i=i, j=j, L=L, subset=subset, category=category, donor_hist=donor_hist,
                                  recombined=recombined, n_pairs=int(m),
                                  success=float(ok[mask].mean()) if m else float('nan'),
                                  base_retained=float(retained[mask].mean()) if m else float('nan'),
                                  hit_perm=float(hit_perm[kk][mask].mean()) if m else float('nan'),
                                  hit_marg=float(hit_marg[kk][mask].mean()) if m else float('nan')))
            for subset, smask in (('all', np.ones_like(ok)), ('both_correct', both)):
                for cname, cmask in cats.items():
                    add(subset, cname, smask & cmask)
            if i == 3:
                for name, hmask in (('train', ~held[donor]), ('held', held[donor])):
                    add('all', 'all', hmask, donor_hist=name)
                    add('all', 'same_state_diff_history', hmask & same_state & ~same_prefix, donor_hist=name)
            if i == j:
                rec_held = np.isin(mc.lex(np.concatenate([R['ops'][donor, :i], R['ops'][base, j:]], 1)), mc.lex(np.array(T['held'])))
                for name, rmask in (('train', ~rec_held), ('held', rec_held)):
                    add('all', 'all', rmask, recombined=name)

    geometry = []
    for i in DEPTHS:
        c, Ht, y = cent[i], Htrain[i], ytrain[i]
        between = np.mean([np.sum((c[a] - c[b])**2) for a in range(mc.N_SYM) for b in range(mc.N_SYM) if a != b])
        within = np.mean(np.sum((Ht - c[y])**2, 1))
        geometry.append(dict(kind='spread_train', i=i, j=i, value=float(within / between)))
        if i == 3:
            Hh, yh = H[3][held], sym_at(R, 3)[held]
            geometry.append(dict(kind='spread_heldout_vs_train_centroids', i=3, j=3, value=float(np.mean(np.sum((Hh - c[yh])**2, 1)) / between)))
        for j in DEPTHS:
            bj = np.mean([np.linalg.norm(cent[j][a] - cent[j][b]) for a in range(mc.N_SYM) for b in range(mc.N_SYM) if a != b])
            geometry.append(dict(kind='centroid_offset', i=i, j=j, value=float(np.mean(np.linalg.norm(cent[i] - cent[j], axis=1)) / bj)))
    return cells, pairs, geometry, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, nargs='+', default=list(mc.STEPS))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    T = c2task.load(c2task.ORIGINAL)
    out = dict(cells=[], pairs=[], geometry=[]); checks = {}
    for ck in mc.checkpoints(args.steps):
        p = mc.load_verified(ck)
        rng = np.random.default_rng([CONTROL_SEED, list(mc.CONDITIONS).index(ck['condition']), ck['seed'], ck['step']])
        cells, pairs, geometry, chk = analyze(p, T, rng)
        tag = dict(condition=ck['condition'], seed=ck['seed'], run_id=ck['run_id'], step=ck['step'])
        for name, rows in (('cells', cells), ('pairs', pairs), ('geometry', geometry)):
            out[name] += [dict(tag, **r) for r in rows]
        checks[f"{ck['run_id']}@{ck['step']}"] = dict(ckpt_sha256=ck['sha256'], **chk)
        diag = [c['S'] for c in cells if c['i'] == c['j']]
        print(ck['run_id'], ck['step'], f'diag S {diag[0]:.3f}', chk, flush=True)
    for name, rows in out.items():
        with (args.out/f'{name}.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.out/'checks.json').write_text(json.dumps(dict(
        control_seed=f'[{CONTROL_SEED}, condition index, init seed, step] per checkpoint', recipient_readout=mc.RECIPIENT_READOUT, per_checkpoint=checks,
        provenance=nets.provenance(args.out, mc.__file__, __file__)), indent=1, default=str))


if __name__ == '__main__':
    main()
