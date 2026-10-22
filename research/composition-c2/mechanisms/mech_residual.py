#!/usr/bin/env python3
"""Follow-up (QUESTION.md, "Follow-up"): which part of the history residual goes with the failures?

At boundary i (2 for every condition, 3 as well), the state h_i of each length-3 row is split into
its training-row class mean c_i(x) (x = the row's true symbol at i) and the residual r = h_i - c_i(x).
U_i is the span of the 7 centered training centroids (orthonormal basis by SVD), fit on training
rows only. Patched states are continued through the row's remaining ops (i = 2: op 3; i = 3: every
suffix of length 1-3, L = 4-6) and read with the recipient readout (V3 when per-step), as in M2.

  dose        c + lam * r, lam in LAMBDAS (1 = natural, 0 = the M2 centroid)
  inside      c + P_U r            outside      c + (r - P_U r)
  rand_*      the same with N_RANDOM random 7-d subspaces (seed 64302), matched in dimension
  swap_*      held-out subspace interchange at i = 2: h_b + P (h_d - h_b) on held-out-program bases,
              every depth-2 donor prefix with a different symbol; target = the base's op 3 on the
              donor's symbol. swap_U / swap_U_complement use P_U / I - P_U, swap_rand(_complement)
              the depth-2 random subspaces, swap_whole the whole donor state (= M2, a reference)

uv run --locked python research/composition-c2/mechanisms/mech_residual.py --out DIR
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

SEED = 64302
LAMBDAS = (0., .25, .5, .75, 1.)
N_RANDOM = 5
RANK = mc.N_SYM - 1


def centroid_basis(Htr, ytr):
    """Class means c (8, d) and an orthonormal basis (d, 7) of their centered span."""
    c = np.stack([Htr[ytr == x].mean(0) for x in range(mc.N_SYM)])
    u, sv, vt = np.linalg.svd(c - c.mean(0), full_matrices=False)
    assert sv[RANK - 1] > 1e-8 * sv[0], sv  # the 7 centered means are linearly independent
    return c, vt[:RANK].T


def random_bases(rng, d):
    return [np.linalg.qr(rng.normal(size=(d, RANK)))[0] for _ in range(N_RANDOM)]


def proj(Q, X):
    return (X @ Q) @ Q.T


def analyze(p, T, rng):
    P = T['P']
    R = c2task.rows(P, c2task.programs(3))
    held = np.isin(mc.lex(R['ops']), mc.lex(np.array(T['held'])))
    H = c2rnn.forward(p, R['ops'], s=R['s'])[0]
    out, checks = [], {}

    def add(i, L, kind, param, subset, ok, retained=None):
        out.append(dict(i=i, L=L, kind=kind, param=param, subset=subset, n=int(ok.size), accuracy=float(ok.mean()),
                        base_retained=float(retained.mean()) if retained is not None else float('nan')))

    fitted = {}
    for i in (2, 3):
        x = R['states'][:, i-1]
        c, U = centroid_basis(H[i][~held], x[~held])
        Qs = random_bases(rng, H[i].shape[1])
        fitted[i] = (c, U, Qs)
        r = H[i] - c[x]
        patched = {('dose', lam): c[x] + lam * r for lam in LAMBDAS}
        patched[('inside', 'U')] = c[x] + proj(U, r)
        patched[('outside', 'U')] = c[x] + r - proj(U, r)
        for k, Q in enumerate(Qs):
            patched[('rand_inside', k)] = c[x] + proj(Q, r)
            patched[('rand_outside', k)] = c[x] + r - proj(Q, r)
        if i == 2:  # each row's own op 3, read as its answer (L = 3)
            for (kind, param), h in patched.items():
                ok = mc.readout(p, mc.run_suffix(p, h, R['ops'][:, 2:], 2)).argmax(1) == R['states'][:, -1]
                for name, m in (('held', held), ('train', ~held), ('all', np.ones_like(held))):
                    add(i, 3, kind, param, name, ok[m])
            natural = mc.readout(p, H[3]).argmax(1)
            dose1 = mc.readout(p, mc.run_suffix(p, patched[('dose', 1.)], R['ops'][:, 2:], 2)).argmax(1)
            checks['dose1_pred_equals_natural'] = bool(np.array_equal(dose1, natural))
        else:  # every suffix of length 1-3 after the depth-3 state, recipient depth j (L = 6 - j)
            for j in (2, 1, 0):
                sfx = mc.suffixes(3 - j)
                ns = len(sfx)
                ops, target = np.tile(sfx, (len(x), 1)), None
                target = mc.apply_ops(P, np.repeat(x, ns), ops)
                hist = np.repeat(held, ns)
                for (kind, param), h in patched.items():
                    ok = mc.readout(p, mc.run_suffix(p, np.repeat(h, ns, axis=0), ops, j)).argmax(1) == target
                    for name, m in (('held_history', hist), ('train_history', ~hist), ('all', np.ones_like(hist))):
                        add(i, 6 - j, kind, param, name, ok[m])

    # held-out subspace interchange at depth 2: bases are held-out-program rows, donors every
    # depth-2 prefix with a different symbol; the base keeps its op 3
    x = R['states'][:, 1]
    c, U, Qs = fitted[2]  # the same U and random subspaces as the depth-2 residual split
    pre = mc.prefixes(2)
    Hd, xd = mc.state(p, pre['s'], pre['ops']), mc.apply_ops(P, pre['s'], pre['ops'])
    bases = np.flatnonzero(held)
    b, d = np.repeat(bases, len(Hd)), np.tile(np.arange(len(Hd)), len(bases))
    keep = xd[d] != x[b]
    b, d = b[keep], d[keep]
    target, base_answer = P[R['ops'][b, 2], xd[d]], R['states'][b, -1]
    delta = Hd[d] - H[2][b]

    def swap(kind, param, h):
        pred = mc.readout(p, mc.run_suffix(p, h, R['ops'][b, 2:], 2)).argmax(1)
        add(2, 3, kind, param, 'held_bases', pred == target, pred == base_answer)
    swap('swap_whole', 'natural', Hd[d])
    for name, Q in [('U', U)] + list(enumerate(Qs)):
        prefix = 'swap_U' if name == 'U' else 'swap_rand'
        swap(prefix, name, H[2][b] + proj(Q, delta))
        swap(prefix + '_complement', name, H[2][b] + delta - proj(Q, delta))
    return out, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    T = c2task.load(c2task.ORIGINAL)
    rows, checks = [], {}
    for ck in mc.checkpoints([10000]):
        rng = np.random.default_rng([SEED, list(mc.CONDITIONS).index(ck['condition']), ck['seed']])
        res, chk = analyze(mc.load_verified(ck), T, rng)
        rows += [dict(condition=ck['condition'], seed=ck['seed'], run_id=ck['run_id'], step=ck['step'], **r) for r in res]
        checks[ck['run_id']] = dict(ckpt_sha256=ck['sha256'], **chk)
        print(ck['run_id'], chk, flush=True)
    with (args.out/'residual.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.out/'checks.json').write_text(json.dumps(dict(
        seed=f'[{SEED}, condition index, init seed] per checkpoint', lambdas=LAMBDAS, n_random=N_RANDOM, rank=RANK,
        per_checkpoint=checks, provenance=nets.provenance(args.out, mc.__file__, __file__)), indent=1, default=str))


if __name__ == '__main__':
    main()
