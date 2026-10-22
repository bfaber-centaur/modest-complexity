#!/usr/bin/env python3
"""Mechanisms gate, part 2 (research/e3/audit/QUESTION-2.md, M1-M5): the nine start files.

Rebuilds arms A, B and C from the source checkpoints alone with an independent real-Fourier
projection (orthonormal cos/sin basis instead of rfft), an independent dominant-frequency
rule and an independent solution of the sham's matching rule (exact breakpoints of the kept
energy as a function of lambda instead of bisection). No pair or label enters any edit. Then
it recomputes the update-0 report with its own forward pass.

uv run --locked python research/e3/audit/start_check.py STARTS_DIR OUT.json LANE_CODE_DIR
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
P, H = 31, 128
F = (P - 1)//2
plan = json.loads((root/'research/e3/plan.json').read_text())
ENTRIES = [e for e in plan['reserved'] + plan['runs'] if e['run_id'].startswith('e3-me-')]
COMMON = np.load(root/'research/e2/data/common-never-trained-30-50.npz')['index']

x = np.arange(P)
# Orthonormal real Fourier basis on the token / label axis: column 0 is DC, then cos k, sin k for k = 1..15.
BASIS = np.column_stack([np.full(P, 1/np.sqrt(P))] + [f(2*np.pi*k*x/P)*np.sqrt(2/P)
                                                      for k in range(1, F + 1) for f in (np.cos, np.sin)])
assert np.allclose(BASIS.T @ BASIS, np.eye(P))


def coefficient_mask(keep_freqs, dc):
    """(P, H) mask over basis columns from per-unit frequency sets."""
    m = np.zeros((P, H), bool)
    m[0] = dc
    for j, fs in enumerate(keep_freqs):
        for k in fs:
            m[2*k - 1, j] = m[2*k, j] = True
    return m


def project(w, keep_freqs):
    """U, V columns onto DC + keep_freqs[j]; W row j onto keep_freqs[j] (no DC)."""
    cu, cv, cw = BASIS.T @ w[0], BASIS.T @ w[1], BASIS.T @ w[2].T
    mi, mo = coefficient_mask(keep_freqs, True), coefficient_mask(keep_freqs, False)
    return [BASIS @ (cu*mi), BASIS @ (cv*mi), (BASIS @ (cw*mo)).T]


def input_energy(w):
    """(F, H): energy of U plus V at frequency k = 1..F per unit (orthonormal coefficients)."""
    c = (BASIS.T @ w[0])**2 + (BASIS.T @ w[1])**2
    return c[1::2] + c[2::2]


def dominant(w):
    live = (np.sum(w[0]**2, 0) + np.sum(w[1]**2, 0) > 0) & (np.sum(w[2]**2, 1) > 0)
    return np.where(live, input_energy(w).argmax(0) + 1, 1)


def sham_sets(w, seed):
    """The rule of research/e3/mechanisms/QUESTION.md: per unit, a seeded order of its 14
    non-dominant frequencies; keep them in order while the kept energy stays below
    lam * dominant energy, keeping the crossing one if that is closer; one lam so that the
    total kept energy is as close as the step function allows to the total dominant energy.
    The unit keeps its first n frequencies, where n counts the midpoints (c_{i-1} + c_i)/2 of
    its cumulative energies c that do not exceed lam * e_dom, so every breakpoint is known."""
    k, e = dominant(w), input_energy(w)
    rng = np.random.default_rng(seed)
    orders = [[int(f) for f in rng.permutation(np.arange(1, F + 1)) if f != k[j]] for j in range(H)]
    edom = e[k - 1, np.arange(H)]
    cum = [np.concatenate([[0.], np.cumsum(e[np.array(o) - 1, j])]) for j, o in enumerate(orders)]
    mids = [(c[:-1] + c[1:])/2 for c in cum]

    def kept(lam):
        n = [int(np.sum(m <= lam*d)) for m, d in zip(mids, edom)]
        return n, sum(c[i] for c, i in zip(cum, n))

    goal = edom.sum()
    breaks = np.unique(np.concatenate([m/d for m, d in zip(mids, edom) if d > 0]))
    inside = np.concatenate([[breaks[0]/2], (breaks[:-1] + breaks[1:])/2, [2*breaks[-1]]])  # one lam per plateau
    totals = np.array([kept(t)[1] for t in inside])
    first = int(np.argmax(totals >= goal))       # the first plateau at or above the goal, or the one
    pick = first if first == 0 or abs(totals[first] - goal) < abs(totals[first - 1] - goal) else first - 1  # below if not farther
    n, total = kept(inside[pick])
    return [o[:i] for o, i in zip(orders, n)], dict(lam_plateau=[float(breaks[pick - 1]) if pick else 0., float(breaks[pick])],
                                                    kept_over_goal=float(total/goal),
                                                    units_unmatchable=int(np.sum(e.sum(0) - edom < edom)))


def update0(w0, w, a, b, y, train):
    z0, z = w0[0][a] + w0[1][b], w[0][a] + w[1][b]
    logits = np.maximum(z, 0) @ w[2]
    lse = np.logaddexp.reduce(logits[train], axis=1)
    acc = lambda ix: float(np.mean(logits[ix].argmax(1) == y[ix]))
    held = np.setdiff1d(np.arange(P*P), train)
    out = {f'displacement_{n}': float(np.linalg.norm(w[i] - w0[i])/np.linalg.norm(w0[i])) for i, n in enumerate('UVW')}
    out.update({f'kept_energy_{n}': float(np.sum(w[i]**2)/np.sum(w0[i]**2)) for i, n in enumerate('UVW')})
    out.update(gate_flips=float(np.mean((z > 0) != (z0 > 0))), train_loss=float(np.mean(lse - logits[train, y[train]])),
               train_acc=acc(train), common_acc=acc(COMMON), own_heldout_acc=acc(held))
    return out


def literal(starts, lane_code):
    """The literal forms of M3 and M4: the files against e2 filter_units and the lane's own
    sham function (edits.py at the lane's commit, copied to LANE_CODE), and B's accuracies
    against e2's published unitfilter table."""
    sys.path[:0] = [str(lane_code), str(root/'research/e2/mechanisms'), str(root/'research/mechanisms')]
    from unitfilter import dominant as e2_dominant, filter_units
    import edits
    e2 = {r['run_id']: r for r in csv.DictReader(open(root/'research/e2/mechanisms/figures/unitfilter/unitfilter.csv'))
          if r['step'] == '50000' and r['kind'] == 'filter dominant'}
    out = {}
    for e in ENTRIES:
        arm = e['run_id'].split('-')[2]
        if arm == 'keep':
            continue
        with np.load(root/e['source_checkpoint']) as z:
            w0 = [z[k] for k in 'UVW']
        with np.load(Path(starts)/f"{e['run_id']}.npz") as z:
            w = [z[k] for k in 'UVW']
        ref = filter_units(w0, e2_dominant(w0)[0]) if arm == 'filt' else edits.sham(w0, e['sham_seed'])[0]
        out[e['run_id']] = dict(equal_to=('e2 filter_units' if arm == 'filt' else 'lane edits.sham'),
                                exact=all(np.array_equal(p, q) for p, q in zip(w, ref)),
                                max_abs_diff=max(float(np.abs(p - q).max()) for p, q in zip(w, ref)))
        if arm == 'filt':
            out[e['run_id']]['e2_published'] = {k: float(e2[e['source_run']][k]) for k in ('common_acc', 'own_heldout_acc', 'train_acc')}
    return out


def main(starts, out_json, lane_code):
    starts = Path(starts)
    lane = {r['run_id']: r for r in csv.DictReader(open(starts/'update0.csv'))}
    a, b = np.repeat(x, P), np.tile(x, P)
    y = (a + b) % P
    res, src = {}, {}
    for e in ENTRIES:
        f = starts/f"{e['run_id']}.npz"
        with np.load(f) as z:
            keys, w = sorted(z.files), [z[k] for k in 'UVW']
        if e['source_run'] not in src:
            with np.load(root/e['source_checkpoint']) as z:
                src[e['source_run']] = [z[k] for k in 'UVW']
            with np.load(root/Path(e['source_checkpoint']).parent/'data.npz') as d:
                train = d['train']
        w0 = src[e['source_run']]
        arm = e['run_id'].split('-')[2]
        r = dict(arm=arm, keys=keys, dtypes=sorted({str(v.dtype) for v in w}), shapes=[list(v.shape) for v in w],
                 sha256=hashlib.sha256(f.read_bytes()).hexdigest(), sha256_matches_lane=None)
        r['sha256_matches_lane'] = r['sha256'] == lane[e['run_id']]['start_sha256']
        r['M1'] = keys == ['U', 'V', 'W'] and r['dtypes'] == ['float64'] and r['shapes'] == [[P, H], [P, H], [H, P]] \
            and r['sha256_matches_lane']
        if arm == 'keep':
            ref, info = w0, {}
        elif arm == 'filt':
            ref, info = project(w0, [[k] for k in dominant(w0)]), {}
        else:
            sets, info = sham_sets(w0, e['sham_seed'])
            info['dominant_kept_anywhere'] = any(k in s for k, s in zip(dominant(w0), sets))
            ref = project(w0, sets)
        r['max_abs_diff_vs_independent'] = max(float(np.abs(p - q).max()) for p, q in zip(w, ref))
        r['exact'] = all(np.array_equal(p, q) for p, q in zip(w, ref))
        r.update(info)
        mine = update0(w0, w, a, b, y, train)
        r['update0_max_abs_diff_vs_lane'] = max(abs(v - float(lane[e['run_id']][k])) for k, v in mine.items())
        r['update0_own'] = mine
        r['M2_M3_M4'] = r['exact'] if arm == 'keep' else r['max_abs_diff_vs_independent'] <= 1e-12
        r['M5'] = r['update0_max_abs_diff_vs_lane'] <= 1e-9
        res[e['run_id']] = r
    for k, v in literal(starts, lane_code).items():
        res[k]['literal'] = v
        if 'e2_published' in v:
            v['matches_own_update0'] = all(abs(v['e2_published'][c] - res[k]['update0_own'][c]) <= 1e-12 for c in v['e2_published'])
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    for k, r in res.items():
        print(k, {c: r[c] for c in ('M1', 'M2_M3_M4', 'M5', 'max_abs_diff_vs_independent', 'update0_max_abs_diff_vs_lane')}
              | {c: r[c] for c in ('kept_over_goal', 'units_unmatchable', 'dominant_kept_anywhere') if c in r}
              | ({c: r['literal'][c] for c in ('exact', 'max_abs_diff', 'matches_own_update0') if c in r['literal']} if 'literal' in r else {}))


if __name__ == '__main__':
    main(*sys.argv[1:4])
