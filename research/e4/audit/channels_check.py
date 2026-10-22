#!/usr/bin/env python3
"""e4 audit, QUESTION.md section 2 and addendum 1: my implementation of the mechanisms lane's
channel decomposition (design d584af7) on published checkpoints, its invariances, and control C2.

Decomposition (object: logits; basis: each unit's update-0 dominant input frequency k_j, never
re-selected; W rows split along the label axis by rfft bin):
    z - mean_c z = mu.W~ (label offset) + (h - mu).W_ref + (h - mu).W_off,
h = relu(U[a] + V[b]), mu = mean of h over the 299 training pairs, W~ = W minus its row means.
Linear margins on the 483 common pairs: z_{a+b} - mean over S, and over N (D5 label sets),
averaged over pairs so that they add across channels.

Arms: B (filt, wd 1 and 0), A (keep, wd 1), C (sham, wd 1 and 0), and the C1 random-feature
runs (rf, wd 1 and 0). Checks M1-M4, readings R1-R4, control C2, coherence and full-table
projections.

uv run --locked python research/e4/audit/channels_check.py OUT_DIR
"""
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rf_starts import COMMON, ROOT, P  # noqa: E402
import experiment  # noqa: E402  (ROOT on sys.path via rf_starts)
from unitfilter import dominant  # noqa: E402
import pairwise  # noqa: E402
from c1_check import load_dir, weights  # noqa: E402

SEEDS = (30000, 30001, 30002)
STEPS = (0, 10, 30, 100, 300, 1000, 2000, 5000, 10000, 15000, 20000)
ME, AU = ROOT/'research/e3/mechanisms/runs', HERE/'runs'
LR = .001


def arms(seed):
    return {('B', 1.): ME/f'e3-me-filt-w128-i{seed}-a1', ('B', 0.): ME/f'e3-me-filt-wd0-w128-i{seed}-a1',
            ('A', 1.): ME/f'e3-me-keep-w128-i{seed}-a1', ('C', 1.): ME/f'e3-me-sham-w128-i{seed}-a1',
            ('C', 0.): ME/f'e3-me-sham-wd0-w128-i{seed}-a1', ('rf', 1.): AU/f'e4-au-rf-w128-i{seed}-a1',
            ('rf', 0.): AU/f'e4-au-rf-wd0-w128-i{seed}-a1'}


def label_split(W, k):
    """W rows by label-axis rfft bin: DC, the bin k_j of each unit, the rest, and the rest's part at bins in K = {k_j}."""
    F = np.fft.rfft(W, axis=1)
    rows = np.arange(len(k))
    dc, ref = np.zeros_like(F), np.zeros_like(F)
    dc[:, 0] = F[:, 0]
    ref[rows, k] = F[rows, k]
    off = F - dc - ref
    in_k = np.zeros_like(F)
    K = np.unique(k)
    in_k[:, K] = off[:, K]
    back = lambda X: np.fft.irfft(X, n=P, axis=1)
    return back(dc), back(ref), back(off), back(in_k)


class Sets:
    def __init__(self, d, common):
        S = pairwise.operand_labels(d['a'], d['b'], d['train'], P)
        a, b, self.y = d['a'][common], d['b'][common], d['yadd'][common]
        self.right = np.eye(P, dtype=bool)[self.y]
        self.S = (S[a] | S[b]) & ~self.right
        self.N = ~(S[a] | S[b]) & ~self.right

    def margins(self, x):
        """Mean over pairs of x_{a+b} - mean over S and - mean over N; x is (pairs, p) or (p,)."""
        x = np.broadcast_to(x, self.right.shape)
        y = x[self.right]
        mean = lambda m: (x*m).sum(1)/m.sum(1)
        return float(np.mean(y - mean(self.S))), float(np.mean(y - mean(self.N)))

    def acc(self, x):
        return float(np.mean(np.broadcast_to(x, self.right.shape).argmax(1) == self.y))

    def scale_free(self, z):
        zc = z - z.mean(1, keepdims=True)
        wrong = zc[~self.right].reshape(len(zc), P - 1)
        nz = np.where(self.N, zc, -np.inf).max(1)
        pref = lambda A, B: float(np.nanmean(pairwise.preference(zc, A, B)[0]))
        return dict(acc=self.acc(zc), sigma_margin=float(np.median((zc[self.right] - nz)/wrong.std(1))),
                    P_yN=pref(self.right, self.N), P_yS=pref(self.right, self.S), P_SN=pref(self.S, self.N))


def channels(w, k, d, common, sets):
    U, V, W = w
    h = np.maximum(U[d['a']] + V[d['b']], 0)
    mu = h[d['train']].mean(0)
    Wdc, Wref, Woff, WoffK = label_split(W, k)
    z = h @ W
    zc = z - z.mean(1, keepdims=True)
    hc = (h - mu)[common]
    ch = dict(offset=mu @ (W - Wdc), ref=hc @ Wref, off=hc @ Woff, off_inK=hc @ WoffK)
    ch['off_outK'] = ch['off'] - ch['off_inK']
    err = np.abs(ch['offset'] + ch['ref'] + ch['off'] - zc[common]).max()/np.abs(zc[common]).max()
    dc_spread = float((h @ Wdc).std(1).max()/np.abs(z).max())
    rms = float(np.sqrt(np.mean(zc[common]**2)))
    # post hoc: a scale that no channel enters; logits are 2-homogeneous in (U, V) and W
    npr = float(np.sqrt(np.sum(U**2) + np.sum(V**2))*np.linalg.norm(W))
    out = dict(identity_rel_err=float(err), label_dc_rel_spread=dc_spread, rms_centred=rms, norm_product=npr,
               acc_full=sets.acc(zc[common]), acc_ref_only=sets.acc(ch['ref']), acc_no_off=sets.acc(ch['offset'] + ch['ref']),
               acc_no_offset=sets.acc(ch['ref'] + ch['off']))
    for name in ('offset', 'ref', 'off', 'off_inK', 'off_outK'):
        mS, mN = sets.margins(ch[name])
        out.update({f'mS_{name}': mS, f'mN_{name}': mN, f'mS_{name}_norm': mS/rms, f'mN_{name}_norm': mN/rms,
                    f'mS_{name}_per_np': mS/npr, f'mN_{name}_per_np': mN/npr})
    mS, mN = sets.margins(zc[common])
    out.update(mS_total=mS, mN_total=mN, mS_total_norm=mS/rms, mN_total_norm=mN/rms)
    return out, dict(h=h, mu=mu, Wref=Wref, z=z)


def coherence(h, W, k, pairs, d):
    """Sum_k |Sum_{j in k} A_j| / Sum_k Sum_{j in k} |A_j|, A_j the unit's coefficient on e^{-2 pi i k_j (a+b-c)/p}."""
    ph = np.exp(-2j*np.pi*np.outer((d['a'][pairs] + d['b'][pairs]) % P, np.arange(P))/P)  # (pairs, freq)
    hk = (h[pairs]*ph[:, k]).sum(0)  # sum over pairs of h_j e^{-i k_j (a+b)}
    wk = (W*np.exp(2j*np.pi*np.outer(k, np.arange(P))/P)).sum(1)  # sum over c of W_j(c) e^{+i k_j c}
    A = hk*wk
    num = sum(abs(A[k == f].sum()) for f in np.unique(k))
    return float(num/np.abs(A).sum())


def projection(z, pairs, d, common, sets):
    """g(d) = mean over pairs of z(a, b, a+b-d); common accuracy implied (1 if argmax g = 0)."""
    a, b = d['a'][pairs], d['b'][pairs]
    g = np.array([z[pairs, (a + b - off) % P].mean() for off in range(P)])
    return dict(argmax_offset=int(g.argmax()), implied_common_acc=float(g.argmax() == 0))


def m3_m4(w, d, common, sets):
    """Invariance (M3) and the decay factor (M4)."""
    z = np.maximum(w[0][d['a']] + w[1][d['b']], 0) @ w[2]
    base = sets.scale_free(z[common])
    rng = np.random.default_rng(0)
    z2 = np.maximum(2*w[0][d['a']] + 2*w[1][d['b']], 0) @ (3*w[2])
    variants = dict(times_3p7=3.7*z, plus_pair_constant=z + rng.normal(0, 10, (len(z), 1)), U_V_by_2_W_by_3=z2)
    m3 = {n: max(abs(sets.scale_free(v[common])[q] - base[q]) for q in base) for n, v in variants.items()}
    lin = sets.margins((z - z.mean(1, keepdims=True))[common])
    lin6 = sets.margins((z2 - z2.mean(1, keepdims=True))[common])
    m3['linear_margin_ratio_under_U_V_by_2_W_by_3'] = [lin6[0]/lin[0], lin6[1]/lin[1]]
    m4 = {}
    for wd in (1., .3, .1, 0.):
        x = [q.copy() for q in w]
        experiment.adam(x, [np.zeros_like(q) for q in w], [np.zeros_like(q) for q in w], [np.zeros_like(q) for q in w], 1, LR, wd)
        m4[str(wd)] = all(np.array_equal(xi, wi*(1 - LR*wd)) for xi, wi in zip(x, w))
    t = np.arange(1, 20001)
    m4['exposure_equals_minus_log_rho_max_rel'] = float(np.max(np.abs(-np.log((1 - LR)**t) - (-t*np.log(1 - LR)))/(t*-np.log(1 - LR))))
    return m3, m4


def main(out_dir):
    common = np.load(COMMON)['index']
    res = dict(rows=[], m1={}, m2={}, r4={}, coherence={}, projection={})
    for seed in SEEDS:
        for (arm, wd), run_dir in arms(seed).items():
            d, steps = load_dir(run_dir)
            sets = Sets(d, common)
            w0 = weights(run_dir, 0)
            k = dominant(w0)[0]
            cache = {}
            for s in STEPS:
                if s not in steps:
                    continue
                w = weights(run_dir, s)
                row, parts = channels(w, k, d, common, sets)
                row.update(arm=arm, wd=wd, seed=seed, step=s, rho2=(1 - LR*wd)**(2*s), **{f'sf_{q}': v for q, v in sets.scale_free(parts['z'][common]).items()})
                res['rows'].append(row)
                cache[s] = (parts, w)
                if s in (0, 100, 2000, 20000) and arm in ('A', 'B') and wd == 1.:
                    sets_ix = dict(all=np.arange(P*P), common=common, train=d['train'])
                    res['coherence'][f'{arm}/{seed}/{s}'] = {n: coherence(parts['h'], w[2], k, ix, d) for n, ix in sets_ix.items()}
                    res['projection'][f'{arm}/{seed}/{s}'] = {n: projection(parts['z'], ix, d, common, sets) for n, ix in sets_ix.items()}
            if arm in ('B', 'rf') and 100 in cache and 2000 in cache:
                (p1, _), (p2, _) = cache[100], cache[2000]
                X1, X2 = (p1['h'] - p1['mu'])[common], (p2['h'] - p2['mu'])[common]
                R1, R2 = p1['Wref'], p2['Wref']
                terms = dict(features_first=dict(features=X2 @ R1 - X1 @ R1, readout=X2 @ R2 - X2 @ R1),
                             readout_first=dict(readout=X1 @ R2 - X1 @ R1, features=X2 @ R2 - X1 @ R2))
                res['r4'][f'{arm}/{seed}/{wd}'] = {o: {n: sets.margins(v) for n, v in t.items()} for o, t in terms.items()}
                res['r4'][f'{arm}/{seed}/{wd}']['total_change'] = sets.margins(X2 @ R2 - X1 @ R1)
            if (arm, wd) == ('B', 1.) and seed == 30000:
                res['m3'], res['m4'] = m3_m4(weights(run_dir, 100), d, common, sets)
    # M2: k_j from the B start equals e2 dominant() of the 50k source (the frequency the filter kept)
    plan3 = {r['run_id']: r for r in json.loads((ROOT/'research/e3/plan.json').read_text())['runs']}
    for seed in SEEDS:
        with np.load(ROOT/plan3[f'e3-me-filt-w128-i{seed}-a1']['source_checkpoint']) as z:
            src = [z[q] for q in 'UVW']
        kb = dominant(weights(ME/f'e3-me-filt-w128-i{seed}-a1', 0))[0]
        res['m2'][seed] = dict(k_from_start_equals_e2_dominant_of_source=bool(np.array_equal(kb, dominant(src)[0])),
                               same_k_wd0_start=bool(np.array_equal(kb, dominant(weights(ME/f'e3-me-filt-wd0-w128-i{seed}-a1', 0))[0])))
    res['m1'] = dict(max_identity_rel_err=max(r['identity_rel_err'] for r in res['rows']),
                     max_label_dc_rel_spread=max(r['label_dc_rel_spread'] for r in res['rows']))
    rows = res['rows']
    get = lambda arm, wd, seed, s: next((r for r in rows if (r['arm'], r['wd'], r['seed'], r['step']) == (arm, wd, seed, s)), None)
    # R1: pure-decay benchmark for the reference channel, from update 100
    res['r1'] = {f'{arm}/{seed}/{wd}': {s: dict(ratio_S=get(arm, wd, seed, s)['mS_ref']/(((1 - LR*wd)**(2*(s - 100)))*get(arm, wd, seed, 100)['mS_ref']),
                                                ratio_N=get(arm, wd, seed, s)['mN_ref']/(((1 - LR*wd)**(2*(s - 100)))*get(arm, wd, seed, 100)['mN_ref']))
                                         for s in STEPS if s >= 100 and get(arm, wd, seed, s)}
                 for arm in ('B', 'rf') for wd in (1., 0.) for seed in SEEDS}
    # C2: B against A at 5k and 20k
    c2 = {}
    for s in (5000, 20000):
        b_rows, a_rows = [get('B', 1., q, s) for q in SEEDS], [get('A', 1., q, s) for q in SEEDS]
        c2[s] = dict(suff_B=[r['acc_ref_only'] for r in b_rows], suff_A=[r['acc_ref_only'] for r in a_rows],
                     mS_norm_B=[r['mS_ref_norm'] for r in b_rows], mS_norm_A=[r['mS_ref_norm'] for r in a_rows],
                     mN_norm_B=[r['mN_ref_norm'] for r in b_rows], mN_norm_A=[r['mN_ref_norm'] for r in a_rows])
        c = c2[s]
        c['specific'] = bool(all(b - a >= .10 for b, a in zip(c['suff_B'], c['suff_A'])) and min(c['mS_norm_B']) > max(c['mS_norm_A'])
                             and min(c['mN_norm_B']) > max(c['mN_norm_A']))
    res['c2'] = c2
    late = [r for r in rows if r['arm'] in ('B', 'rf')]
    res['offset_summary'] = dict(max_abs_acc_change_removing_offset=max(abs(r['acc_no_offset'] - r['acc_full']) for r in late),
                                 max_abs_offset_margin_norm=max(max(abs(r['mS_offset_norm']), abs(r['mN_offset_norm'])) for r in late))
    res['K_sizes'] = {f'{arm}/{seed}': int(len(np.unique(dominant(weights(p, 0))[0]))) for seed in SEEDS for (arm, wd), p in arms(seed).items() if wd == 1.}
    Path(out_dir, 'channels-check.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    keys = list(rows[0])
    with open(Path(out_dir, 'channels-check.csv'), 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(str(round(r[q], 6) if isinstance(r[q], float) else r[q]) for q in keys) + '\n')
    print(json.dumps(dict(m1=res['m1'], m2=res['m2'], m3=res['m3'], m4=res['m4'], c2=res['c2']), indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
