"""c3 correction-lane review (QUESTION.md section 2): C1-C7, CX-C1, CX-C2. The forward pass is the
c2 reviewer's (rev_pin.my_forward); prefix sets, split, centroids, maps, decoders and measures are
written here from the lane's QUESTION.md. The lane's own tables are read from git at LANE_COMMIT.

run_jobs.py runs it as c3-rev-corr-a1:
uv run --locked python research/composition-c3/review/rev_corr.py --out DIR --lane c3f03cd
"""
import argparse
import csv
import io
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
C3, ROOT = HERE.parent, HERE.parents[2]
C2 = ROOT/'research'/'composition-c2'
sys.path[:0] = [str(C3/'common'), str(C2/'review')]
from rev_pin import FAIL, check, get, my_forward  # noqa: E402
import c2probe  # noqa: E402
import c2task  # noqa: E402
import c3correct  # noqa: E402
import nets  # noqa: E402

K, LR, UPDATES = 64, 1e-3, 2000
MAP_SEED, SPLIT_SEED, PROBE_SEED = 71200, 71300, 71301
CX_MAP, CX_PROBE = (73010, 73011, 73012), (73020, 73021, 73022)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1., 10.)
LANE = 'research/composition-c3/correction'


def lane_csv(commit, path):
    return list(csv.DictReader(io.StringIO(subprocess.run(['git', 'show', f'{commit}:{LANE}/{path}'], cwd=ROOT,
                                                          capture_output=True, text=True, check=True).stdout)))


def lane_json(commit, path):
    return json.loads(subprocess.run(['git', 'show', f'{commit}:{LANE}/{path}'], cwd=ROOT, capture_output=True, text=True, check=True).stdout)


def bases():
    rec = {(f['run_id'], f['file']): f['sha256'] for f in json.loads((C2/'recovered'/'RECOVERY.json').read_text())['files']}
    out = []
    for arm in ('r1-rev-comp-rnn-untied', 'r1-comp-rnn-final'):
        for s in (51200, 51201, 51202):
            run = f'{arm}-s{s}-a1'
            out.append(dict(set='original', run_id=run, tid=c2task.ORIGINAL, path=C2/'recovered'/run/'ckpt-10000.npz',
                            sha=rec[(run, 'ckpt-10000.npz')], tied_final=arm.endswith('final')))
    for tid in ('ops63000-ls63100', 'ops63000-ls63101', 'ops63001-ls63102', 'ops63001-ls63103'):
        for s in (63200, 63201):
            run = f'c2-gen-final-{tid}-s{s}-a1'; d = C2/'generalization'/'runs'/run
            sha = [e['sha256'] for e in json.loads((d/'MANIFEST.json').read_text()) if e['file'] == 'ckpt-10000.npz'][0]
            out.append(dict(set='f2', run_id=run, tid=tid, path=d/'ckpt-10000.npz', sha=sha, tied_final=True))
    return out


def my_task(tid):
    """Rows of all 512 length-3 programs (program-major, start-minor), their prefix index
    lex(op1, op2)*8 + s, held flags, and the 128 prefixes with their depth-2 symbol."""
    T = c2task.load(tid); P = np.asarray(T['P']); held = {tuple(int(o) for o in q) for q in T['held']}
    s_, ops_, st_, pr_, he_ = [], [], [], [], []
    for qi, q in enumerate(product(range(4), repeat=3)):
        for s in range(8):
            x1 = P[q[0]][s]; x2 = P[q[1]][x1]; x3 = P[q[2]][x2]
            s_.append(s); ops_.append(q); st_.append((x1, x2, x3)); pr_.append((qi // 4)*8 + s); he_.append(q in held)
    R = dict(s=np.array(s_), ops=np.array(ops_), states=np.array(st_), prefix=np.array(pr_), held=np.array(he_))
    pre_s = np.array([s for _ in product(range(4), repeat=2) for s in range(8)])
    pre_ops = np.array([q for q in product(range(4), repeat=2) for _ in range(8)])
    x = np.array([P[q[1]][P[q[0]][s]] for q, s in zip(pre_ops, pre_s)])
    return R, dict(s=pre_s, ops=pre_ops, x=x)


def my_split(x):
    """Within each symbol 0..7, its prefixes in index order, permuted by one generator, cut 8/4/4."""
    rng = np.random.default_rng(SPLIT_SEED); out = dict(fit=[], val=[], eval=[])
    for k in range(8):
        idx = rng.permutation(np.flatnonzero(x == k))
        out['fit'] += list(idx[:8]); out['val'] += list(idx[8:12]); out['eval'] += list(idx[12:])
    return {k: np.array(sorted(v)) for k, v in out.items()}


def my_map(H, Y, seed, k=K, lr=LR, updates=UPDATES):
    r = np.random.default_rng(seed)
    q = dict(W1=r.normal(0, np.sqrt(2/H.shape[1]), (H.shape[1], k)), b1=np.zeros(k),
             W2=r.normal(0, np.sqrt(1/k), (k, Y.shape[1])), b2=np.zeros(Y.shape[1]))
    st = nets.adam_state(q)
    for _ in range(updates):
        Z = np.maximum(H @ q['W1'] + q['b1'], 0); G = np.maximum(Z @ q['W2'] + q['b2'], 0)
        dG = 2*(G - Y)/G.size*(G > 0); dZ = (dG @ q['W2'].T)*(Z > 0)
        nets.adam_step(q, dict(W1=H.T @ dZ, b1=dZ.sum(0), W2=Z.T @ dG, b2=dG.sum(0)), st, lr)
    return lambda h: np.maximum(np.maximum(h @ q['W1'] + q['b1'], 0) @ q['W2'] + q['b2'], 0)


def nearest(C, H):
    return ((H[:, None] - C[None])**2).sum(-1).argmin(1)


def ridge(Hf, xf, lam):
    mu = Hf.mean(0); Y = np.eye(8)[xf]; yb = Y.mean(0); X = Hf - mu
    W = np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ (Y - yb))
    return lambda H: ((H - mu) @ W + yb).argmax(1)


def final_acc(p, R, Hc_pre, sel):
    """Final accuracy on rows sel when each row's depth-2 state is Hc_pre[its prefix]; step 3 is mine."""
    h = Hc_pre[R['prefix'][sel]]
    h3 = np.maximum(h @ get(p, 'A', 3) + np.eye(4)[R['ops'][sel, 2]] @ get(p, 'B', 3) + get(p, 'bh', 3), 0)
    return float(np.mean((h3 @ get(p, 'V', 3) + get(p, 'bv', 3)).argmax(1) == R['states'][sel, 2]))


def evaluate(b, p, R, pre, split):
    """My numbers for one base and split: {(arm, group, rows): acc}, {(arm, group): (sym, cerr, dec)},
    plus the counterexample arms."""
    H = my_forward(p, pre['s'], pre['ops'], [1, 2])[0][2]; x = pre['x']
    sp = dict(fit=np.arange(128)) if split == 'main' else my_split(x)
    f = sp['fit']; Hf, xf = H[f], x[f]
    C = np.stack([Hf[xf == k].mean(0) for k in range(8)])
    d = ((C[:, None] - C[None])**2).sum(-1); scale = d[~np.eye(8, dtype=bool)].mean()
    probe = lambda seed: nets.forward(c2probe.fit_probe(Hf, xf, seed), H)[1]['y'].argmax(1)
    xd = probe(PROBE_SEED)
    arms = dict(A=H, B=my_map(Hf, C[xf], MAP_SEED)(H), C=my_map(Hf, Hf, MAP_SEED)(H), D=C[xd], E=C[x])
    dec = dict(D=xd)
    for s in CX_MAP:
        arms[f'B@{s}'] = my_map(Hf, C[xf], s)(H)
    for s in CX_PROBE:
        dec[f'D@{s}'] = probe(s); arms[f'D@{s}'] = C[dec[f'D@{s}']]
    dec['D-nc'] = nearest(C, H); arms['D-nc'] = C[dec['D-nc']]
    if split == 'grouped':
        v = sp['val']; best = max(LAMBDAS, key=lambda lam: (np.mean(ridge(Hf, xf, lam)(H[v]) == x[v]), lam))
        dec['D-ridge'] = ridge(Hf, xf, best)(H); arms['D-ridge'] = C[dec['D-ridge']]
    acc, st, extra = {}, {}, {}
    for g, idx in sp.items():
        in_g = np.isin(R['prefix'], idx)
        for arm, Hc in arms.items():
            for rows, flag in (('held', True), ('train', False)):
                acc[(arm, g, rows)] = final_acc(p, R, Hc, np.flatnonzero(in_g & (R['held'] == flag)))
            st[(arm, g)] = (float(np.mean(nearest(C, Hc[idx]) == x[idx])), float(np.mean(((Hc[idx] - C[x[idx]])**2).sum(1)))/scale,
                            float(np.mean(dec[arm][idx] == x[idx])) if arm in dec else None)
        # C6: rows whose B-corrected state has the wrong nearest-centroid symbol
        sel = np.flatnonzero(in_g & R['held'])
        wrong = nearest(C, arms['B'])[R['prefix'][sel]] != R['states'][sel, 1]
        extra[g] = (int(wrong.sum()), final_acc(p, R, arms['B'], sel[wrong]) if wrong.any() else None)
    if split == 'grouped':
        extra['ridge_lambda'] = best
    return acc, st, extra, sp


def gain_bin(acc, arm, g):
    a, e = acc[('A', g, 'held')], acc[('E', g, 'held')]
    if e - a < .1:
        return 'n/a'
    r = (acc[(arm, g, 'held')] - a)/(e - a)
    return '<20%' if r < .2 else ('20-60%' if r <= .6 else '>60%')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--lane', required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    runs = {('original', 'main'): 'c3-corr-main-a1', ('original', 'grouped'): 'c3-corr-grouped-a1',
            ('f2', 'main'): 'c3-corr-f2-main-a1', ('f2', 'grouped'): 'c3-corr-f2-grouped-a1'}
    lane = {k: dict(acc={(r['run_id'], r['arm'], r['prefixes'], r['rows']): (float(r['accuracy']), int(r['n'])) for r in lane_csv(a.lane, f'runs/{v}/accuracy.csv')},
                    st={(r['run_id'], r['arm'], r['prefixes']): r for r in lane_csv(a.lane, f'runs/{v}/states.csv')},
                    checks=lane_json(a.lane, f'runs/{v}/checks.json')) for k, v in runs.items()}
    mine, dump = {}, []
    for b in bases():
        sha = nets.sha256_file(b['path']); p, st_, meta = nets.load(b['path'])
        h = c3correct.params_sha256(p)
        R, pre = my_task(b['tid'])
        for split in ('main', 'grouped'):
            L = lane[(b['set'], split)]
            recs = [r for r in L['checks']['bases'] if r['run_id'] == b['run_id']]
            check(f"C1 {b['run_id']} {split}: file hash, step 10000, params hash equal to the lane's before/after records",
                  sha == b['sha'] and st_['t'] == meta['step'] == 10000 and len(recs) == 1
                  and recs[0]['params_sha256'] == recs[0]['params_sha256_after'] == recs[0].get('params_sha256_after_fit', h) == h)
            acc, st, extra, sp = evaluate(b, p, R, pre, split)
            mine[(b['run_id'], split)] = (acc, st, extra)
            if split == 'grouped':
                pl = L['checks'].get('prefixes_by_task', {}).get(b['tid']) or L['checks'].get('prefixes')
                rows_g = {g: set(np.flatnonzero(np.isin(R['prefix'], v)).tolist()) for g, v in sp.items()}
                check(f"C2 {b['run_id']}: my grouped split equals the lane's; 8/4/4 per symbol; disjoint; each prefix's 4 rows in one group; eval = 32 held + 96 train rows",
                      pl is not None and all(sorted(pl[g]) == sp[g].tolist() for g in sp)
                      and all(np.bincount(pre['x'][sp[g]], minlength=8).tolist() == [n]*8 for g, n in (('fit', 8), ('val', 4), ('eval', 4)))
                      and not (rows_g['fit'] & rows_g['eval'] or rows_g['fit'] & rows_g['val'] or rows_g['val'] & rows_g['eval'])
                      and len(rows_g['fit'] | rows_g['val'] | rows_g['eval']) == 512
                      and int(R['held'][sorted(rows_g['eval'])].sum()) == 32 and len(rows_g['eval']) == 128)
            bad, worst = [], 0
            for (run, arm, g, rows), (v, n) in L['acc'].items():
                if run != b['run_id']:
                    continue
                d = abs(acc[(arm, g, rows)] - v)*n
                worst = max(worst, d) if arm in 'BC' else worst
                if (arm in 'ADE' and d > 1e-9) or d > 1 + 1e-9:
                    bad.append((arm, g, rows, v, acc[(arm, g, rows)]))
            n_l = sum(1 for k in L['acc'] if k[0] == b['run_id'])
            check(f"C3 {b['run_id']} {split}: my accuracy equals the lane's ({n_l} cells): A, D, E exact; B, C within 1 row", not bad and n_l > 0,
                  f'worst B/C {worst:.0f} rows; {bad[:3]}')
            bad = []
            for (run, arm, g), r in L['st'].items():
                if run != b['run_id']:
                    continue
                s_, c_, d_ = st[(arm, g)]
                ok = abs(float(r['symbol_acc']) - s_) < (1e-9 if arm in 'ADE' else .04) and abs(float(r['centroid_err']) - c_) < (1e-9 if arm in 'ADE' else .02)
                ok = ok and (r['decoder_acc'] == '' or abs(float(r['decoder_acc']) - d_) < 1e-9)
                if not ok:
                    bad.append((arm, g, r['symbol_acc'], s_, r['centroid_err'], c_))
            check(f"C4 {b['run_id']} {split}: symbol accuracy, centroid error and decoder accuracy equal the lane's", not bad, str(bad[:2]))
            g = 'fit' if split == 'main' else 'eval'
            nw, aw = extra[g]
            line = dict(run=b['run_id'], split=split, group=g, **{arm: round(acc[(arm, g, 'held')], 4) for arm in sorted({k[0] for k in acc})}, wrong_symbol_rows=nw, B_acc_on_wrong_symbol_rows=aw, ridge_lambda=extra.get('ridge_lambda'))
            dump.append(line)
            print('   ', json.dumps(line))
            check(f"C6 {b['run_id']} {split} ({g} held): B <= D + 1/32", acc[('B', g, 'held')] <= acc[('D', g, 'held')] + 1/32 + 1e-12,
                  f"B {acc[('B', g, 'held')]:.3f} D {acc[('D', g, 'held')]:.3f}")
            check(f"C6 {b['run_id']} {split}: B answers <= 1/8 of its held rows whose corrected symbol is wrong", aw is None or aw <= 1/8 + 1e-12,
                  f'{nw} rows, B correct on {aw}')
    # CX-C1: bins of gain recovered across map and probe draws, six original bases.
    for split, g in (('main', 'fit'), ('grouped', 'eval')):
        for arm, alts in (('B', [f'B@{s}' for s in CX_MAP]), ('D', [f'D@{s}' for s in CX_PROBE])):
            same, detail = 0, []
            for b in [b for b in bases() if b['set'] == 'original']:
                acc = mine[(b['run_id'], split)][0]
                bins = [gain_bin(acc, x, g) for x in [arm] + alts]
                same += len(set(bins)) == 1; detail.append(f"{b['run_id'][3:-3]} {'/'.join(bins)}")
            check(f'CX-C1 {split} {arm}: bin of gain recovered unchanged across 3 other draws in >= 5 of 6 bases', same >= 5, f'{same}/6: ' + '; '.join(detail))
    # CX-C2: other decoders on grouped eval prefixes of the tied/final bases (original and F2).
    worst_dec, worst_fin, rows_ = -1., -1., []
    for b in [b for b in bases() if b['tied_final']]:
        acc, st, extra = mine[(b['run_id'], 'grouped')]
        for alt in ('D-nc', 'D-ridge'):
            dd = st[(alt, 'eval')][2] - st[('D', 'eval')][2]; df = acc[(alt, 'eval', 'held')] - acc[('D', 'eval', 'held')]
            worst_dec, worst_fin = max(worst_dec, dd), max(worst_fin, df)
            rows_.append(f"{b['run_id'].replace('c2-gen-final-', '')[:22]} {alt} dec {st[(alt, 'eval')][2]:.3f} (D {st[('D', 'eval')][2]:.3f}) held {acc[(alt, 'eval', 'held')]:.3f} (D {acc[('D', 'eval', 'held')]:.3f})")
    for r in rows_:
        print('   ', r)
    check('CX-C2 tied/final grouped eval: nearest-centroid and val-tuned ridge decoders raise decoder accuracy by <= 0.15 and held final accuracy by <= 0.1 over the probe, in every base',
          worst_dec <= .15 + 1e-12 and worst_fin <= .1 + 1e-12, f'max rise: decoder {worst_dec:+.3f}, held final {worst_fin:+.3f}')
    # C7: every value in the two figure CSVs equals my recomputation.
    run_of = lambda cond, seed: f"r1-rev-comp-rnn-untied-s{seed}-a1" if cond == 'per-step readout' else f'r1-comp-rnn-final-s{seed}-a1'
    for fig, key in (('figures/correction.csv', lambda r: run_of(r['condition'], r['seed'])),
                     ('figures/followup_f2.csv', lambda r: f"c2-gen-final-{r['task_id']}-s{r['seed']}-a1")):
        n, bad, panels = 0, [], set()
        for r in lane_csv(a.lane, fig):
            split = 'grouped' if 'grouped' in r['panel'] else 'main'
            acc, st, _ = mine[(key(r), split)]
            if 'accuracy' in r['panel'] and 'symbol' not in r['panel'] and 'decoder' not in r['panel']:
                v = acc[(r['arm'], r['prefixes'], r['rows'])]
            elif 'symbol' in r['panel']:
                v = st[(r['arm'], r['prefixes'])][0]
            elif 'centroid' in r['panel']:
                v = st[(r['arm'], r['prefixes'])][1]
            else:
                panels.add(r['panel']); continue
            n += 1; tol = 1e-9 if r['arm'] in 'ADE' else (1.01/int(r['n']) if 'accuracy' in r['panel'] or 'symbol' in r['panel'] else .02)
            if abs(float(r['value']) - v) > tol:
                bad.append((r['panel'], r['seed'], r['arm'], r['value'], v))
        check(f'C7 {fig}: every value recomputes from my numbers ({n} values)', not bad and n > 0, f'unmatched panels {sorted(panels)}; {bad[:3]}')
    (a.out/'rev_corr.json').write_text(json.dumps(dump, indent=1))


if __name__ == '__main__':
    main()
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
