#!/usr/bin/env python3
"""e4 audit, QUESTION.md section 3 and addendum 3: the dynamics runs at wd 0.1 and 0.3 (D6, Y1-Y7),
the decay-off runs (D13, Y1'-Y4'), the restricted-decay patch e9cbc84 (D15, Z1-Z4) and the
xref/onlyref runs (D15, X1-X5).

Replays use research/audit/independent.py for the forward pass, gradients and evaluation, and the
AdamW step and frequency projection written here; nothing training-related is imported from
experiment.py. The patch checks import experiment.py (the pin) and experiment.py of 72cfb5d, on
purpose, to compare them.

uv run --locked python research/e4/audit/dynamics_check.py patch OUT_DIR
uv run --locked python research/e4/audit/dynamics_check.py noflag DIAG_RUN_DIR OUT_DIR   (Z1 b)
uv run --locked python research/e4/audit/dynamics_check.py d6 OUT_DIR
uv run --locked python research/e4/audit/dynamics_check.py d13 OUT_DIR
uv run --locked python research/e4/audit/dynamics_check.py d15 OUT_DIR
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for d in ('research/audit', 'research/e3/audit', 'research/e2/audit'):
    sys.path.insert(0, str(ROOT/d))
import independent as ind  # noqa: E402
from pairwise import ladder_sets  # noqa: E402
from margins_check import COMMON, tr30, tr50  # noqa: E402

P, H, LR, B1, B2, EPS = 31, 128, 1e-3, .9, .999, 1e-8
SEEDS = (30000, 30001, 30002)
PIN0, PIN = '72cfb5d', '8108d8b'
SHARED = ('experiment.py', 'launch_runs.py', 'report_map.py', 'publish_run.py', 'instrumentation.py',
          'pyproject.toml', 'uv.lock', 'test_exploration.py', 'test_instrumentation.py')
DY, ME, E3 = ROOT/'research/e4/dynamics', ROOT/'research/e4/mechanisms', ROOT/'research/e3/mechanisms/runs'
STARTS = ROOT/'research/e3/mechanisms/starts'
PLAN = json.loads((ROOT/'research/e4/plan.json').read_text())
PLAN_RUNS = {r['run_id']: {**PLAN['common'], **r} for r in PLAN['runs']}
WALL = {'seconds', 'elapsed_after_diagnostics'}
BOOK = {'out', 'run_id', 'purpose', 'session_label', 'host', 'runtime', 'checkpoint_steps', 'diagnostic_schedule'}
TOL_REL, TOL_ABS, TOL_W = 1e-8, 1e-14, 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(x, y):
    return float(np.linalg.norm(x - y)/max(np.linalg.norm(y), 1e-300))


# ---------- run files ----------

def load(run_dir):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir/'config.json').read_text())
    rows = [json.loads(l) for l in (run_dir/'metrics.jsonl').read_text().splitlines() if l.strip()]
    with np.load(run_dir/'data.npz') as z:
        d = {k: z[k] for k in z.files}
    for k in ('train', 'cal', 'audit'):
        if d[k].dtype == bool:
            d[k] = np.flatnonzero(d[k])
    ck = {int(f.stem.split('_')[1]): f for f in run_dir.glob('checkpoint_*.npz')}
    man = json.loads((run_dir/'manifest.json').read_text()) if (run_dir/'manifest.json').exists() else {}
    return dict(dir=run_dir, cfg=cfg, rows={r['step']: r for r in rows}, d=d, ck=ck, man=man)


def state(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def own(row, d):
    n_cal, n_aud = len(d['cal']), len(d['audit'])
    return (row['cal']*n_cal + row['audit']*n_aud)/(n_cal + n_aud)


def registry(path):
    ev = {}
    for l in Path(path).read_text().splitlines():
        if l.strip():
            e = json.loads(l)
            ev.setdefault(e['run_id'], []).append(e)
    return ev


def shared_code(commit):
    """Shared files of `commit` against 72cfb5d and against the pin: True where identical."""
    same = lambda a, b: subprocess.run(['git', 'diff', '--quiet', a, b, '--', *SHARED], cwd=ROOT).returncode == 0
    return dict(commit=commit, equals_72cfb5d=same(PIN0, commit), equals_pin_8108d8b=same(PIN, commit))


def config_diff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def metric_rows_equal(r1, r2):
    """Equal except wall-clock fields."""
    return {k: v for k, v in r1.items() if k not in WALL} == {k: v for k, v in r2.items() if k not in WALL}


# ---------- my projection, AdamW and replay ----------

FOURIER = np.exp(-2j*np.pi*np.outer(np.arange(P), np.arange(P))/P)  # FOURIER[k, x]


def my_projection(k, live):
    """[P_U, P_V, P_W]: each live unit's component at its frequency k_j (cosine and sine), along the
    token axis of U and V and the label axis of W, from an explicit DFT per unit."""
    k, live = np.asarray(k), np.asarray(live, dtype=float)
    basis = FOURIER[k]  # (h, p): exp(-2 pi i k_j x / p)

    def proj(X):  # X is (p, h), token axis first
        c = np.einsum('jx,xj->j', basis, X)
        return (2/P)*np.real(np.conj(basis).T*c[None, :])*live[None, :]
    return [proj, proj, lambda X: proj(X.T).T]


def rest_of(parts):
    return [lambda X, f=f: X - f(X) for f in parts]


def my_dominant(U, V):
    """Dominant input frequency per unit (1..15) from summed |c_k|^2 of the U and V columns; no labels."""
    e = np.abs(ind.token_fourier(U))**2 + np.abs(ind.token_fourier(V))**2
    live = np.any(U != 0, 0) | np.any(V != 0, 0)
    return 1 + np.argmax(e[1:], 0), live


def my_adamw(w, g, m, v, t, lr, wd, D=None):
    """PyTorch order: decay (all of w, or D(w)), moments, bias-corrected update. Returns the norms."""
    sa = sd = dot = 0.
    for i in range(3):
        decay = -lr*wd*(w[i] if D is None else D[i](w[i]))
        w[i] = w[i] + decay
        m[i] = B1*m[i] + (1 - B1)*g[i]
        v[i] = B2*v[i] + (1 - B2)*g[i]*g[i]
        adapt = -(lr/(1 - B1**t))*m[i]/(np.sqrt(v[i])/np.sqrt(1 - B2**t) + EPS)
        w[i] = w[i] + adapt
        sa += float(np.vdot(adapt, adapt)); sd += float(np.vdot(decay, decay)); dot += float(np.vdot(adapt, decay))
    return dict(adaptive_norm=np.sqrt(sa), decay_norm=np.sqrt(sd), cosine=dot/np.sqrt(sa*sd) if sa*sd > 0 else None)


class Data:
    def __init__(self, d):
        self.d = d
        self.Xa, self.Xb = ind.one_hot_inputs(d['a'], d['b'], P)
        self.tr = d['train']
        self.sets = [('train', d['train']), ('cal', d['cal']), ('audit', d['audit'])]

    def grad(self, w):
        return ind.loss_and_grad(w, self.Xa[self.tr], self.Xb[self.tr], self.d['y'][self.tr])

    def row(self, w):
        z = ind.forward(w, self.Xa, self.Xb)[0]
        out = dict(norm=float(np.sqrt(sum(np.sum(x*x) for x in w))))
        for name, ix in self.sets:
            e = ind.evaluate(z[ix], self.d['y'][ix])
            out[name] = e['acc']; out[f'{name}_ce'] = e['cross_entropy']
        return out


def compare_row(mine, pub, upd):
    """Worst relative gaps of my logged quantities against a published metric row."""
    acc_equal = all(mine[k] == pub[k] for k in ('train', 'cal', 'audit'))
    gaps = {f'{k}_ce': abs(mine[f'{k}_ce'] - pub['objective'][k]['cross_entropy']) for k in ('train', 'cal', 'audit')}
    scale = {f'{k}_ce': abs(pub['objective'][k]['cross_entropy']) for k in ('train', 'cal', 'audit')}
    gaps['norm'], scale['norm'] = abs(mine['norm'] - pub['norm']), pub['norm']
    if upd is not None and pub.get('optimizer_update'):
        for k in ('adaptive_norm', 'decay_norm'):
            gaps[k], scale[k] = abs(upd[k] - pub['optimizer_update'][k]), pub['optimizer_update'][k]
    within = all(gaps[k] <= TOL_REL*scale[k] + TOL_ABS for k in gaps)
    return acc_equal, within, max(gaps[k]/max(scale[k], 1e-300) for k in gaps)


def replay(run, w, m, v, t0, t1, wd, D=None, count0=None, early=300):
    """Updates t0+1..t1 from (w, m, v); the bias-correction count is t, or t - count0 if given.
    Compares every logged row and published checkpoint; 'early' splits the tolerance window."""
    data = Data(run['d'])
    w, m, v = [x.copy() for x in w], [x.copy() for x in m], [x.copy() for x in v]
    res = dict(rows=0, rows_acc_equal=0, rows_within=0, worst_row_rel_early=0., worst_row_rel_late=0.,
               checkpoints={}, first_row_fail=None)
    for t in range(t0 + 1, t1 + 1):
        g = data.grad(w)[1]
        upd = my_adamw(w, g, m, v, t - (count0 or 0), LR, wd, D)
        if t in run['rows']:
            acc_eq, within, worst = compare_row(data.row(w), run['rows'][t], upd)
            res['rows'] += 1; res['rows_acc_equal'] += acc_eq; res['rows_within'] += within
            key = 'worst_row_rel_early' if t <= early else 'worst_row_rel_late'
            res[key] = max(res[key], worst)
            if not (acc_eq and within) and res['first_row_fail'] is None:
                res['first_row_fail'] = t
        if t in run['ck']:
            s = state(run['ck'][t])
            res['checkpoints'][t] = max(rel(w[i], s[k]) for i, k in enumerate('UVW'))
    res['final'] = (w, m, v)
    return res


def summarise(res, until=None):
    ck = {t: e for t, e in res['checkpoints'].items() if until is None or t <= until}
    return dict({k: v for k, v in res.items() if k not in ('final', 'checkpoints')},
                checkpoints={str(t): e for t, e in sorted(ck.items())}, worst_checkpoint_rel=max(ck.values(), default=None))


# ---------- the restricted-decay patch (Z1-Z4) ----------

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def old_experiment(tmp):
    tmp.mkdir(parents=True, exist_ok=True)
    src = subprocess.run(['git', 'show', f'{PIN0}:experiment.py'], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    (tmp/'experiment_72cfb5d.py').write_text(src)
    return module(tmp/'experiment_72cfb5d.py', 'experiment_72cfb5d')


def start(seed):
    with np.load(STARTS/f'e3-me-filt-w128-i{seed}-a1.npz') as z:
        return [z[k].copy() for k in 'UVW']


def proj_file(seed):
    return ME/'intervention'/f'decay_projection_i{seed}.npz'


def patch(out_dir):
    sys.path.insert(0, str(ROOT))
    import experiment as new
    old = old_experiment(Path(out_dir)/'tmp')
    run0 = load(E3/'e3-me-filt-w128-i30000-a1')
    a, b, y, tr = run0['d']['a'], run0['d']['b'], run0['d']['y'], run0['d']['train']
    res = dict(z1a={}, z2={}, z3={}, z4={})
    for seed in SEEDS:
        for wd in (1., .3):
            wo, wn = start(seed), start(seed)
            mo, vo = [np.zeros_like(x) for x in wo], [np.zeros_like(x) for x in wo]
            mn, vn = [np.zeros_like(x) for x in wn], [np.zeros_like(x) for x in wn]
            equal, norms_equal = True, True
            for t in range(1, 301):
                g = new.lossgrad(wn, a[tr], b[tr], y[tr])[1]
                uo = old.adam(wo, [x.copy() for x in g], mo, vo, t, LR, wd, measure=True)
                un = new.adam(wn, [x.copy() for x in g], mn, vn, t, LR, wd, measure=True)
                equal &= all(np.array_equal(p, q) for p, q in zip(wo + mo + vo, wn + mn + vn))
                norms_equal &= uo == un
            res['z1a'][f'{seed}/wd{wd:g}'] = dict(updates=300, bitwise=bool(equal), measured_norms_equal=bool(norms_equal))
        w0 = start(seed)
        with np.load(proj_file(seed)) as z:
            k, live = z['k'], z['live']
        mine = my_projection(k, live)
        rng = np.random.default_rng(41200 + seed)
        z2 = dict(vs_decay_parts=0., idempotent=0., pythagoras=0., cross=0., dc_kept=0.)
        for on in ('reference', 'rest'):
            theirs = new.decay_parts(proj_file(seed), P, H, on)
            ours = mine if on == 'reference' else rest_of(mine)
            for X in [w0[0], w0[1], w0[2]] + [rng.standard_normal((P, H)) for _ in range(5)] + [rng.standard_normal((H, P)) for _ in range(5)]:
                i = 2 if X.shape == (H, P) else 0
                # relative to the input: the rest of W0 is zero (P(W0) = W0), so an output-relative gap is roundoff over roundoff
                z2['vs_decay_parts'] = max(z2['vs_decay_parts'], float(np.linalg.norm(theirs[i](X) - ours[i](X))/np.linalg.norm(X)))
                z2['vs_decay_parts_output_rel'] = max(z2.get('vs_decay_parts_output_rel', 0.), rel(theirs[i](X), ours[i](X)))
        for X, i in [(w0[0], 0), (w0[1], 1), (w0[2], 2)] + [(rng.standard_normal((P, H)), 0) for _ in range(5)]:
            px = mine[i](X); qx = X - px
            n2 = np.sum(X**2)
            z2['idempotent'] = max(z2['idempotent'], rel(mine[i](px), px))
            z2['pythagoras'] = max(z2['pythagoras'], abs(np.sum(px**2) + np.sum(qx**2) - n2)/n2)
            z2['cross'] = max(z2['cross'], abs(np.sum(px*qx))/n2)
            if i < 2:
                z2['dc_kept'] = max(z2['dc_kept'], float(np.abs(qx.mean(0) - X.mean(0)).max()))
        n2 = sum(np.sum(x**2) for x in w0)
        z2['decay_norm_sq_split_update1'] = abs(sum(np.sum((LR*mine[i](w0[i]))**2) + np.sum((LR*(w0[i] - mine[i](w0[i])))**2)
                                                    for i in range(3)) - LR**2*n2)/(LR**2*n2)
        z2['PASS'] = bool(z2['vs_decay_parts'] <= 1e-12 and z2['idempotent'] <= 1e-12 and z2['pythagoras'] <= 1e-12
                          and z2['cross'] <= 1e-12 and z2['dc_kept'] <= 1e-14 and z2['decay_norm_sq_split_update1'] <= 1e-12)
        z2['note'] = 'random matrices: 5 of (p, h) and 5 of (h, p), numpy seed 41200 + init seed'
        res['z2'][seed] = z2
        z3 = {}
        for on in ('reference', 'rest'):
            theirs = new.decay_parts(proj_file(seed), P, H, on)
            ours = mine if on == 'reference' else rest_of(mine)
            w = [x.copy() for x in w0]; m = [np.zeros_like(x) for x in w]; v = [np.zeros_like(x) for x in w]
            upd = new.adam(w, [np.zeros_like(x) for x in w], m, v, 1, LR, 1., measure=True, decay_part=theirs)
            expect = [w0[i] - LR*1.*theirs[i](w0[i]) for i in range(3)]
            mine_expect = [w0[i] - LR*ours[i](w0[i]) for i in range(3)]
            zero_grad_bitwise = all(np.array_equal(w[i], expect[i]) for i in range(3))
            zero_grad_mine = max(rel(w[i], mine_expect[i]) for i in range(3))
            dn = np.sqrt(sum(np.sum((LR*ours[i](w0[i]))**2) for i in range(3)))
            g = new.lossgrad(w0, a[tr], b[tr], y[tr])[1]
            wr, mr, vr = [x.copy() for x in w0], [np.zeros_like(x) for x in w0], [np.zeros_like(x) for x in w0]
            wf, mf, vf = [x.copy() for x in w0], [np.zeros_like(x) for x in w0], [np.zeros_like(x) for x in w0]
            ur = new.adam(wr, [x.copy() for x in g], mr, vr, 1, LR, 1., measure=True, decay_part=theirs)
            new.adam(wf, [x.copy() for x in g], mf, vf, 1, LR, 1., measure=True)
            moments = all(np.array_equal(p, q) for p, q in zip(mr + vr, mf + vf))
            sep = max(float(np.abs((wr[i] - (w0[i] - LR*theirs[i](w0[i]))) - (wf[i] - w0[i]*(1 - LR))).max()) for i in range(3))
            # the same Adam update, rebuilt from the (shared) moments with experiment.adam's expression
            u = [-LR*(mr[i]/(1 - .9**1))/(np.sqrt(vr[i]/(1 - .999**1)) + 1e-8) for i in range(3)]
            rebuilt = all(np.array_equal(wr[i], (w0[i] - LR*1.*theirs[i](w0[i])) + u[i]) and np.array_equal(wf[i], w0[i]*(1 - LR*1.) + u[i])
                          for i in range(3))
            dn_real = np.sqrt(sum(np.sum((LR*ours[i](w0[i]))**2) for i in range(3)))
            z3[on] = dict(zero_grad_bitwise=bool(zero_grad_bitwise), zero_grad_vs_my_P=zero_grad_mine,
                          decay_norm_rel=abs(upd['decay_norm'] - dn)/dn, real_grad_moments_bitwise=bool(moments),
                          real_grad_update_separation_abs=sep, real_grad_rebuilt_bitwise=bool(rebuilt),
                          real_grad_decay_norm_rel=abs(ur['decay_norm'] - dn_real)/dn_real)
            z3[on]['PASS'] = bool(zero_grad_bitwise and zero_grad_mine <= 1e-14 and z3[on]['decay_norm_rel'] <= 1e-12
                                  and moments and (rebuilt or sep <= 1e-15) and z3[on]['real_grad_decay_norm_rel'] <= 1e-12)
        res['z3'][seed] = z3
        kk, ll = my_dominant(w0[0], w0[1])
        meta = json.loads((ME/'intervention/decay_projection.json').read_text())
        entry = next(f for f in meta['files'] if f['source'] == seed)
        plan_sha = {r['decay_projection_sha256'] for r in PLAN_RUNS.values()
                    if r.get('decay_projection') and r['decay_projection'].endswith(f'i{seed}.npz')}
        with np.load(proj_file(seed)) as z:
            start_named = str(z['start_file']); start_sha = str(z['start_sha256'])
        z4 = dict(sha256=sha(proj_file(seed)), equals_json=sha(proj_file(seed)) == entry['sha256'],
                  plan_hashes=sorted(plan_sha), equals_plan=plan_sha == {sha(proj_file(seed))},
                  k_equals_my_dominant=bool(np.array_equal(k, kk)), live_equals_mine=bool(np.array_equal(live, ll)),
                  live_units=int(live.sum()), start_file=start_named,
                  start_sha_matches=start_sha == sha(STARTS/f'e3-me-filt-w128-i{seed}-a1.npz'),
                  P_W0_equals_W0=rel(mine[2](w0[2]), w0[2]),
                  share_of_UV_sq_norm_in_P=float(sum(np.sum(mine[i](w0[i])**2) for i in (0, 1))/sum(np.sum(w0[i]**2) for i in (0, 1))))
        z4['PASS'] = bool(z4['equals_json'] and z4['equals_plan'] and z4['k_equals_my_dominant'] and z4['live_equals_mine']
                          and z4['start_sha_matches'] and z4['P_W0_equals_W0'] <= 1e-12)
        res['z4'][seed] = z4
    res['z1a_PASS'] = all(r['bitwise'] and r['measured_norms_equal'] for r in res['z1a'].values())
    return res


def noflag(diag_dir):
    """Z1 b: a diagnostic CLI run at the pin, no decay flags, against the published D7 wd 1 replay."""
    mine, pub = load(diag_dir), load(ME/'runs/e4-me-filt-wd1-replay-w128-i30001-a1')
    steps = sorted(s for s in pub['ck'] if s in mine['ck'])
    arrays = {}
    for s in steps:
        x, y = state(mine['ck'][s]), state(pub['ck'][s])
        arrays[s] = all(np.array_equal(x[k], y[k]) for k in y)
    rows = sorted(set(mine['rows']) & set(pub['rows']))
    res = dict(commit=mine['cfg']['runtime']['git_commit'], shared=shared_code(mine['cfg']['runtime']['git_commit']),
               decay_flags=[mine['cfg'].get(k) for k in ('decay_projection', 'decay_on', 'decay_projection_sha256')],
               checkpoints_compared=steps, checkpoints_bitwise={str(s): v for s, v in arrays.items()},
               rows_compared=len(rows), rows_equal=sum(metric_rows_equal(mine['rows'][s], pub['rows'][s]) for s in rows),
               config_differences=config_diff(mine['cfg'], pub['cfg']))
    res['PASS'] = bool(all(arrays.values()) and res['rows_equal'] == len(rows) and set(steps) >= {0, 100, 300}
                       and res['shared']['equals_pin_8108d8b'])
    return res


# ---------- D6 (Y1-Y7) ----------

def sigma_margin_and_acc(w, d, sets):
    S, D, N, yc = sets
    z = ind.forward(w, *ind.one_hot_inputs(d['a'][COMMON], d['b'][COMMON], P))[0]
    hit = np.eye(P, dtype=bool)[yc]
    wrong = z[~hit].reshape(len(yc), P - 1)
    sd = wrong.std(1)
    best_n = np.where(N, z, -np.inf).max(1)
    return float(np.mean(z.argmax(1) == yc)), float(np.median((z[hit] - best_n)/sd))


def decline(acc, theta):
    steps = sorted(acc)
    peak = max((s for s in steps if s <= 1000), key=lambda s: (acc[s], -s))
    below = [s for s in steps if s > peak and acc[s] < theta]
    t = below[0] if below else None
    after = [s for s in steps if t is not None and s >= t]
    recross = sum(1 for s0, s1 in zip(after, after[1:]) if acc[s0] < theta <= acc[s1])
    return peak, t, recross


def d6(out_dir):
    reg = registry(DY/'registry.jsonl')
    lane_ck = list(csv.DictReader(open(DY/'figures/decay_checkpoints.csv')))
    lane_sum = {r['run_id']: r for r in csv.DictReader(open(DY/'figures/decay_summary.csv'))}
    lane_dec = {(r['run_id'], r['threshold']): r for r in csv.DictReader(open(DY/'figures/decay_decision.csv'))}
    res = dict(runs={}, y5_worst=0., y5_rows=0)
    sets = None
    summ = {}
    for seed in SEEDS:
        src1, src0 = load(E3/f'e3-me-filt-w128-i{seed}-a1'), load(E3/f'e3-me-filt-wd0-w128-i{seed}-a1')
        for wd in (0., .1, .3, 1.):
            rid = {0.: src0['cfg']['run_id'], 1.: src1['cfg']['run_id']}.get(wd, f'e4-dy-filt-wd0p{int(wd*10)}-w128-i{seed}-a1')
            run = src0 if wd == 0 else src1 if wd == 1 else load(DY/'runs'/rid)
            d, cfg = run['d'], run['cfg']
            if sets is None:
                sets = ladder_sets(d['a'], d['b'], tr30, tr50, COMMON, P)
            r = dict(wd=wd, seed=seed)
            if 0 < wd < 1:
                ev = reg.get(rid, [])
                plan = PLAN_RUNS[rid]
                start_file = ROOT/cfg['init_weights']
                r['y1'] = dict(
                    registry=[e['status'] for e in ev], completed=[e.get('completed_steps') for e in ev if e['status'] == 'complete'],
                    manifest_completed=run['man']['completed_steps'],
                    init_hash_plan=plan['init_weights_sha256'] == cfg['init_weights_sha256'] == cfg['expect_init_sha256'] == sha(start_file),
                    init_weights_plan=plan['init_weights'] == cfg['init_weights'], wd_plan=plan['wd'] == cfg['wd'],
                    checkpoint_steps_plan=plan['checkpoint_steps'] == cfg['checkpoint_steps'],
                    config_differences_vs_wd1=config_diff(cfg, src1['cfg']),
                    shared=shared_code(cfg['runtime']['git_commit']), dirty=run['man']['producing_tree_dirty'])
                y1 = r['y1']
                y1['PASS'] = bool(y1['registry'] == ['running', 'complete'] and y1['completed'] == [20000]
                                  and y1['manifest_completed'] == 20000 and y1['init_hash_plan'] and y1['init_weights_plan']
                                  and y1['wd_plan'] and y1['checkpoint_steps_plan'] and set(y1['config_differences_vs_wd1']) <= BOOK | {'wd'}
                                  and y1['shared']['equals_72cfb5d'] and not y1['dirty'])
                s0, w0, wd1_0 = state(run['ck'][0]), start(seed), state(src1['ck'][0])
                r['y2'] = dict(step0_equals_start=all(np.array_equal(s0[k], w0[i]) for i, k in enumerate('UVW')),
                               step0_equals_wd1_step0=all(np.array_equal(s0[k], wd1_0[k]) for k in 'UVW'),
                               split=cfg['data_hashes']['split_sha256'],
                               split_ok=cfg['data_hashes']['split_sha256'].startswith('aa8f7504'),
                               row0_equals_wd1_row0=metric_rows_equal(run['rows'][0], src1['rows'][0]),
                               data_equal=all(np.array_equal(d[k], src1['d'][k]) for k in d))
                r['y2']['PASS'] = bool(all(v for k, v in r['y2'].items() if k != 'split'))
                zeros = [np.zeros_like(x) for x in w0]
                rp = replay(run, w0, zeros, zeros, 0, 10, wd)
                r['y3'] = dict(init_age=cfg['init_age'], resume=cfg['resume'], update10_rel=rp['checkpoints'][10],
                               rows=summarise(rp))
                r['y3']['PASS'] = bool(cfg['init_age'] == 50000 and cfg['resume'] is None and rp['checkpoints'][10] <= 1e-12)
                if (seed, wd) in ((30001, .3), (30002, .1)):
                    rp = replay(run, w0, zeros, zeros, 0, 300, wd)
                    r['y4'] = summarise(rp)
                    r['y4']['PASS'] = bool(rp['rows_acc_equal'] == rp['rows'] == rp['rows_within'] == 30
                                           and max(rp['checkpoints'].values()) <= TOL_W)
            # Y5: my common accuracy and sigma-margin at every published checkpoint against the lane's table
            lane_rows = {int(x['step']): x for x in lane_ck if x['run_id'] == rid}
            worst = 0.
            for s, path in sorted(run['ck'].items()):
                w = [state(path)[k] for k in 'UVW']
                acc, mg = sigma_margin_and_acc(w, d, sets)
                lr_ = lane_rows[s]
                worst = max(worst, abs(acc - float(lr_['common_acc'])), abs(mg - float(lr_['margin_N_sigma'])))
                res['y5_rows'] += 1
            r['y5'] = dict(checkpoints=len(run['ck']), lane_rows=len(lane_rows), same_steps=sorted(lane_rows) == sorted(run['ck']),
                           worst_abs=worst)
            res['y5_worst'] = max(res['y5_worst'], worst)
            # Y6: timing from metrics.jsonl
            acc = {s: own(row, d) for s, row in run['rows'].items()}
            ls_ = lane_sum[rid]
            y6 = dict(own_update0=acc[0], lane_own_update0=float(ls_['own_update0']))
            for name, theta in (('update0', acc[0]), ('fixed010', .10)):
                peak, t, rc = decline(acc, theta)
                y6[name] = dict(peak_step=peak, peak=acc[peak], T=t if t is not None else f'>{max(acc)}', recross=rc,
                                E_at_T=-t*np.log1p(-LR*wd) if t is not None else None,
                                lane=dict(peak_step=int(ls_['peak_step']), T=ls_[f'T_{name}'], recross=int(ls_[f'recross_{name}'])))
                y6[name]['equal'] = bool(peak == int(ls_['peak_step']) and str(y6[name]['T']) == ls_[f'T_{name}']
                                         and rc == int(ls_[f'recross_{name}']))
            r['y6'] = y6
            summ[(seed, wd)] = (y6, rid)
            res['runs'][rid] = r
    # Y6: r and its classification, from my crossing times
    for (seed, wd), (y6, rid) in summ.items():
        if wd in (0., 1.):
            continue
        for name in ('update0', 'fixed010'):
            e_ref = summ[(seed, 1.)][0][name]['E_at_T']
            pred = e_ref/(-np.log1p(-LR*wd))
            t = y6[name]['T']
            cens = not isinstance(t, int)
            rr = (20000 if cens else t)/pred
            verdict = 'slower (censored)' if cens else 'slower' if rr > 1.5 else 'faster' if rr < 1/1.5 else 'exposure clock'
            ld = lane_dec[(rid, name)]
            y6[name]['decision'] = dict(T_pred=pred, r=rr, verdict=verdict, lane_r=ld['r'], lane_verdict=ld['verdict'],
                                        r_equal=(not cens and abs(rr - float(ld['r'])) <= 1e-12), verdict_equal=verdict == ld['verdict'])
    # Y7: exposure is log of the shrinkage that experiment.adam applies when wd acts on all of w (M4; Z1 a)
    t = np.arange(0, 20001, 1000)
    res['y7'] = {f'wd{wd:g}': float(np.max(np.abs(-t*np.log1p(-LR*wd) + t*np.log(1 - LR*wd)))) for wd in (.1, .3, 1.)}
    runs = res['runs'].values()
    res['PASS'] = dict(
        y1=all(r['y1']['PASS'] for r in runs if 'y1' in r), y2=all(r['y2']['PASS'] for r in runs if 'y2' in r),
        y3=all(r['y3']['PASS'] for r in runs if 'y3' in r), y4=all(r['y4']['PASS'] for r in runs if 'y4' in r),
        y5=bool(res['y5_worst'] <= 1e-9 and all(r['y5']['same_steps'] for r in runs)),
        y6=all(r['y6'][n]['equal'] and r['y6'][n].get('decision', dict(r_equal=True, verdict_equal=True))['verdict_equal']
               and r['y6'][n].get('decision', dict(r_equal=True))['r_equal'] for r in runs for n in ('update0', 'fixed010')))
    return res


# ---------- D13 (Y1'-Y4') ----------

def d13(out_dir):
    reg = registry(DY/'registry.jsonl')
    lane = {r['run_id']: r for r in csv.DictReader(open(DY/'figures/offswitch/offswitch_readings.csv'))}
    res = dict(runs={})
    for s in (300, 1000):
        for seed in SEEDS:
            rid = f'e4-dy-filt-wd1to0-u{s}-w128-i{seed}-a1'
            run, src1, src0 = load(DY/'runs'/rid), load(E3/f'e3-me-filt-w128-i{seed}-a1'), load(E3/f'e3-me-filt-wd0-w128-i{seed}-a1')
            cfg, plan = run['cfg'], PLAN_RUNS[rid]
            ev = reg.get(rid, [])
            resume = ROOT/cfg['resume']
            r = dict(seed=seed, s=s)
            r['y1'] = dict(registry=[e['status'] for e in ev], completed=[e.get('completed_steps') for e in ev if e['status'] == 'complete'],
                           manifest_completed=run['man']['completed_steps'], resume=cfg['resume'],
                           resume_is_source=resume == src1['ck'][s], resume_sha=sha(resume),
                           resume_sha_recorded=cfg['resume_sha256'] == cfg['expect_init_sha256'] == sha(resume) == plan.get('resume_sha256', plan.get('expect_init_sha256')),
                           init_weights=cfg['init_weights'], wd=cfg['wd'], steps=cfg['steps'],
                           config_differences_vs_source=config_diff(cfg, src1['cfg']),
                           shared=shared_code(cfg['runtime']['git_commit']), dirty=run['man']['producing_tree_dirty'],
                           decay_flags=[cfg.get(k) for k in ('decay_projection', 'decay_on')])
            y1 = r['y1']
            allowed = BOOK | {'wd', 'resume', 'resume_sha256', 'expect_init_sha256', 'init_rule', 'init_weights_sha256', 'init_age',
                              'init_weights', 'decay_projection', 'decay_on', 'decay_projection_sha256', 'expect_decay_projection_sha256'}
            y1['PASS'] = bool(y1['registry'] == ['running', 'complete'] and y1['completed'] == [20000] and y1['manifest_completed'] == 20000
                              and y1['resume_is_source'] and y1['resume_sha_recorded'] and cfg['wd'] == 0 and cfg['steps'] == 20000
                              and cfg['init_weights'] is None and set(y1['config_differences_vs_source']) <= allowed
                              and (y1['shared']['equals_72cfb5d'] or (y1['shared']['equals_pin_8108d8b'] and y1['decay_flags'] == [None, None]))
                              and not y1['dirty'])
            src = state(src1['ck'][s])
            a_, b_ = run['rows'][s], src1['rows'][s]
            r['y2'] = dict(first_row=min(run['rows']), row_s_equals_source=metric_rows_equal(a_, b_),
                           row_s_differing_fields=sorted(k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k)),
                           row_s_optimizer_update=a_.get('optimizer_update'),
                           row_s_equal_except_optimizer_update=metric_rows_equal({**a_, 'optimizer_update': 0}, {**b_, 'optimizer_update': 0}),
                           checkpoint_at_s=s in run['ck'],
                           checkpoint_s_bitwise=(all(np.array_equal(state(run['ck'][s])[k], src[k]) for k in src) if s in run['ck'] else None),
                           data_equal=all(np.array_equal(run['d'][k], src1['d'][k]) for k in run['d']))
            r['y2']['PASS'] = bool(r['y2']['first_row'] == s and r['y2']['row_s_equals_source'] and r['y2']['data_equal']
                                   and r['y2']['checkpoint_s_bitwise'] in (True, None))
            first = min(t for t in run['ck'] if t > s)
            w, m, v = [src[k] for k in 'UVW'], [src['m'+k] for k in 'UVW'], [src['v'+k] for k in 'UVW']
            carried = replay(run, w, m, v, s, first, 0.)
            reset = replay(run, w, m, v, s, first, 0., count0=s)
            r['y3'] = dict(source_step=int(src['step']), first_checkpoint=first, carried=summarise(carried), reset=summarise(reset))
            r['y3']['PASS'] = bool(carried['rows'] == carried['rows_acc_equal'] == carried['rows_within']
                                   and carried['checkpoints'][first] <= TOL_W)
            r['y3']['control_informative'] = bool(reset['checkpoints'][first] > TOL_W)
            # Y4': the reading
            A = lambda rr, t: own(rr['rows'][t], rr['d'])
            out = {}
            for W in (500, 1000, 2000):
                a1s, a1w, a0s, a0w, aoff = A(src1, s), A(src1, s + W), A(src0, s), A(src0, s + W), A(run, s + W)
                d1, d0, doff = a1w - a1s, a0w - a0s, aoff - a1s
                out[W] = dict(A1_s=a1s, A1_sW=a1w, A0_s=a0s, A0_sW=a0w, Aoff_sW=aoff, d1=d1, d0=d0, doff=doff, f=(doff - d0)/(d1 - d0))
            f = out[1000]['f']
            band = 'decay must stay on' if f <= 1/3 else 'sustains itself' if f >= 2/3 else 'partial'
            a1s = out[1000]['A1_s']
            after = {t: A(run, t) for t in run['rows'] if t > s}
            rec = [t for t in sorted(after) if after[t] >= a1s + .05]
            below = [t for t in sorted(after) if after[t] < .10]
            cks = sorted(set(run['ck']) & set(src1['ck']) & set(src0['ck']))
            near = min((t for t in cks if t > s), key=lambda t: abs(t - (s + 1000)))
            sets = ladder_sets(run['d']['a'], run['d']['b'], tr30, tr50, COMMON, P)
            ca = {name: sigma_margin_and_acc([state(rr['ck'][t])[k] for k in 'UVW'], rr['d'], sets)[0]
                  for name, rr, t in (('off', run, near), ('one', src1, near), ('zero', src0, near), ('one_s', src1, s), ('zero_s', src0, s))}
            f_common = ((ca['off'] - ca['one_s']) - (ca['zero'] - ca['zero_s']))/((ca['one'] - ca['one_s']) - (ca['zero'] - ca['zero_s']))
            r['y4'] = dict(windows={str(W): v for W, v in out.items()}, f=f, band=band, recovery_steps=rec[:5], recovery=bool(rec),
                           A_end=A(run, 20000), A1_end=A(src1, 20000), A0_end=A(src0, 20000),
                           T010_after_switch=below[0] if below else '>20000',
                           f_common=dict(step=near, W=near - s, f=f_common, acc=ca))
            ln = lane[rid]
            o = out[1000]
            gaps = {q: abs(float(ln[q]) - v) for q, v in (('A1_s', o['A1_s']), ('Aoff_sW', o['Aoff_sW']), ('A1_sW', o['A1_sW']),
                                                          ('A0_s', o['A0_s']), ('A0_sW', o['A0_sW']), ('f', f))}
            r['y4']['lane'] = dict(row={q: ln[q] for q in ('f', 'reading', 'recovery', 'T_010_after', 'own_end', 'max_after', 'max_after_step')},
                                   worst_gap=max(gaps.values()), gaps=gaps)
            r['y4']['lane_equal'] = bool(max(gaps.values()) <= 1e-12)
            res['runs'][rid] = r
    res['PASS'] = {k: all(r[k]['PASS'] for r in res['runs'].values()) for k in ('y1', 'y2', 'y3')}
    res['y2_state_and_measures_equal'] = all(r['y2']['row_s_equal_except_optimizer_update'] and r['y2']['checkpoint_s_bitwise']
                                             for r in res['runs'].values())
    res['y3_controls_informative'] = all(r['y3']['control_informative'] for r in res['runs'].values())
    return res


# ---------- D15 runs (X1-X5) ----------

def d15(out_dir):
    reg = registry(ME/'registry.jsonl')
    res = dict(runs={}, baseline={})
    sets = None
    for seed in SEEDS:
        rep1 = load(ME/f'runs/e4-me-filt-wd1-replay-w128-i{seed}-a1')
        rep0 = load(ME/f'runs/e4-me-filt-wd0-replay-w128-i{seed}-a1')
        w0 = start(seed)
        zeros = [np.zeros_like(x) for x in w0]
        base = replay(rep1, w0, zeros, zeros, 0, 5000, 1.)
        res['baseline'][seed] = summarise(base)
        base_late = max([base['worst_row_rel_late']] + list(base['checkpoints'].values()))
        if sets is None:
            sets = ladder_sets(rep1['d']['a'], rep1['d']['b'], tr30, tr50, COMMON, P)
        with np.load(proj_file(seed)) as z:
            mine = my_projection(z['k'], z['live'])
        for arm, on in (('xref', 'rest'), ('onlyref', 'reference')):
            rid = f'e4-me-filt-wd1-{arm}-w128-i{seed}-a1'
            if not (ME/'runs'/rid).exists():
                res['runs'][rid] = 'not published'
                continue
            run = load(ME/'runs'/rid)
            cfg, ev = run['cfg'], reg.get(rid, [])
            r = dict(seed=seed, arm=arm)
            r['x1'] = dict(registry=[e['status'] for e in ev], completed=[e.get('completed_steps') for e in ev if e['status'] == 'complete'],
                           decay_on=cfg['decay_on'], projection=cfg['decay_projection'],
                           projection_sha=cfg['decay_projection_sha256'] == sha(ROOT/cfg['decay_projection']) == PLAN_RUNS[rid]['decay_projection_sha256'],
                           config_differences_vs_replay=config_diff(cfg, rep1['cfg']), shared=shared_code(cfg['runtime']['git_commit']),
                           dirty=run['man']['producing_tree_dirty'])
            x1 = r['x1']
            x1['PASS'] = bool(x1['registry'] == ['running', 'complete'] and x1['completed'] == [5000] and cfg['decay_on'] == on
                              and Path(cfg['decay_projection']).name == proj_file(seed).name and x1['projection_sha']
                              and set(x1['config_differences_vs_replay']) <= BOOK | {'decay_projection', 'decay_on', 'decay_projection_sha256',
                                                                                     'expect_decay_projection_sha256'}
                              and x1['shared']['equals_pin_8108d8b'] and not x1['dirty'])
            s0 = state(run['ck'][0])
            r['x2'] = dict(step0_equals_start=all(np.array_equal(s0[k], w0[i]) for i, k in enumerate('UVW')),
                           row0_equals_replay=metric_rows_equal(run['rows'][0], rep1['rows'][0]))
            r['x2']['PASS'] = all(r['x2'].values())
            D = mine if on == 'reference' else rest_of(mine)
            rp = replay(run, w0, zeros, zeros, 0, 5000, 1., D)
            r['x3'] = summarise(rp)
            late = max([rp['worst_row_rel_late']] + [e for t, e in rp['checkpoints'].items() if t > 300])
            r['x3']['late_gap_vs_baseline'] = dict(arm=late, baseline=base_late, ratio=late/base_late if base_late else None)
            r['x3']['PASS_early'] = bool(rp['worst_row_rel_early'] <= TOL_REL and max(e for t, e in rp['checkpoints'].items() if t <= 300) <= TOL_W)
            # X5 values, and the training damage and size of the arm against the replay
            rows = {}
            for t in sorted(set(run['ck']) & set(rep1['ck'])):
                wa, wb = [state(run['ck'][t])[k] for k in 'UVW'], [state(rep1['ck'][t])[k] for k in 'UVW']
                rows[t] = dict(acc=sigma_margin_and_acc(wa, run['d'], sets)[0], acc_replay=sigma_margin_and_acc(wb, rep1['d'], sets)[0],
                               rel_diff_vs_replay={k: rel(wa[i], wb[i]) for i, k in enumerate('UVW')},
                               train=run['rows'][t]['train'], train_ce=run['rows'][t]['objective']['train']['cross_entropy'],
                               decay_norm=(run['rows'][t]['optimizer_update'] or {}).get('decay_norm'),
                               decay_norm_replay=(rep1['rows'][t]['optimizer_update'] or {}).get('decay_norm'))
            fx = {}
            for t in (2000, 5000):
                a_arm = rows[t]['acc'] if t in rows else None
                a1 = sigma_margin_and_acc([state(rep1['ck'][t])[k] for k in 'UVW'], rep1['d'], sets)[0]
                a0 = sigma_margin_and_acc([state(rep0['ck'][t])[k] for k in 'UVW'], rep0['d'], sets)[0]
                fx[t] = dict(acc_arm=a_arm, acc_wd1=a1, acc_wd0=a0, f=(a_arm - a1)/(a0 - a1) if a_arm is not None else None)
            ln = next(x for x in csv.DictReader(open(ME/'results/intervention.csv')) if x['arm'] == arm and int(x['init_seed']) == seed)
            gaps = {f'{q}_{t}': abs(float(ln[f'{q}_{t}']) - fx[t][k]) for t in (2000, 5000)
                    for q, k in (('acc', 'acc_arm'), ('acc_wd1', 'acc_wd1'), ('acc_wd0', 'acc_wd0'), ('f', 'f'))}
            r['x5'] = dict(f={str(t): v for t, v in fx.items()}, states={str(t): v for t, v in rows.items()},
                           lane_worst_gap=max(gaps.values()), lane_equal_to_6_decimals=bool(max(gaps.values()) <= 5e-7))
            res['runs'][rid] = r
    return res


def main():
    mode, args = sys.argv[1], sys.argv[2:]
    out_dir = Path(args[-1]); out_dir.mkdir(parents=True, exist_ok=True)
    res = dict(patch=lambda: patch(out_dir), noflag=lambda: noflag(Path(args[0])), d6=lambda: d6(out_dir),
               d13=lambda: d13(out_dir), d15=lambda: d15(out_dir))[mode]()
    (out_dir/f'dynamics-{mode}.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps(res.get('PASS', res) if mode != 'patch' else {k: v for k, v in res.items() if k != 'z1a'} | {'z1a_PASS': res['z1a_PASS']},
                     indent=1, default=str)[:6000])


if __name__ == '__main__':
    main()
