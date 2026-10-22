"""Reviewer's independent checks of research/regimes/common/ (REVIEW.md, section 0).

These do not rerun test_nets.py. The loss is rewritten here from its definition
(per-head mean cross-entropy over labeled rows, weighted sum) and every parameter
entry is checked against central differences of that rewrite, not of nets.py.
Adam is a separate loop. The runner is driven with a throwaway plan in a temp dir.

uv run --locked python research/regimes/review/check_common.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'research/regimes/common'))
import nets  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''))
    if not ok:
        FAIL.append(name)


def my_loss(p, X, targets, weights, layers, heads):
    """Independent rewrite: explicit layer list, log-softmax per row, row masks."""
    H = X
    for i in range(1, layers+1):
        H = np.maximum(0., H.dot(p['W%d' % i]) + p['b%d' % i])
    total = 0.
    for h in heads:
        if h not in targets:
            continue
        y = np.asarray(targets[h]); mask = (y >= 0).astype(float)
        if mask.sum() == 0:
            continue
        z = H.dot(p['W_' + h]) + p['b_' + h]
        logp = z - np.log(np.sum(np.exp(z - z.max(1)[:, None]), 1))[:, None] - z.max(1)[:, None]
        ce = -logp[np.arange(len(y)), np.where(y >= 0, y, 0)]
        total += weights.get(h, 1.) * float((ce*mask).sum() / mask.sum())
    return total


def preacts(p, X, layers):
    H, out = X, []
    for i in range(1, layers+1):
        a = H @ p[f'W{i}'] + p[f'b{i}']; out.append(a); H = np.maximum(a, 0)
    return out


def gradients():
    # Three hidden layers, three heads, rows unlabeled per head, one head fully
    # unlabeled, one row unlabeled everywhere, non-unit head weights.
    r = np.random.default_rng(55300)
    n, d = 17, 7
    X = r.integers(0, 2, (n, d)).astype(float)
    heads = dict(final=8, step1=8, aux=2)
    p = nets.init(d, (6, 5, 4), heads, seed=55301)
    for k in ('b1', 'b2', 'b3'):  # off the ReLU kink (all-zero rows, zero biases)
        p[k] = r.normal(0, .3, p[k].shape)
    t = dict(final=r.integers(0, 8, n), step1=r.integers(0, 8, n), aux=-np.ones(n, int))
    t['final'][[2, 5, 11]] = -1; t['step1'][[0, 5, 9, 10]] = -1
    t['final'][16] = t['step1'][16] = -1                     # row 16 unlabeled everywhere
    w = dict(final=1., step1=.37)
    margin = min(np.abs(a).min() for a in preacts(p, X, 3))
    check('preactivations off the kink for finite differences', margin > 1e-4, f'min |preact| = {margin:.2e}')
    total, per_head, g = nets.loss_grad(p, X, t, w)
    mine = my_loss(p, X, t, w, 3, heads)
    check('loss value equals independent rewrite', abs(total-mine) < 1e-12, f'{total:.15f} vs {mine:.15f}')
    check('fully unlabeled head is absent from per-head losses', 'aux' not in per_head, str(sorted(per_head)))
    worst, eps = 0., 1e-6
    for k in p:
        for ix in np.ndindex(p[k].shape):
            old = p[k][ix]
            p[k][ix] = old+eps; lp = my_loss(p, X, t, w, 3, heads)
            p[k][ix] = old-eps; lm = my_loss(p, X, t, w, 3, heads)
            p[k][ix] = old
            fd = (lp-lm)/(2*eps)
            worst = max(worst, abs(fd-g[k][ix]) / max(1e-6, abs(fd)+abs(g[k][ix])))
    check('every gradient entry matches central differences of the rewrite', worst < 1e-5,
          f'{sum(v.size for v in p.values())} entries, worst relative error {worst:.1e}')
    check('fully unlabeled head gets exactly zero gradient', not (np.any(g['W_aux']) or np.any(g['b_aux'])))
    X2 = X.copy(); X2[16] = 1 - X2[16]
    g2 = nets.loss_grad(p, X2, t, w)[2]
    check('a row unlabeled for every head has no effect on any gradient',
          all(np.array_equal(g[k], g2[k]) for k in p))
    q = nets.init(d, (), dict(final=8), seed=1)                 # depth 0 edge case
    _, _, gq = nets.loss_grad(q, X, dict(final=t['final']))
    lq = lambda: my_loss(q, X, dict(final=t['final']), {}, 0, ['final'])
    q['W_final'][3, 2] += eps; lp = lq(); q['W_final'][3, 2] -= 2*eps; lm = lq(); q['W_final'][3, 2] += eps
    check('no-hidden-layer network gradient', abs((lp-lm)/(2*eps) - gq['W_final'][3, 2]) < 1e-7)


def my_adam(p, g, st, lr, wd, names, b1=.9, b2=.999, eps=1e-8):
    st['t'] += 1
    for k in names:
        st['m'][k] = b1*st['m'][k] + (1-b1)*g[k]
        st['v'][k] = b2*st['v'][k] + (1-b2)*g[k]*g[k]
        mh = st['m'][k] / (1-b1**st['t']); vh = st['v'][k] / (1-b2**st['t'])
        p[k] = p[k] - lr*wd*p[k] - lr*mh/(np.sqrt(vh)+eps)


def adam():
    r = np.random.default_rng(55302)
    X = r.integers(0, 2, (20, 5)).astype(float)
    t = dict(A=r.integers(0, 2, 20), B=r.integers(0, 3, 20))
    p = nets.init(5, (6, 6), dict(A=2, B=3), seed=55303)
    ref = {k: v.copy() for k, v in p.items()}
    st = nets.adam_state(p); rs = dict(t=0, m={k: 0.*v for k, v in p.items()}, v={k: 0.*v for k, v in p.items()})
    trunk = ['W1', 'b1', 'W2', 'b2']; head_b = ['W_B', 'b_B']
    schedule = [None]*3 + [head_b]*4 + [trunk+head_b]*3   # all, then frozen trunk+A, then unfreeze trunk
    for i, names in enumerate(schedule):
        if i == 3:
            a_head = {k: (p[k].copy(), st['m'][k].copy(), st['v'][k].copy()) for k in ('W_A', 'b_A')}
        g = nets.loss_grad(p, X, t)[2]; gr = nets.loss_grad(ref, X, t)[2]
        nets.adam_step(p, g, st, .01, .2, trainable=names)
        my_adam(ref, gr, rs, .01, .2, list(p) if names is None else names)
    worst = max(float(np.max(np.abs(p[k]-ref[k]))) for k in p)
    check('Adam + decoupled decay + freeze/unfreeze schedule matches a separate loop', worst < 1e-13, f'max |diff| {worst:.1e}')
    check('A head (frozen after step 3): weights and moments bitwise unchanged, no decay',
          all(np.array_equal(x, y) for k, old in a_head.items() for x, y in zip(old, (p[k], st['m'][k], st['v'][k]))))
    # Reset semantics.
    fresh = nets.adam_state(p)
    aliased = any(fresh['m'][k] is st['m'][k] or np.shares_memory(fresh['m'][k], p[k]) or
                  np.shares_memory(fresh['v'][k], p[k]) for k in p)
    check('adam_state is a full reset (t=0, zero moments, no aliasing of weights or old state)',
          fresh['t'] == 0 and not aliased and not any(np.any(x) for x in (*fresh['m'].values(), *fresh['v'].values())))
    a = {k: v.copy() for k, v in p.items()}; b = {k: v.copy() for k, v in p.items()}
    sa = nets.adam_state(a); sb = fresh
    for _ in range(3):
        nets.adam_step(a, nets.loss_grad(a, X, t)[2], sa, .01)
        nets.adam_step(b, nets.loss_grad(b, X, t)[2], sb, .01)
    check('reset state continues bitwise like a state built on the copied weights',
          all(np.array_equal(a[k], b[k]) for k in p))
    # Semantics that lanes must know. (1) A head left out of `targets` but still
    # trainable keeps moving on carried momentum and decays with wd.
    c = {k: v.copy() for k, v in p.items()}; sc = nets.adam_state(c)
    for _ in range(5):
        nets.adam_step(c, nets.loss_grad(c, X, t)[2], sc, .01)
    before = c['W_A'].copy(); sc_reset = nets.adam_state(c); c2 = {k: v.copy() for k, v in c.items()}
    for _ in range(5):
        nets.adam_step(c, nets.loss_grad(c, X, dict(B=t['B']))[2], sc, .01)               # carried state, wd 0
        nets.adam_step(c2, nets.loss_grad(c2, X, dict(B=t['B']))[2], sc_reset, .01)       # reset, wd 0
    moved = float(np.max(np.abs(c['W_A']-before)))
    print(f'INFO head absent from targets, optimizer carried, wd=0: W_A moves {moved:.2e} in 5 steps')
    check('head absent from targets after a reset with wd=0 stays bitwise fixed', np.array_equal(c2['W_A'], before))
    # (2) Freeze then unfreeze without a reset: shared t makes bias correction
    # ~1 while the unfrozen moments start at 0, so the first steps are larger.
    g = {'w': np.array([1.])}
    for k_frozen in (0, 1000):
        s = dict(t=k_frozen, m={'w': np.zeros(1)}, v={'w': np.zeros(1)}); w = {'w': np.zeros(1)}
        steps = []
        for _ in range(20):
            old = w['w'][0]; nets.adam_step(w, g, s, 1.); steps.append(abs(w['w'][0]-old))
        print(f'INFO step sizes / lr after unfreezing at t={k_frozen}: first {steps[0]:.2f}, max of 20 {max(steps):.2f}')


def checkpoints():
    r = np.random.default_rng(55304)
    X = r.integers(0, 2, (12, 4)).astype(float); t = dict(A=r.integers(0, 2, 12))
    p = nets.init(4, (5,), dict(A=2), seed=55305); st = nets.adam_state(p)
    for _ in range(7):
        nets.adam_step(p, nets.loss_grad(p, X, t)[2], st, .01, .1)
    with tempfile.TemporaryDirectory() as d:
        f1, f2 = Path(d)/'a.npz', Path(d)/'b.npz'
        nets.save(f1, p, st, phase='A', step=7, seeds=dict(init=55305))
        q, sq, meta = nets.load(f1)
        same = (set(q) == set(p) and all(q[k].dtype == p[k].dtype and np.array_equal(q[k], p[k]) for k in p)
                and all(np.array_equal(sq[x][k], st[x][k]) for x in 'mv' for k in p) and sq['t'] == 7)
        check('save/load round-trips weights, both moments, t, dtypes and nested metadata',
              same and meta == dict(phase='A', step=7, seeds=dict(init=55305)))
        for _ in range(5):
            nets.adam_step(p, nets.loss_grad(p, X, t)[2], st, .01, .1)
            nets.adam_step(q, nets.loss_grad(q, X, t)[2], sq, .01, .1)
        check('loaded state continues bitwise', all(np.array_equal(p[k], q[k]) for k in p))
        time.sleep(2.1)  # zip member timestamps have 2 s resolution
        nets.save(f2, *nets.load(f1)[:2], **nets.load(f1)[2])
        h1, h2 = (hashlib.sha256(f.read_bytes()).hexdigest() for f in (f1, f2))
        check('identical contents saved 2 s apart give the same file SHA-256 (zip entries carry a fixed date)',
              h1 == h2)


def runner():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        plan = dict(runs=[
            dict(run_id='ok', argv=['-c', 'import sys,pathlib; pathlib.Path(sys.argv[1]).mkdir()', '{out}'], timeout=30),
            dict(run_id='bad', argv=['-c', 'import sys; sys.exit(3)'], timeout=30),
            dict(run_id='slow', argv=['-c', 'import time; print("started", flush=True); time.sleep(30)'], timeout=1),
            dict(run_id='exists', argv=['-c', 'pass'], timeout=30)])
        (d/'plan.json').write_text(json.dumps(plan)); (d/'root').mkdir(); (d/'root'/'exists').mkdir()
        cmd = [sys.executable, str(ROOT/'research/regimes/common/run_jobs.py'), '--plan', str(d/'plan.json'),
               '--root', str(d/'root'), '--registry', str(d/'reg.jsonl'), '--session-label', 'r1-rev-check']
        start = time.perf_counter()
        out = subprocess.run(cmd + ['--ids', 'ok', 'bad', 'slow'], capture_output=True, text=True)
        wall = time.perf_counter() - start
        ev = [json.loads(x) for x in (d/'reg.jsonl').read_text().splitlines()]
        term = {e['run_id']: e for e in ev if e['status'] != 'running'}
        check('runner: complete / failed (exit code) / capped (timeout) recorded once each',
              {k: v['status'] for k, v in term.items()} == dict(ok='complete', bad='failed', slow='capped')
              and len(ev) == 6 and term['bad']['returncode'] == 3 and term['slow']['returncode'] is None,
              f'runner wall {wall:.1f} s')
        check('runner: capped job keeps its log', 'started' in (d/'root'/'slow.log').read_text())
        for ids, why in ((['ok'], 'registry entry'), (['exists'], 'existing output directory')):
            again = subprocess.run(cmd + ['--ids', *ids], capture_output=True, text=True)
            check(f'runner refuses a reused run ID ({why})', again.returncode != 0 and 'never reuse' in again.stderr)
        again = subprocess.run(cmd + ['--ids', 'nope'], capture_output=True, text=True)
        check('runner refuses an ID not in the plan', again.returncode != 0 and 'not in plan' in again.stderr)
        n = len((d/'reg.jsonl').read_text().splitlines())
        check('refused invocations append nothing to the registry', n == 6)


if __name__ == '__main__':
    for f in (gradients, adam, checkpoints, runner):
        print(f'--- {f.__name__}'); f()
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)
