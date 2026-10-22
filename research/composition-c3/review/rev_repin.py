"""c3 re-pin check of 82cef77 (QUESTION.md section 4, DECISIONS D10: --reset-optimizer-at).
My Adam update is written from the recipe (moments, bias correction by the count, decoupled
decay p*(1 - lr*wd)); the gradient is the pinned c2rnn.loss_grad, which my pin check found
bitwise r1's. Code under test is imported read-only, and the test suites run in their own processes.

uv run --locked python research/composition-c3/review/rev_repin.py --out OUT --runs runs/c3
(--runs holds this reviewer's c3-rev-* runs)
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from rev_c3pin import C2, C3, R1, REC, ROOT, SEEDS, ckpt_equal, git, my_exposure, recipe, rows, sched  # noqa: E402
from rev_pin import FAIL, check  # noqa: E402
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import c3train  # noqa: E402
import nets  # noqa: E402

LR, WD, B1, B2, EPS = 1e-3, .1, .9, .999, 1e-8
KEYS = ('step', 'train', 'held', 'held_final_pred', 'held_subset', 'step_weights', 'diag')


def my_adam(p, g, m, v, k):
    """One update from the recipe, in a fresh dict: count k is this update's bias-correction count."""
    m = {n: B1*m[n] + (1 - B1)*g[n] for n in p}
    v = {n: B2*v[n] + (1 - B2)*g[n]**2 for n in p}
    return {n: p[n]*(1 - LR*WD) - LR*(m[n]/(1 - B1**k)) / (np.sqrt(v[n]/(1 - B2**k)) + EPS) for n in p}, m, v


def max_diff(a, b):
    return max(float(np.abs(a[n] - b[n]).max()) for n in a)


def try_source(ck, cfg, rec, sch, adam_t=None):
    _, st, meta = nets.load(ck)
    try:
        c3train.check_source(ck, nets.sha256_file(ck), cfg, meta, meta.get('step', st['t']), st['t'] if adam_t is None else adam_t, rec, sch)
        return True, ''
    except ValueError as e:
        return False, str(e).replace('--resume recipe check failed: ', '')[:200]


def p4_cases():
    """The 41 pairings of rev_c3pin.py P4 (27 accepted, 14 refused), unchanged."""
    inter = lambda s: R1/'composition'/'runs'/f'r1-comp-rnn-inter-s{s}-a1'/'config.json'
    final = lambda s: R1/'composition'/'runs'/f'r1-comp-rnn-final-s{s}-a1'/'config.json'
    untied = lambda s: R1/'review'/'runs'/f'r1-rev-comp-rnn-untied-s{s}-a1'/'config.json'
    tt20 = C2/'time'/'runs'/'c2-time-tt-inter-s51200-a1'
    ck = lambda s, step, arm='inter': REC/f'r1-comp-rnn-{arm}-s{s}-a1'/f'ckpt-{step}.npz'
    out = [(True, ck(s, step), inter(s), recipe(), sch_) for s in SEEDS for step in (1000, 10000)
           for sch_ in (sched(), sched(step), sched(step, 2000), sched(step + 500))]
    out += [(True, tt20/'ckpt-20000.npz', tt20/'config.json', recipe(), sched()),
            (True, tt20/'ckpt-20000.npz', tt20/'config.json', recipe(), sched(20000)),
            (True, REC/'r1-rev-comp-rnn-untied-s51200-a1'/'ckpt-10000.npz', untied(51200), recipe('untied'), sched()),
            (False, ck(51200, 10000), inter(51201), recipe(), sched()),
            (False, ck(51202, 1000), inter(51200), recipe(), sched()),
            (False, ck(51200, 10000), final(51200), recipe(), sched(before=(0., 0., 1.))),
            (False, ck(51201, 10000, 'final'), inter(51201), recipe(), sched()),
            (False, ck(51200, 10000), inter(51200), recipe(), sched(1000)),
            (False, ck(51200, 10000), inter(51200), recipe(), sched(9999, 2000)),
            (False, ck(51200, 10000), inter(51200), recipe(lr=2e-3), sched()),
            (False, ck(51200, 10000), inter(51200), recipe(wd=0.), sched()),
            (False, ck(51200, 10000), inter(51200), recipe(task_id='ops51010-ls51110'), sched()),
            (False, ck(51200, 10000), inter(51200), recipe(hidden=64), sched()),
            (False, ck(51200, 10000), inter(51200), recipe(adam=[.9, .999, 1e-7]), sched()),
            (False, tt20/'ckpt-20000.npz', tt20/'config.json', recipe(), sched(10000)),
            (False, REC/'r1-rev-comp-rnn-untied-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(), sched()),
            (False, REC/'r1-rev-comp-rnn-untied-s51200-a1'/'ckpt-10000.npz', untied(51201), recipe('untied'), sched())]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--runs', type=Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    R = a.runs
    summary = {}

    # RP0. Scope.
    changed = git('diff', '--name-only', 'be41d16', '82cef77', '--', 'research/composition-c3/common/').stdout.split()
    check('RP0 be41d16..82cef77 changes only c3train.py, test_c3.py and README.md in c3 common/',
          sorted(Path(f).name for f in changed) == ['README.md', 'c3train.py', 'test_c3.py'], str(changed))
    for rev, path in (('82cef77', 'research/composition-c3/common/'), ('be41d16', 'research/composition-c3/common/c3correct.py'),
                      ('b3e8132', 'research/composition-c2/common/'), ('be41d16', 'research/regimes/common/')):
        check(f'RP0 {path} at HEAD equals {rev}', git('diff', '--quiet', rev, 'HEAD', '--', path).returncode == 0)

    # RP1. Test suites, each in its own process.
    for suite, n in (('research/composition-c3/common/test_c3.py', 9), ('research/composition-c2/common/test_c2.py', None)):
        t0 = time.time()
        r = subprocess.run([sys.executable, '-m', 'unittest', '-v', suite], cwd=ROOT, capture_output=True, text=True)
        tail = r.stderr.strip().splitlines()
        ran = [x for x in tail if x.startswith('Ran ')]
        check(f'RP1 {suite} in its own process', r.returncode == 0 and (n is None or (ran and ran[0].startswith(f'Ran {n} tests '))),
              f'{ran} {tail[-1]}; {time.time() - t0:.1f} s')

    # RP2. Runs without the flag, bitwise against my be41d16 runs.
    for new, old in (('c3-rev-repin-drop-s51200-a1', 'c3-rev-drop-s51200-a1'), ('c3-rev-repin-taper-s51201-a1', 'c3-rev-taper-whole-s51201-a1')):
        cks = sorted(f.name for f in (R/old).glob('ckpt-*.npz'))
        same = cks == sorted(f.name for f in (R/new).glob('ckpt-*.npz')) and all(ckpt_equal(R/old/f, R/new/f) for f in cks)
        rn, ro = rows(R/new/'metrics.jsonl'), rows(R/old/'metrics.jsonl')
        check(f'RP2 {new} (82cef77) equals {old} (be41d16) bitwise: checkpoints {cks} and every metrics row incl. diag', same and rn == ro, f'{len(rn)} rows')

    # RP3. Old sources under the new signature.
    res = [(want, try_source(ck, cfg, rec, sch_)) for want, ck, cfg, rec, sch_ in p4_cases()]
    acc = sum(ok for _, (ok, _) in res)
    check("RP3 rev_c3pin's 41 P4 pairings give the same outcomes with the new check_source (27 accepted, 14 refused)",
          len(res) == 41 and all(want == ok for want, (ok, _) in res) and acc == 27, f'{acc} accepted; mismatches {[i for i, (w, (o, _)) in enumerate(res) if w != o]}')
    tw = R/'c3-rev-taper-whole-s51201-a1'
    ok1 = try_source(tw/'ckpt-1800.npz', tw/'config.json', recipe(), sched(1000, 2000))
    ok2 = try_source(tw/'ckpt-1800.npz', tw/'config.json', recipe(), dict(sched(1000, 2000), reset_at=1500))
    check('RP3 be41d16 taper ckpt-1800 (config without reset_at): accepted with its own flags, refused with a reset at 1500',
          ok1[0] and not ok2[0], f'{ok1[1]} | {ok2[1]}')

    # RP4. The reset, against my Adam from zero moments on the carried weights.
    T = c2task.load(); d = c2task.rows(T['P'], T['train']); ys = [d['states'][:, t] for t in range(3)]
    for run, s, w in (('c3-rev-reset-drop-s51200-a1', 51200, [0., 0., 1.]), ('c3-rev-reset-keep-s51201-a1', 51201, [1., 1., 1.])):
        p, st0, _ = nets.load(REC/f'r1-comp-rnn-inter-s{s}-a1'/'ckpt-1000.npz')
        if (R/run/'ckpt-1000.npz').exists():
            check(f'RP4 {run} ckpt-1000 equals the r1 source in weights, moments and count (1000): saved before the reset',
                  ckpt_equal(R/run/'ckpt-1000.npz', REC/f'r1-comp-rnn-inter-s{s}-a1'/'ckpt-1000.npz'))
        m = {n: np.zeros_like(x) for n, x in p.items()}; v = {n: np.zeros_like(x) for n, x in p.items()}
        worst, bitwise, counts = 0., True, []
        for k in (1, 2, 3):
            g = c2rnn.loss_grad(p, d['s'], d['ops'], ys, w)[2]
            q, m, v = my_adam(p, g, m, v, k)
            if k == 1:
                big = np.concatenate([(np.abs(g[n]) > 1e-4).ravel() for n in p])
                rel = np.concatenate([np.abs(np.abs(q[n] - p[n]*(1 - LR*WD)) / LR - 1).ravel() for n in p])
                first = (float(rel[big].max()), float(big.mean()))
            p = q
            pc, sc, mc = nets.load(R/run/f'ckpt-{1000 + k}.npz')
            counts.append((sc['t'], mc['step']))
            worst = max(worst, max_diff(p, pc), max_diff(m, sc['m']), max_diff(v, sc['v']))
            bitwise &= all(np.array_equal(x[n], y[n]) for x, y in ((p, pc), (m, sc['m']), (v, sc['v'])) for n in p)
        check(f'RP4 {run}: at 1000+k (k=1,2,3) the count is k, meta step 1000+k, and weights and moments equal my Adam from zero moments',
              counts == [(k, 1000 + k) for k in (1, 2, 3)] and worst <= 1e-15, f'counts {counts}; max |diff| {worst:.1e}; bitwise {bitwise}')
        check(f'RP4 {run}: in the update at 1000 every coordinate with |g| > 1e-4 moves by lr within 1e-4 relative (decay removed)',
              first[0] < 1e-4, f'max rel {first[0]:.1e} over {first[1]:.1%} of coordinates')

    # RP5. Continuation after R and at R.
    whole = R/'c3-rev-reset-drop-s51200-a1'; rw = rows(whole/'metrics.jsonl')
    for cont, at in (('c3-rev-reset-drop-cont1200-s51200-a1', 1200), ('c3-rev-reset-drop-cont1000-s51200-a1', 1000)):
        rc = rows(R/cont/'metrics.jsonl')
        sel = lambda rs: [{k: r_[k] for k in KEYS if k in r_} for r_ in rs]
        check(f'RP5 {cont}: ckpt-1400 and rows {at}-1400 (incl. diag) equal the uninterrupted reset branch bitwise',
              ckpt_equal(whole/'ckpt-1400.npz', R/cont/'ckpt-1400.npz') and sel(rc) == sel([r_ for r_ in rw if r_['step'] >= at]), f'{len(rc)} rows')

    # RP6. Refusals and acceptances, called directly.
    cfg = whole/'config.json'
    sr = lambda Rr: dict(sched(1000), reset_at=Rr)
    inter0 = R1/'composition'/'runs'/'r1-comp-rnn-inter-s51200-a1'/'config.json'
    keep_r = lambda Rr: dict(sched(), reset_at=Rr)
    cases = [('reset branch ckpt-1200, no reset', False, whole/'ckpt-1200.npz', cfg, sr(None), None),
             ('reset branch ckpt-1200, reset at 1100', False, whole/'ckpt-1200.npz', cfg, sr(1100), None),
             ('reset branch ckpt-1200, reset at 1300', False, whole/'ckpt-1200.npz', cfg, sr(1300), None),
             ('reset branch ckpt-1200, reset at 1000', True, whole/'ckpt-1200.npz', cfg, sr(1000), None),
             ('reset branch ckpt-1200, reset at 1000, count forged to 1200', False, whole/'ckpt-1200.npz', cfg, sr(1000), 1200),
             ('reset branch ckpt-1000 (pre-reset), no reset', True, whole/'ckpt-1000.npz', cfg, sr(None), None),
             ('r1 s51200 ckpt-1000, keep, reset at 500', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-1000.npz', inter0, keep_r(500), None),
             ('r1 s51200 ckpt-10000, keep, reset at 1000', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter0, keep_r(1000), None),
             ('r1 s51200 ckpt-10000, keep, reset at 10000', True, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter0, keep_r(10000), None),
             ('r1 s51200 ckpt-10000, keep, reset at 20000', True, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter0, keep_r(20000), None)]
    for name, want, ck, cf, sch_, forged in cases:
        ok, detail = try_source(ck, cf, recipe(), sch_, forged)
        check(f"RP6 {'accepted' if want else 'refused'}: {name}", ok == want, detail)

    # RP7. Records.
    c, ck_ = json.loads(cfg.read_text()), json.loads((R/'c3-rev-reset-keep-s51201-a1'/'config.json').read_text())
    ex = c['exposure']
    check('RP7 reset configs record schedule.reset_at = 1000 (drop and keep), and the drop exposure equals a drop at 1000 without a reset (mine)',
          c['schedule'].get('reset_at') == 1000 and ck_['schedule'].get('reset_at') == 1000
          and ex['labels_this_run']['updates_with_weight'] == [0, 0, 400] and ex['labels_from_init']['updates_with_weight'] == [1000, 1000, 1400]
          and ex['labels_from_init']['weighted_row_presentations'] == [float(x) for x in my_exposure(len(d['s']), 0, 1000)['weighted_row_presentations'][:2]] + [float(len(d['s'])*1400)],
          json.dumps(ex['labels_from_init']))

    # CX-R1. Step size after a reset versus carried, in-process, 100 updates from r1 tied/inter ckpt-1000.
    cx = {}
    for s in SEEDS:
        p0, st0, _ = nets.load(REC/f'r1-comp-rnn-inter-s{s}-a1'/'ckpt-1000.npz')
        N = sum(x.size for x in p0.values())
        for arm, w in (('keep', [1., 1., 1.]), ('drop', [0., 0., 1.])):
            for opt in ('carried', 'reset'):
                p = {n: x.copy() for n, x in p0.items()}
                if opt == 'carried':
                    m, v, k0 = st0['m'], st0['v'], st0['t']
                else:
                    m, v, k0 = {n: np.zeros_like(x) for n, x in p.items()}, {n: np.zeros_like(x) for n, x in p.items()}, 0
                rms = []
                for k in range(1, 101):
                    g = c2rnn.loss_grad(p, d['s'], d['ops'], ys, w)[2]
                    q, m, v = my_adam(p, g, m, v, k0 + k)
                    rms.append(float(np.sqrt(sum(float(np.sum((q[n] - p[n]*(1 - LR*WD))**2)) for n in p) / N)))
                    p = q
                cx[f'{arm} s{s} {opt}'] = rms
            r_ = [np.array(cx[f'{arm} s{s} {o}']) for o in ('reset', 'carried')]
            summary[f'{arm} s{s}'] = dict(ratio_mean=float(r_[0].mean() / r_[1].mean()), ratio_first=float(r_[0][0] / r_[1][0]),
                                          ratio_last10=float(r_[0][-10:].mean() / r_[1][-10:].mean()),
                                          rms_reset_mean=float(r_[0].mean()), rms_carried_mean=float(r_[1].mean()))
    for key, val in summary.items():
        print(f"     CX-R1 {key}: mean RMS move 1000-1099 reset {val['rms_reset_mean']:.2e} vs carried {val['rms_carried_mean']:.2e}; "
              f"ratio {val['ratio_mean']:.2f} (first update {val['ratio_first']:.2f}, last 10 {val['ratio_last10']:.2f})")
    check('CX-R1 reset/carried mean RMS move over 1000-1099 >= 1.5 in all six seed x {keep, drop} pairs',
          all(v_['ratio_mean'] >= 1.5 for v_ in summary.values()), f"min {min(v_['ratio_mean'] for v_ in summary.values()):.2f}")
    (a.out/'rev_repin.json').write_text(json.dumps(dict(summary=summary, rms=cx), indent=1))


if __name__ == '__main__':
    main()
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
