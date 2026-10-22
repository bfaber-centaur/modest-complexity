"""c3 pin check of be41d16 (QUESTION.md section 1). The forward pass, loss and oracle are the
c2 reviewer's own (../../composition-c2/review/rev_pin.py, written from r1's tables); the
schedule is re-derived from DECISIONS D5. Code under test is imported read-only, and the test
suites run in their own processes.

uv run --locked python research/composition-c3/review/rev_c3pin.py RUN_ROOT
(RUN_ROOT holds this reviewer's c3-rev-* runs; runs/c3)
"""
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
C3, ROOT = HERE.parent, HERE.parents[2]
C2 = ROOT/'research'/'composition-c2'
sys.path[:0] = [str(C3/'common'), str(C2/'review')]
from rev_pin import FAIL, HELD, P, TRAIN, check, get, my_forward, my_loss, oracle_rows  # noqa: E402
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import c3correct  # noqa: E402
import c3train  # noqa: E402
import nets  # noqa: E402

SEEDS = (51200, 51201, 51202)
REC, R1 = C2/'recovered', ROOT/'research'/'regimes'
ADAM = [.9, .999, 1e-8]
R1_KEYS = ('step', 'train', 'held', 'held_final_pred')
C2_KEYS = R1_KEYS + ('held_subset',)


def my_w(t, S=None, D=1):
    """D5: before S all three weights are 1; the update at S + k uses 1 - (k+1)/D for the
    intermediate heads (0 from S + D - 1 on); the final weight is always 1."""
    if S is None or t < S:
        return [1., 1., 1.]
    f = (t - S + 1) / D
    x = 1 - f if f < 1 else 0.
    return [x, x, 1.]


def my_exposure(n, start, end, S=None, D=1):
    W = [my_w(t, S, D) for t in range(start, end)]
    return dict(updates_with_weight=[sum(w[k] != 0 for w in W) for k in range(3)],
                weighted_row_presentations=[n*sum(w[k] for w in W) for k in range(3)])


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def pick(rs, keys):
    return [{k: r[k] for k in keys if k in r} for r in rs]


def ckpt_equal(a, b):
    (pa, sa, _), (pb, sb, _) = nets.load(a), nets.load(b)
    return sa['t'] == sb['t'] and pa.keys() == pb.keys() and all(
        np.array_equal(x[k], y[k]) for x, y in ((pa, pb), (sa['m'], sb['m']), (sa['v'], sb['v'])) for k in x)


def git(*a):
    return subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True)


def recipe(readout='tied', **kw):
    return dict(dict(task_id='ops51000-ls51100', transition='tied', readout=readout, hidden=128, lr=1e-3, wd=.1, adam=ADAM), **kw)


def sched(S=None, D=1, before=(1., 1., 1.)):
    return dict(before=list(before), switch_step=S, after=None if S is None else [0., 0., 1.], taper=D)


def try_source(ckpt, cfg, rec, sch):
    """(accepted, detail) of c3train.check_source for this pairing."""
    _, st, meta = nets.load(ckpt)
    try:
        out = c3train.check_source(ckpt, nets.sha256_file(ckpt), cfg, meta, st['t'], rec, sch)
        return True, f"manifest_listed={out['manifest_listed']}, meta run_id={meta.get('run_id')}"
    except ValueError as e:
        return False, str(e).replace('--resume recipe check failed: ', '')[:160]


def main(run_root):
    run_root = Path(run_root)

    # P0. The pin.
    py = [f'research/composition-c3/common/{f}' for f in ('c3train.py', 'c3correct.py', 'test_c3.py')]
    check('P0 common/*.py at HEAD equals the pin be41d16', git('diff', '--quiet', 'be41d16', 'HEAD', '--', *py).returncode == 0)
    check("P0 c2's common/ at HEAD equals its pin b3e8132", git('diff', '--quiet', 'b3e8132', 'HEAD', '--', 'research/composition-c2/common/').returncode == 0)

    # P1. Test suites, each in its own process.
    for suite in ('research/composition-c3/common/test_c3.py', 'research/composition-c2/common/test_c2.py'):
        t0 = time.time()
        r = subprocess.run([sys.executable, '-m', 'unittest', '-v', suite], cwd=ROOT, capture_output=True, text=True)
        tail = r.stderr.strip().splitlines()
        check(f'P1 {suite} in its own process', r.returncode == 0, f'{tail[-3]} {tail[-1]}; {time.time() - t0:.1f} s')

    # P2. Schedule and exposure.
    for S in (1000, 10000):
        for D in (1, 2000):
            s = sched(S, D)
            bad = [t for t in range(50000) if c3train.weights_at(t, s) != my_w(t, S, D)]
            check(f'P2 weights_at equals D5 at every t < 50000, S={S} D={D}', not bad, f'first mismatch {bad[:1]}')
            if D > 1:
                check(f'P2 S={S} D={D}: inter weight at S is 1 - 1/D, at S+D-2 is 1/D, 0 from S+D-1; final exactly 1',
                      c3train.weights_at(S, s)[0] == 1 - 1/D and np.isclose(c3train.weights_at(S+D-2, s)[0], 1/D, rtol=1e-9, atol=0)
                      and c3train.weights_at(S+D-1, s)[0] == 0 and all(c3train.weights_at(t, s)[2] == 1. for t in range(S-2, S+D+2)))
            ex, mine = c3train.exposure(384, s, S, 50000), my_exposure(384, S, 50000, S, D)
            check(f'P2 exposure of the planned branch S={S} D={D} (this run)', ex['updates_with_weight'] == mine['updates_with_weight']
                  and np.allclose(ex['weighted_row_presentations'], mine['weighted_row_presentations'], rtol=1e-12),
                  f"{ex['updates_with_weight']} {ex['weighted_row_presentations']}")
    total = 3 * sum(50000 - S for S in (1000, 10000) for D in (1, 2000))
    check('P2 the 12 planned branches total 534,000 updates', total == 534000, str(total))

    # P3. Scheduled loss and gradient on the scaffolding sources, with my loss.
    tr = oracle_rows(TRAIN); ys = [tr['states'][:, t] for t in range(3)]
    rng = np.random.default_rng(73000)
    worst_l, worst_d = 0., 0.
    for s in SEEDS:
        for step in (1000, 10000):
            p = nets.load(REC/f'r1-comp-rnn-inter-s{s}-a1'/f'ckpt-{step}.npz')[0]
            u = {k: rng.normal(size=v.shape) for k, v in p.items()}
            nu = np.sqrt(sum(float(np.sum(x*x)) for x in u.values())); u = {k: x/nu for k, x in u.items()}
            for w in ([0., 0., 1.], [.5, .5, 1.], [1., 1., 1.]):
                total_, _, g = c2rnn.loss_grad(p, tr['s'], tr['ops'], ys, w)
                worst_l = max(worst_l, abs(total_ - my_loss(p, tr['s'], tr['ops'], ys, w)))
                e = 1e-5
                fd = (my_loss({k: p[k] + e*u[k] for k in p}, tr['s'], tr['ops'], ys, w)
                      - my_loss({k: p[k] - e*u[k] for k in p}, tr['s'], tr['ops'], ys, w)) / (2*e)
                an = sum(float(np.sum(g[k]*u[k])) for k in p)
                worst_d = max(worst_d, abs(fd - an) / max(abs(an), 1e-12))
    check('P3 c2rnn.loss_grad total equals my loss at w = 001, .5 .5 1, 111 (6 tied/inter states)', worst_l < 1e-12, f'max |diff| {worst_l:.1e}')
    check('P3 its gradient matches a central finite difference of my loss along a random direction', worst_d < 1e-6, f'max rel diff {worst_d:.1e}')

    # P4 / CX-P1. The resume recipe check, called directly (no training).
    inter = lambda s: R1/'composition'/'runs'/f'r1-comp-rnn-inter-s{s}-a1'/'config.json'
    final = lambda s: R1/'composition'/'runs'/f'r1-comp-rnn-final-s{s}-a1'/'config.json'
    untied = lambda s: R1/'review'/'runs'/f'r1-rev-comp-rnn-untied-s{s}-a1'/'config.json'
    tt20 = C2/'time'/'runs'/'c2-time-tt-inter-s51200-a1'
    cases = []
    for s in SEEDS:
        for step in (1000, 10000):
            ck = REC/f'r1-comp-rnn-inter-s{s}-a1'/f'ckpt-{step}.npz'
            for sch_, name in ((sched(), 'keep'), (sched(step), f'drop at {step}'), (sched(step, 2000), f'taper at {step}'), (sched(step + 500), f'drop at {step + 500}')):
                cases.append((f'inter s{s} ckpt-{step}, own config, {name}', True, ck, inter(s), recipe(), sch_))
    cases += [
        ('c2 time tt-inter s51200 ckpt-20000, own config, keep', True, tt20/'ckpt-20000.npz', tt20/'config.json', recipe(), sched()),
        ('c2 time tt-inter s51200 ckpt-20000, own config, drop at 20000', True, tt20/'ckpt-20000.npz', tt20/'config.json', recipe(), sched(20000)),
        ('per-step-readout s51200 ckpt-10000, own config', True, REC/'r1-rev-comp-rnn-untied-s51200-a1'/'ckpt-10000.npz', untied(51200), recipe('untied'), sched()),
        ('(a) inter s51200 ckpt-10000, s51201 config', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter(51201), recipe(), sched()),
        ('(a) inter s51202 ckpt-1000, s51200 config', False, REC/'r1-comp-rnn-inter-s51202-a1'/'ckpt-1000.npz', inter(51200), recipe(), sched()),
        ('(b) inter s51200 ckpt-10000, final config, weights 0 0 1', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', final(51200), recipe(), sched(before=(0., 0., 1.))),
        ('(b) final s51201 ckpt-10000, inter config, weights 1 1 1', False, REC/'r1-comp-rnn-final-s51201-a1'/'ckpt-10000.npz', inter(51201), recipe(), sched()),
        ('(c) inter s51200 ckpt-10000, switch at 1000', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(), sched(1000)),
        ('(c) inter s51200 ckpt-10000, taper from 9999', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(), sched(9999, 2000)),
        ('(d) lr 2e-3', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(lr=2e-3), sched()),
        ('(d) wd 0', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(wd=0.), sched()),
        ('(d) task ops51010-ls51110', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(task_id='ops51010-ls51110'), sched()),
        ('(d) width 64', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(hidden=64), sched()),
        ('(d) Adam eps 1e-7', False, REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(adam=[.9, .999, 1e-7]), sched()),
        ('(e) c2 time ckpt-20000, switch at 10000', False, tt20/'ckpt-20000.npz', tt20/'config.json', recipe(), sched(10000)),
        ('(f) per-step-readout s51200 ckpt-10000, tied/inter s51200 config', False, REC/'r1-rev-comp-rnn-untied-s51200-a1'/'ckpt-10000.npz', inter(51200), recipe(), sched()),
        ('(f) per-step-readout s51200 ckpt-10000, s51201 per-step config', False, REC/'r1-rev-comp-rnn-untied-s51200-a1'/'ckpt-10000.npz', untied(51201), recipe('untied'), sched()),
    ]
    for name, want, ck, cfg, rec, sch_ in cases:
        ok, detail = try_source(ck, cfg, rec, sch_)
        check(f"P4 {'accepted' if want else 'refused'}: {name}", ok == want, detail)
    metas = {d.name: nets.load(d/f)[2].get('run_id') for d in REC.iterdir() if d.is_dir() for f in ('ckpt-10000.npz',) if (d/f).exists()}
    check('CX-P1 every recovered ckpt-10000 carries its own run ID in its meta (the binding when no MANIFEST.json lists it)',
          all(v == k for k, v in metas.items()), str({k: v for k, v in metas.items() if v != k}))
    check('CX-P1 the per-step-readout run directories have no MANIFEST.json (only MANIFEST.sha256), so the run ID is their only binding',
          all(not (untied(s).parent/'MANIFEST.json').exists() and (untied(s).parent/'MANIFEST.sha256').exists() for s in SEEDS))

    # P5. c3correct on the six correction bases, all 512 length-3 rows, depth 2.
    al = oracle_rows(list(product(range(4), repeat=3)))
    is_held = np.array([tuple(q) in set(HELD) for q in al['ops'].tolist()])
    acc = {}
    for arm in ('r1-rev-comp-rnn-untied', 'r1-comp-rnn-final'):
        for s in SEEDS:
            p = c3correct.frozen(nets.load(REC/f'{arm}-s{s}-a1'/'ckpt-10000.npz')[0])
            h0 = c3correct.params_sha256(p)
            Hm, Zm = my_forward(p, al['s'], al['ops'], [1, 2, 3])
            out = c3correct.corrected_forward(p, al['s'], al['ops'], 2)
            d_id = max(float(np.abs(a - b).max()) for a, b in zip(out['logits'] + out['H'], Zm + Hm))
            Ht = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[0][2]
            key = tr['s']*16 + tr['ops'][:, 0]*4 + tr['ops'][:, 1]
            _, first = np.unique(key, return_index=True)  # one row per unique depth-2 state
            C = np.stack([Ht[first][tr['states'][first, 1] == y].mean(0) for y in range(8)])
            sym = al['states'][:, 1]
            cm = c3correct.corrected_forward(p, al['s'], al['ops'], 2, correct=lambda h: C[sym])
            h3 = np.maximum(C[sym] @ get(p, 'A', 3) + np.eye(4)[al['ops'][:, 2]] @ get(p, 'B', 3) + get(p, 'bh', 3), 0)
            z3 = h3 @ get(p, 'V', 3) + get(p, 'bv', 3)
            d_cm = float(np.abs(cm['logits'][2] - z3).max())
            same12 = all(np.array_equal(a, b) for a, b in zip(cm['logits'][:2], out['logits'][:2]))
            acc[(arm, s)] = float(np.mean(cm['logits'][2][is_held].argmax(1) == al['states'][is_held, 2]))
            check(f'P5 {arm} s{s}: identity equals my forward; class-mean logits equal my V3 continuation; steps 1-2 untouched',
                  d_id < 1e-12 and d_cm < 1e-12 and same12, f'identity {d_id:.1e}, class mean {d_cm:.1e}, held-out class-mean acc {acc[(arm, s)]:.3f}')
            try:
                p['Ws'][0, 0] = 0.; wrote = True
            except ValueError:
                wrote = False
            try:
                nets.adam_step(p, {k: np.ones_like(v) for k, v in p.items()}, nets.adam_state(p), 1e-3, .1); stepped = True
            except ValueError:
                stepped = False
            q = {k: v.copy() for k, v in p.items()}; q['bv'] = q['bv'].copy(); q['bv'][0] = np.nextafter(q['bv'][0], np.inf)
            check(f'P5 {arm} s{s}: frozen refuses a write and an Adam step; hash unchanged by the forwards, changed by one ulp',
                  not wrote and not stepped and c3correct.params_sha256(p) == h0 and c3correct.params_sha256(q) != h0)
    u = [acc[('r1-rev-comp-rnn-untied', s)] for s in SEEDS]; f = [acc[('r1-comp-rnn-final', s)] for s in SEEDS]
    check('P5 class-mean held-out accuracy reproduces c2 O4: per-step readout 1.00, tied/final 0.88 / 0.88 / 0.97',
          all(x == 1. for x in u) and [round(x, 2) for x in f] == [.88, .88, .97], f'{u} {[round(x, 3) for x in f]}')
    try:
        c3correct.corrected_forward(nets.load(REC/'r1-rev-comp-rnn-untied-s51200-a1'/'ckpt-10000.npz')[0], al['s'][:8], np.zeros((8, 5), int), 2)
        raised = False
    except ValueError:
        raised = True
    check('P5 a per-step-readout model at length 5 without a step map is refused', raised)

    # P6. Oracle at any length.
    T = c2task.load()
    check("P6 c2task's original draw and split equal r1's tables", np.array_equal(T['P'], np.array(P))
          and sorted(map(tuple, T['held'])) == HELD and sorted(map(tuple, T['train'])) == sorted(TRAIN))
    r = np.random.default_rng(73001)
    for L in range(1, 8):
        progs = list(product(range(4), repeat=L))
        if L > 5:
            progs = [progs[i] for i in r.choice(len(progs), 3000, replace=False)]
        a, b = c2task.rows(T['P'], progs), oracle_rows(progs)
        check(f'P6 c2task.rows equals my oracle at L = {L} ({len(progs)} programs)', all(np.array_equal(a[k], b[k]) for k in ('s', 'ops', 'states')))

    # P7. The coordinator's verification and calibration runs.
    co = C3/'coordinator'/'runs'
    v = rows(co/'c3-co-verify-keep-s51200-a1'/'metrics.jsonl')
    pub = {r_['step']: r_ for r_ in pick(rows(R1/'composition'/'runs'/'r1-comp-rnn-inter-s51200-a1'/'metrics.jsonl'), R1_KEYS)}
    check('P7 verification: 41 rows 1000-3000, each equal to r1 published on every r1 key', len(v) == 41 and [r_['step'] for r_ in v] == list(range(1000, 3001, 50))
          and all(pick([r_], R1_KEYS)[0] == pub[r_['step']] for r_ in v), f'{len(v)} rows')
    cfg = json.loads((co/'c3-co-verify-keep-s51200-a1'/'config.json').read_text())
    src = cfg['resumed_from']['source']
    check('P7 verification: weights all 1 1 1; source manifest-listed, run ID r1-comp-rnn-inter-s51200-a1; resumed at 1000',
          all(r_['step_weights'] == [1., 1., 1.] for r_ in v) and src['manifest_listed'] and src['run_id'] == 'r1-comp-rnn-inter-s51200-a1'
          and cfg['resumed_from']['step'] == 1000)
    for name, S, D in (('c3-co-calib-keep-s900200-a1', None, 1), ('c3-co-calib-taper-s900200-a1', 500, 1000)):
        rs, cf = rows(co/name/'metrics.jsonl'), json.loads((co/name/'config.json').read_text())
        rt = json.loads((co/name/'runtime.json').read_text())
        mine = my_exposure(384, 0, 2000, S, D)
        check(f'P7 {name}: logged weights equal mine at all {len(rs)} rows; exposure equals mine; runtime complete',
              all(r_['step_weights'] == my_w(r_['step'], S, D) for r_ in rs) and rt['status'] == 'complete'
              and cf['exposure']['labels_from_init']['updates_with_weight'] == mine['updates_with_weight']
              and np.allclose(cf['exposure']['labels_from_init']['weighted_row_presentations'], mine['weighted_row_presentations'], rtol=1e-12),
              str(cf['exposure']['labels_from_init']))
    for d in co.iterdir():
        man = json.loads((d/'MANIFEST.json').read_text())
        check(f'P7 {d.name}: every committed file matches its MANIFEST.json hash',
              all(nets.sha256_file(d/e['file']) == e['sha256'] for e in man if (d/e['file']).exists()), f"{sum((d/e['file']).exists() for e in man)} of {len(man)} files present")
    reg = [json.loads(x) for x in (C3/'coordinator'/'registry.jsonl').read_text().splitlines()]
    check('P7 coordinator registry: each run has one complete record and nothing failed or capped',
          sorted(r_['run_id'] for r_ in reg if r_['status'] == 'complete') == sorted(d.name for d in co.iterdir())
          and {r_['status'] for r_ in reg} == {'running', 'complete'})

    # P8. Run IDs.
    ids = {}
    for f_ in sorted(ROOT.glob('research/**/plan.json')):
        d_ = json.loads(f_.read_text())
        for e in (d_.get('runs', []) if isinstance(d_, dict) else []):
            ids.setdefault(e['run_id'], []).append(str(f_.relative_to(ROOT)))
    dup = {k: v for k, v in ids.items() if len(v) > 1}
    check(f'P8 run IDs unique across {sum(1 for _ in ROOT.glob("research/**/plan.json"))} plan.json files ({len(ids)} IDs)', not dup, str(dup)[:300])

    # P9. My verification runs.
    mine = rows(run_root/'c3-rev-verify-keep10k-s51202-a1'/'metrics.jsonl')
    ref = [r_ for r_ in rows(C2/'time'/'runs'/'c2-time-tt-inter-s51202-a1'/'metrics.jsonl') if r_['step'] <= 11000]
    check('P9 keep from 10k: all 21 rows equal c2-time-tt-inter-s51202-a1 on r1 keys and held_subset; diag on every row but the last',
          len(mine) == 21 == len(ref) and pick(mine, C2_KEYS) == pick(ref, C2_KEYS)
          and all(a['diag'] == b['diag'] for a, b in zip(mine[:-1], ref[:-1])))
    cf = json.loads((run_root/'c3-rev-verify-keep10k-s51202-a1'/'config.json').read_text())['resumed_from']['source']
    check('P9 keep from 10k: source manifest-listed, run ID r1-comp-rnn-inter-s51202-a1', cf['manifest_listed'] and cf['run_id'] == 'r1-comp-rnn-inter-s51202-a1')
    a, b = run_root/'c3-rev-drop-s51200-a1', run_root/'c3-rev-c2final-s51200-a1'
    ra, rb = rows(a/'metrics.jsonl'), rows(b/'metrics.jsonl')
    check('P9 drop at 10k equals c2train final-only from the same state: ckpt-10300 (weights, m, v, step) and every row incl. diag',
          ckpt_equal(a/'ckpt-10300.npz', b/'ckpt-10300.npz') and pick(ra, C2_KEYS + ('diag',)) == pick(rb, C2_KEYS + ('diag',))
          and all(r_['step_weights'] == [0., 0., 1.] for r_ in ra))
    ex = json.loads((a/'config.json').read_text())['exposure']
    check('P9 drop exposure: this run [0, 0, 300] updates; from init [10000, 10000, 10300]',
          ex['labels_this_run']['updates_with_weight'] == [0, 0, 300] and ex['labels_from_init']['updates_with_weight'] == [10000, 10000, 10300])
    (p0, s0, _), (p1, s1, _) = nets.load(REC/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz'), nets.load(a/'ckpt-10300.npz')
    moved = {k: float(np.abs(p1[k] - p0[k]).max()) for k in p0}
    check('P9 drop: every parameter moves over 10000-10300 (no frozen head) and the Adam step count carries (10300)',
          all(x > 0 for x in moved.values()) and s1['t'] == 10300 and all(np.abs(s1['m'][k]).max() > 0 for k in s1['m']),
          ', '.join(f'{k} {x:.1e}' for k, x in moved.items()))
    w_, c_ = run_root/'c3-rev-taper-whole-s51201-a1', run_root/'c3-rev-taper-cont-s51201-a1'
    rw, rc = rows(w_/'metrics.jsonl'), rows(c_/'metrics.jsonl')
    keys = C2_KEYS + ('step_weights', 'diag')
    check('P9 mid-taper resume at 1800 continues the uninterrupted run bitwise: ckpt-2600 and rows 1800-2600 incl. weights and diag',
          ckpt_equal(w_/'ckpt-2600.npz', c_/'ckpt-2600.npz') and pick(rc, keys) == pick([r_ for r_ in rw if r_['step'] >= 1800], keys), f'{len(rc)} rows')
    check('P9 taper: logged weights equal D5 at every row of the whole run (S=1000, D=2000)', all(r_['step_weights'] == my_w(r_['step'], 1000, 2000) for r_ in rw),
          f"1800: {[r_['step_weights'] for r_ in rw if r_['step'] == 1800][0]}")
    pw = pick([r_ for r_ in rw if r_['step'] == 1000], R1_KEYS)[0]
    check('P9 taper whole: its row at 1000 equals r1 published (branch point untouched)',
          [pw] == [r_ for r_ in pick(rows(R1/'composition'/'runs'/'r1-comp-rnn-inter-s51201-a1'/'metrics.jsonl'), R1_KEYS) if r_['step'] == 1000])
    src = json.loads((c_/'config.json').read_text())['resumed_from']['source']
    check('P9 taper cont: source is the whole run (run ID bound, no MANIFEST.json in a local run directory)',
          src['run_id'] == 'c3-rev-taper-whole-s51201-a1' and not src['manifest_listed'])


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
