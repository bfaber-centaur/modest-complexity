"""Time lane checks (QUESTION.md 2.1 and Revisions 04:11; REVIEW.md section 2). Analysis only: lane
files are read from git at a lane commit; r1 records from research/regimes; recovered checkpoints
from recovered/. Thresholds are recomputed here from the joined trajectories; checkpoints are
rescored with rev_pin.py's forward pass; then counterexamples CX-T1 and CX-T2.

uv run --locked python research/composition-c2/review/rev_time.py LANE_COMMIT [F1_COMMIT]
"""
import csv
import hashlib
import io
import json
import subprocess
import sys
from itertools import product
import numpy as np

from rev_pin import C2, FAIL, HELD, ROOT, TRAIN, check, my_forward, oracle_rows
from rev_gen import notn_mask
import nets

SEEDS = (51200, 51201, 51202)
ARMS = {'tt-inter': ('composition', 'r1-comp-rnn-inter-s{}-a1', [1., 1., 1.], 'tied'),
        'tu-inter': ('review', 'r1-rev-comp-rnn-untied-s{}-a1', [1., 1., 1.], 'untied'),
        'tt-final': ('composition', 'r1-comp-rnn-final-s{}-a1', [0., 0., 1.], 'tied')}
LANE = 'research/composition-c2/time'
KEYS = ('train', 'held', 'held_final_pred')


def git(commit, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{path}'])


def crossing(steps, vals, need, last):
    """First logged t with vals >= need at t and the next four logged evaluations; None if censored."""
    for n in range(len(steps) - 4):
        if all(v >= need for v in vals[n:n+5]):
            return steps[n]
    return None


def adam_parts(p, st, lr=1e-3, wd=.1, b1=.9, b2=.999, eps=1e-8):
    """The last update's Adam-direction and decay parts from a saved state (moments after update t; decay
    uses the weights after it, a 1e-4 relative approximation), and sqrt(vhat) against eps."""
    t = st['t']; ks = sorted(p)
    g = np.concatenate([st['m'][k].ravel() for k in ks]) / (1 - b1**t)
    sv = np.sqrt(np.concatenate([st['v'][k].ravel() for k in ks]) / (1 - b2**t))
    w = np.concatenate([p[k].ravel() for k in ks])
    adam, decay = lr * g / (sv + eps), lr * wd * w
    return np.linalg.norm(adam), np.linalg.norm(decay), float(np.mean(sv < eps)), np.linalg.norm(adam) / np.linalg.norm(lr / eps * g)


def main(commit):
    rec = json.loads((C2/'recovered'/'RECOVERY.json').read_text())['files']
    sha10k = {e['run_id']: e['sha256'] for e in rec if e['file'] == 'ckpt-10000.npz'}
    he, tr, everything = oracle_rows(HELD), oracle_rows(TRAIN), oracle_rows(sorted(set(HELD) | set(TRAIN)))
    notn = notn_mask(he)
    recipe_ok, join_ok, man_ok, score_ok, n_ck = True, True, True, True, 0
    J, P = {}, {}
    for arm, (lane, name, w, ro) in ARMS.items():
        for s in SEEDS:
            run, src = f'c2-time-{arm}-s{s}-a1', name.format(s)
            base = f'{LANE}/runs/{run}'
            c = json.loads(git(commit, f'{base}/config.json'))
            recipe_ok &= (c['step_weights'] == w and c['readout'] == ro and c['transition'] == 'tied' and c['task_id'] == 'ops51000-ls51100'
                          and c['lr'] == 1e-3 and c['wd'] == .1 and c['hidden'] == 128 and c['updates'] == 50000 and c['resume_sha256'] == sha10k[src]
                          and c['resumed_from']['meta']['run_id'] == src and c['resumed_from']['step'] == 10000
                          and c['resumed_from']['meta']['condition'] == ('final' if w[0] == 0 else 'inter'))
            m = [json.loads(l) for l in git(commit, f'{base}/metrics.jsonl').decode().splitlines()]
            r1 = [json.loads(l) for l in (ROOT/'research'/'regimes'/lane/'runs'/src/'metrics.jsonl').read_text().splitlines()]
            join_ok &= m[0]['step'] == 10000 == r1[-1]['step'] and all(m[0][k] == r1[-1][k] for k in KEYS)
            steps = [r['step'] for r in r1] + [r['step'] for r in m[1:]]
            join_ok &= steps == list(range(0, 50001, 50))
            J[arm, s] = r1 + m[1:]
            man = json.loads(git(commit, f'{base}/MANIFEST.json'))
            for e in man:
                man_ok &= hashlib.sha256(git(commit, f'{base}/{e["file"]}')).hexdigest() == e['sha256']
            byt = {r['step']: r for r in m}
            for step in (20000, 30000, 40000, 50000):
                p, st, _ = nets.load(io.BytesIO(git(commit, f'{base}/ckpt-{step}.npz'))); n_ck += 1
                pred = my_forward(p, he['s'], he['ops'], [1, 2, 3])[1][-1].argmax(1)
                pt = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[1][-1].argmax(1) == tr['states'][:, -1]
                row = byt[step]
                score_ok &= (st['t'] == step and ''.join(map(str, pred)) == row['held_final_pred'] and pt.mean() == row['train']['final']['acc']
                             and (pred == he['states'][:, -1])[notn].mean() == row['held_subset']['notN'])
                P[arm, s, step] = (p, st)
    check('recipe against source (C1/D9): 9 continuations have their source\'s step weights, readout, task, lr, wd, width; resume hash = RECOVERY.json; '
          'resumed_from is the source run at step 10,000 with the matching condition', recipe_ok)
    check('joining: each continuation\'s step-10,000 row equals r1\'s published step-10,000 row (train, held, held_final_pred); joined steps 0-50,000 every 50',
          join_ok)
    check('MANIFEST.json of all 9 runs: every listed file matches its SHA-256', man_ok)
    check(f'{n_ck} committed checkpoints (20k-50k) rescored with my forward: step count, held-out string, train final, notN equal the logged row', score_ok)

    # Thresholds from the joined trajectories, against figures/time_threshold.csv.
    lane_t = {(r['arm'], int(r['seed']), r['measure']): r for r in csv.DictReader(io.StringIO(git(commit, f'{LANE}/figures/time_threshold.csv').decode()))}
    ok = True
    for (arm, s), rows in J.items():
        steps = [r['step'] for r in rows]
        notn_v = [r['held_subset']['notN'] if 'held_subset' in r else
                  (np.array([int(c) for c in r['held_final_pred']]) == he['states'][:, -1])[notn].mean() for r in rows]
        for meas, vals, need in (('held_final', [r['held']['final']['acc'] for r in rows], 122/128), ('held_notN', notn_v, 40/42)):
            t = crossing(steps, vals, need - 1e-12, 50000); L = lane_t[arm, s, meas]
            mine_min = min(vals[steps.index(t):]) if t is not None else None
            ok &= (str(t) if t is not None else '') == L['crossing_step'] and (t is None) == (L['censored'] == 'True')
            ok &= t is None or abs(mine_min - float(L['min_after_crossing'])) < 1e-12
            ok &= abs(vals[-1] - float(L['value_at_end'])) < 1e-12 and abs((vals[-1] - vals[steps.index(40000)]) - float(L['change_last_10k'])) < 1e-12
            print(f'INFO {arm} s{s} {meas}: crossing {t if t is not None else "censored (> 50,000)"}; at 10k {vals[steps.index(10000)]:.3f}; at 50k {vals[-1]:.3f}')
    check('thresholds recomputed (>= 122/128 or >= 40/42 at five consecutive evaluations, joined 0-50k): crossing, censoring, min after, end value and '
          'last-10k change equal time_threshold.csv', ok)

    # CX-T1 (tt-final): logit gap on rows wrong at 50k, 40k -> 50k; held-out CE; the late Adam step.
    print('CX-T1 tt-final: median gap (largest wrong minus correct logit) on held-out rows wrong at 50k, at 40k and 50k; held-out CE 40k -> 50k; '
          'last Adam step at 50k: |Adam part|, |decay part|, frac sqrt(vhat) < eps, |Adam part| / |(lr/eps) mhat|')
    shrink = 0
    for s in SEEDS:
        z = {}
        for step in (40000, 50000):
            p = P['tt-final', s, step][0]; z[step] = my_forward(p, he['s'], he['ops'], [1, 2, 3])[1][-1]
        y = he['states'][:, -1]; wrong = z[50000].argmax(1) != y
        gap = {k: np.median((np.where(np.eye(8)[y].astype(bool), -np.inf, v).max(1) - v[np.arange(len(y)), y])[wrong]) for k, v in z.items()}
        ce = {r['step']: r['held']['final']['ce'] for r in J['tt-final', s] if r['step'] in (40000, 45000, 50000)}
        a, dcy, frac, ratio = adam_parts(*P['tt-final', s, 50000])
        shrink += (gap[40000] - gap[50000]) / gap[40000] > .10
        print(f'  s{s}: gap {gap[40000]:.2f} -> {gap[50000]:.2f} ({(gap[50000] - gap[40000]) / gap[40000]:+.0%}); CE {ce[40000]:.3f} -> {ce[45000]:.3f} -> {ce[50000]:.3f}; '
              f'|Adam| {a:.2e}, |decay| {dcy:.2e}, frac below eps {frac:.3f}, Adam/GD-like {ratio:.3f}')
    check('CX-T1 prediction: the median wrong-row gap shrinks by > 10% from 40k to 50k in at least two seeds', shrink >= 2)
    parts = [adam_parts(*P['tt-final', s, 50000]) for s in SEEDS]
    check('CX-T1 prediction, first half: the Adam part of the last update is larger than the decay part, in every seed', all(a > d for a, d, _, _ in parts),
          'ratios ' + ', '.join(f'{a / d:.3f}' for a, d, _, _ in parts))
    check('CX-T1 prediction, second half: "so this is not R4\'s eps regime" (fewer than half the coordinates with sqrt(vhat) < eps)',
          all(fr < .5 for _, _, fr, _ in parts), 'fractions ' + ', '.join(f'{fr:.3f}' for _, _, fr, _ in parts))
    for arm in ('tt-inter', 'tu-inter'):
        print(f'INFO {arm} at 50k: fraction sqrt(vhat) < eps', [round(adam_parts(*P[arm, s, 50000])[2], 3) for s in SEEDS])

    # CX-T2 (tu-inter): readout distances and cross-readout, 10k and 50k.
    print('CX-T2 tu-inter: |V_k - V_3| / |V_3| for k = 1, 2; accuracy of V_v on h_t over all 512 rows (rows v, columns t)')
    lo = 0
    for s in SEEDS:
        for step in (10000, 50000):
            p = nets.load(C2/'recovered'/f'r1-rev-comp-rnn-untied-s{s}-a1'/'ckpt-10000.npz')[0] if step == 10000 else P['tu-inter', s, 50000][0]
            H = my_forward(p, everything['s'], everything['ops'], [1, 2, 3])[0]
            dist = [float(np.linalg.norm(p[f'V{k}'] - p['V3']) / np.linalg.norm(p['V3'])) for k in (1, 2)]
            acc = [[float(np.mean((H[t] @ p[f'V{v}'] + p[f'bv{v}']).argmax(1) == everything['states'][:, t-1])) for t in (1, 2, 3)] for v in (1, 2, 3)]
            if step == 50000:
                lo += acc[2][0] < .9
            print(f'  s{s} {step}: distances {dist[0]:.2f}, {dist[1]:.2f}; V1 {acc[0]}; V2 {acc[1]}; V3 {acc[2]}')
    check('CX-T2 prediction: at 50k, V3 on h1 stays below 0.9 in at least two seeds (catch-up without re-tied readouts)', lo >= 2)


def nbr_labels(he):
    """Per held-out row: labels of its one-op-slot training neighbours (same start), and of its op-3 neighbours."""
    from rev_gen import compose
    lab = {(s, *q): compose(q)[s] for q in TRAIN for s in range(8)}
    anyn, pre = [], []
    for s, q in zip(he['s'], he['ops']):
        s, q = int(s), [int(x) for x in q]
        nb = lambda j: {lab[k] for v in range(4) if v != q[j] for k in [(s, *q[:j], v, *q[j+1:])] if k in lab}
        anyn.append(nb(0) | nb(1) | nb(2)); pre.append(nb(2))
    return anyn, pre


def compat(pred, y, anyn, pre):
    w = np.where(pred != y)[0]
    if len(w) == 0:
        return 0, None, None, None, None
    f = lambda sets: (np.mean([pred[i] in sets[i] for i in w]), np.mean([len(sets[i] - {y[i]}) / 7 for i in w]))
    a, b = f(anyn), f(pre)
    return len(w), a[0], a[1], b[0], b[1]


def f1(commit):
    """Follow-up F1 (QUESTION.md Revisions, time F1): tt-final 50k -> 100k; checks 1-7, CX-T3, CX-T4."""
    he, tr, d6 = oracle_rows(HELD), oracle_rows(TRAIN), oracle_rows(list(product(range(4), repeat=6)))
    notn = notn_mask(he); y = he['states'][:, -1]; anyn, pre = nbr_labels(he)
    pstr = lambda r: np.array([int(c) for c in r['held_final_pred']])
    rec_ok, join_ok, man_ok, score_ok, J, P = True, True, True, True, {}, {}
    for s in SEEDS:
        base, ext = f'{LANE}/runs/c2-time-tt-final-s{s}-a1', f'{LANE}/runs/c2-time-tt-final-ext-s{s}-a1'
        c = json.loads(git(commit, f'{ext}/config.json'))
        sha50 = {e['file']: e['sha256'] for e in json.loads(git(commit, f'{base}/MANIFEST.json'))}['ckpt-50000.npz']
        rec_ok &= (c['step_weights'] == [0., 0., 1.] and c['readout'] == 'tied' and c['transition'] == 'tied' and c['task_id'] == 'ops51000-ls51100'
                   and c['init_seed'] == s and c['lr'] == 1e-3 and c['wd'] == .1 and c['hidden'] == 128 and c['updates'] == 100000
                   and c['resume_sha256'] == sha50 and hashlib.sha256(git(commit, f'{base}/ckpt-50000.npz')).hexdigest() == sha50
                   and c['resumed_from']['step'] == 50000 and c['resumed_from']['meta']['run_id'] == f'c2-time-tt-final-s{s}-a1')
        r1 = [json.loads(l) for l in (ROOT/'research'/'regimes'/'composition'/'runs'/f'r1-comp-rnn-final-s{s}-a1'/'metrics.jsonl').read_text().splitlines()]
        m50 = [json.loads(l) for l in git(commit, f'{base}/metrics.jsonl').decode().splitlines()]
        mx = [json.loads(l) for l in git(commit, f'{ext}/metrics.jsonl').decode().splitlines()]
        join_ok &= mx[0]['step'] == 50000 == m50[-1]['step'] and all(mx[0][k] == m50[-1][k] for k in KEYS + ('held_subset',))
        J[s] = r1 + m50[1:] + mx[1:]
        join_ok &= [r['step'] for r in J[s]] == list(range(0, 100001, 50))
        for e in json.loads(git(commit, f'{ext}/MANIFEST.json')):
            man_ok &= hashlib.sha256(git(commit, f'{ext}/{e["file"]}')).hexdigest() == e['sha256']
        by = {r['step']: r for r in mx}
        P[s, 50000] = nets.load(io.BytesIO(git(commit, f'{base}/ckpt-50000.npz')))
        for step in (75000, 100000):
            p, st, _ = P[s, step] = nets.load(io.BytesIO(git(commit, f'{ext}/ckpt-{step}.npz')))
            ph = my_forward(p, he['s'], he['ops'], [1, 2, 3])[1][-1].argmax(1); pt = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[1][-1].argmax(1)
            r = by[step]
            score_ok &= (st['t'] == step and ''.join(map(str, ph)) == r['held_final_pred'] and (pt == tr['states'][:, -1]).mean() == r['train']['final']['acc']
                         and (ph == y)[notn].mean() == r['held_subset']['notN'])
    check('F1 recipe (D9): 3 extensions are tied/tied, step weights 0 0 1, task, seed, lr, wd, width, 100,000 updates; resume hash = the 50k run\'s '
          'MANIFEST entry and the committed ckpt-50000 bytes; resumed_from = that run at step 50,000', rec_ok)
    check('F1 joining: each extension\'s step-50,000 row equals its 50k run\'s last row (incl. held_subset); joined steps 0-100,000 every 50', join_ok)
    check('F1 MANIFEST.json of the 3 extensions: every listed file matches its SHA-256', man_ok)
    check('F1 ckpt-75000 and ckpt-100000 (6) rescored with my forward: step count, held-out string, train final, notN equal the logged rows', score_ok)

    # Classification at 100k, against time_followup_threshold.csv.
    lane = {(int(r['seed']), r['measure']): r for r in csv.DictReader(io.StringIO(git(commit, f'{LANE}/figures/time_followup_threshold.csv').decode()))}
    ok, F = True, {}
    for s in SEEDS:
        rows = J[s]; steps = [r['step'] for r in rows]; by = dict(zip(steps, rows))
        nv = [(pstr(r) == y)[notn].mean() for r in rows]
        diff10 = int((pstr(by[100000]) != pstr(by[90000])).sum())
        last1k = [pstr(by[t]) for t in range(99000, 100001, 50)]
        ch1k = int(np.any([a != b for a, b in zip(last1k, last1k[1:])], axis=0).sum())
        ce = (by[100000]['held']['final']['ce'] - by[90000]['held']['final']['ce']) / by[90000]['held']['final']['ce']
        for meas, vals, need in (('held_final', [r['held']['final']['acc'] for r in rows], 122/128), ('held_notN', nv, 40/42)):
            tc = crossing(steps, vals, need - 1e-12, 100000); L = lane[s, meas]
            mn = min(vals[steps.index(tc):]) if tc is not None else None
            F[s, meas] = (tc, vals[-1])
            ok &= ((str(tc) if tc is not None else '') == L['crossing_step'] and (tc is None) == (L['censored'] == 'True')
                   and (tc is None or (abs(mn - float(L['min_after_crossing'])) < 1e-12 and (mn >= need - 1e-12) == (L['stays_above'] == 'True')))
                   and abs(vals[steps.index(10000)] - float(L['value_at_10k'])) < 1e-12 and abs(vals[-1] - float(L['value_at_end'])) < 1e-12
                   and abs(vals[-1] - vals[steps.index(90000)] - float(L['change_last_10k'])) < 1e-12 and diff10 == int(L['preds_differ_last_10k'])
                   and ch1k == int(L['pred_changes_last_1k']) and abs(ce - float(L['held_ce_rel_change_last_10k'])) < 1e-12)
        print(f'INFO tt-final s{s}: held 50k/75k/100k ' + ' / '.join(f'{by[t]["held"]["final"]["acc"]:.3f}' for t in (50000, 75000, 100000))
              + '; notN ' + ' / '.join(f'{nv[steps.index(t)]:.3f}' for t in (50000, 75000, 100000)) + f'; N at 100k {(pstr(by[100000]) == y)[~notn].mean():.3f}'
              + f'; crossing {F[s, "held_final"][0]}, notN crossing {F[s, "held_notN"][0]}; 90k->100k {vals[-1] - vals[steps.index(90000)]:+.3f} (notN), '
              f'held {by[100000]["held"]["final"]["acc"] - by[90000]["held"]["final"]["acc"]:+.3f}, CE {ce:+.0%}; {diff10} rows differ 90k vs 100k; {ch1k} rows change within 99k-100k')
    check('F1 classification recomputed over the joined 0-100k trajectories: crossing, censoring, min after and stays-above, values at 10k and 100k, '
          'last-10k change, rows differing 90k vs 100k, rows changing in the last 1,000 updates, held-out CE change equal time_followup_threshold.csv', ok)

    # Error compatibility, both CSVs.
    ok, rng, n = True, {'any': [], 'pre': [], 'anyu': [], 'preu': [], 'pre_all': []}, 0
    main_J = {}
    for arm, (lane_, name, w, ro) in ARMS.items():
        for s in SEEDS:
            r1 = [json.loads(l) for l in (ROOT/'research'/'regimes'/lane_/'runs'/name.format(s)/'metrics.jsonl').read_text().splitlines()]
            m = [json.loads(l) for l in git(commit, f'{LANE}/runs/c2-time-{arm}-s{s}-a1/metrics.jsonl').decode().splitlines()]
            main_J[arm, s] = {r['step']: r for r in r1 + m[1:]}
    for fname, src in (('time_error_compatibility.csv', main_J), ('time_followup_error_compatibility.csv', {('tt-final', s): {r['step']: r for r in J[s]} for s in SEEDS})):
        for r in csv.DictReader(io.StringIO(git(commit, f'{LANE}/figures/{fname}').decode())):
            k, st = (r['arm'], int(r['seed'])), int(r['step']); n += 1
            nw, a, au, b, bu = compat(pstr(src[k][st]), y, anyn, pre)
            ok &= nw == int(r['n_wrong']) and (nw == 0 or all(abs(u - float(v)) < 1e-12 for u, v in zip((a, au, b, bu), (r['any_nbr'], r['any_nbr_uniform'], r['same_prefix'], r['same_prefix_uniform']))))
            if nw and st >= 10000:
                rng['any'].append(a); rng['anyu'].append(au); rng['preu'].append(bu); rng['pre_all'].append(b)
                if st in (10000, 20000, 30000, 40000, 50000, 75000, 100000):
                    rng['pre'].append(b)
    check(f'error compatibility: all {n} rows of both CSVs (wrong count, any-neighbour and same-prefix shares and their uniform baselines) equal mine', ok)
    print(f'INFO compatibility from 10k on (rows with wrong answers): any {min(rng["any"]):.3f}-{max(rng["any"]):.3f} (uniform {min(rng["anyu"]):.2f}-{max(rng["anyu"]):.2f}); '
          f'same prefix at the checkpoint steps {min(rng["pre"]):.3f}-{max(rng["pre"]):.3f}, at every 1,000 {min(rng["pre_all"]):.3f}-{max(rng["pre_all"]):.3f} '
          f'(uniform {min(rng["preu"]):.2f}-{max(rng["preu"]):.2f})')
    at = [compat(pstr({r['step']: r for r in J[s]}[t]), y, anyn, pre)[3] for s in SEEDS for t in (75000, 100000)]
    check('NOTE: same-prefix compatibility 96-100% at 75k and 100k (tt-final)', abs(min(at) - .96) <= .005 + 1e-9 and max(at) == 1., f'{min(at):.3f}-{max(at):.3f}')

    # NOTE F1 numbers on the eps regime.
    med, frac, gain = [], True, []
    for s in SEEDS:
        rows = [r for r in J[s] if 'diag' in r and r['step'] >= 10000]
        te = next(r['step'] for r in rows if r['diag']['adam_sqrt_vhat']['median'] < 1e-8)
        frac &= all(r['diag']['adam_sqrt_vhat']['frac_below_eps'] == 1. for r in rows if r['step'] >= 29650)
        med.append([r for r in rows if r['step'] == 99950][0]['diag']['adam_sqrt_vhat']['median'])
        by = {r['step']: r for r in J[s]}; h = lambda t: by[t]['held']['final']['acc']
        gain.append((h(100000) - h(te)) / (h(100000) - h(10000)))
    check('NOTE: every coordinate below eps from 29,650 to 99,950 in tt-final; median sqrt(vhat) about 5-6e-10 at the end (99,950); '
          '75-86% of the 10k->100k held-out gain after the median crosses eps', frac and min(med) >= 4.5e-10 and max(med) < 6.5e-10 and abs(min(gain) - .75) <= .005 and abs(max(gain) - .86) <= .005,
          f'median {min(med):.1e}-{max(med):.1e}; gain after {min(gain):.3f}-{max(gain):.3f}')

    # Checkpoint distances, against time_followup_checkpoints.csv.
    ok = True
    for r in csv.DictReader(io.StringIO(git(commit, f'{LANE}/figures/time_followup_checkpoints.csv').decode())):
        s, st = int(r['seed']), int(r['step']); p = P[s, st][0]; vec = lambda q: np.concatenate([q[k].ravel() for k in sorted(q)])
        ok &= abs(np.linalg.norm(vec(p)) - float(r['param_norm'])) < 1e-9
        if st > 50000:
            ok &= abs(np.linalg.norm(vec(p) - vec(P[s, st - 25000][0])) - float(r['dist_prev'])) < 1e-9
    check('time_followup_checkpoints.csv: parameter norms and distances between checkpoints (50k, 75k, 100k) equal mine', ok)

    # CX-T3 and CX-T4.
    print('CX-T3 tt-final: median wrong-row gap (rows wrong at 100k) at 75k and 100k; held-out CE 75k -> 100k')
    shrink = 0
    for s in SEEDS:
        z = {t: my_forward(P[s, t][0], he['s'], he['ops'], [1, 2, 3])[1][-1] for t in (75000, 100000)}
        wrong = z[100000].argmax(1) != y
        gap = {t: np.median((np.where(np.eye(8)[y].astype(bool), -np.inf, v).max(1) - v[np.arange(len(y)), y])[wrong]) for t, v in z.items()}
        by = {r['step']: r for r in J[s]}
        shrink += (gap[75000] - gap[100000]) / gap[75000] > .1 and by[100000]['held']['final']['ce'] < by[75000]['held']['final']['ce']
        print(f'  s{s}: {int(wrong.sum())} wrong; gap {gap[75000]:.2f} -> {gap[100000]:.2f} ({(gap[100000] - gap[75000]) / gap[75000]:+.0%}); '
              f'CE {by[75000]["held"]["final"]["ce"]:.3f} -> {by[100000]["held"]["final"]["ce"]:.3f}')
    check('CX-T3 prediction: the gap shrinks > 10% from 75k to 100k and CE falls, in at least two seeds', shrink >= 2)
    print('CX-T4 tt-final: accuracy of the shared readout on h_t (= length-t accuracy) over all 32,768 length-6 rows, t = 1..6, at 50k and 100k')
    ok = True
    for s in SEEDS:
        for t_ in (50000, 100000):
            Z = my_forward(P[s, t_][0], d6['s'], d6['ops'], [1] * 6)[1]
            acc = [float((Z[t].argmax(1) == d6['states'][:, t]).mean()) for t in range(6)]
            if t_ == 100000:
                ok &= max(acc[0], acc[1]) < .5 and max(acc[3:]) < .3
            print(f'  s{s} {t_}: ' + ' '.join(f'{a:.3f}' for a in acc))
    check('CX-T4 prediction: at 100k the readout reads h1 and h2 below 0.5 and L = 4-6 stays below 0.3, in every seed', ok)


def note_m4(commit):
    """The time NOTE's remaining numbers (M4, e44349b), from logged rows and committed checkpoints."""
    he = oracle_rows(HELD); notn = notn_mask(he); y = he['states'][:, -1]
    pstr = lambda r: np.array([int(c) for c in r['held_final_pred']])
    M = {}
    for arm in ARMS:
        for s in SEEDS:
            M[arm, s] = {r['step']: r for r in map(json.loads, git(commit, f'{LANE}/runs/c2-time-{arm}-s{s}-a1/metrics.jsonl').decode().splitlines())}
    sub = lambda r, m: (pstr(r) == y)[m].mean()
    tu30 = [(sub(M['tu-inter', s][30000], ~notn), sub(M['tu-inter', s][30000], notn)) for s in SEEDS]
    tf50 = [sub(M['tt-final', s][50000], ~notn) for s in SEEDS]
    near = lambda v, a, b: abs(min(v) - a) <= .005 + 1e-9 and abs(max(v) - b) <= .005 + 1e-9
    check('NOTE: at 30k tu-inter is 0.98-1.00 on N and 0.67-0.93 on notN; tt-final at 50k is 0.78-0.91 on N',
          near([a for a, _ in tu30], .98, 1.) and near([b for _, b in tu30], .67, .93) and near(tf50, .78, .91),
          f'{min(a for a, _ in tu30):.3f}-{max(a for a, _ in tu30):.3f}, {min(b for _, b in tu30):.3f}-{max(b for _, b in tu30):.3f}; {min(tf50):.3f}-{max(tf50):.3f}')
    ch = [int((pstr(M['tt-final', s][50000]) != pstr(M['tt-final', s][40000])).sum()) for s in SEEDS]
    still = all(all((pstr(M[a, s][t]) == pstr(M[a, s][50000])).all() for t in range(49000, 50001, 50)) for a in ('tt-inter', 'tu-inter') for s in SEEDS)
    ce = [(M['tt-inter', s][50000]['held']['final']['ce'] - M['tt-inter', s][40000]['held']['final']['ce']) / M['tt-inter', s][40000]['held']['final']['ce'] for s in SEEDS]
    check('NOTE: 14-20 tt-final held-out predictions change 40k->50k; none in tt-inter or tu-inter over 49k-50k; tt-inter held-out CE falls 19-56% from 40k to 50k',
          min(ch) == 14 and max(ch) == 20 and still and near([-c for c in ce], .19, .56), f'{ch}; CE {[round(c, 3) for c in ce]}')
    dist, norms, peak = {}, [], []
    for arm in ARMS:
        for s in SEEDS:
            base = f'{LANE}/runs/c2-time-{arm}-s{s}-a1'
            v = {st: (lambda q: np.concatenate([q[k].ravel() for k in sorted(q)]))(nets.load(io.BytesIO(git(commit, f'{base}/ckpt-{st}.npz')))[0]) for st in (20000, 30000, 40000, 50000)}
            dist.setdefault(arm, []).append(np.linalg.norm(v[50000] - v[40000])); norms += [np.linalg.norm(v[k]) for k in (40000, 50000)]
            if arm == 'tt-inter' and s == 51200:
                peak = [np.linalg.norm(v[k]) for k in (20000, 30000, 40000, 50000)]
    rng = {a: (min(d), max(d)) for a, d in dist.items()}
    check('NOTE: distance ckpt-40000 -> ckpt-50000 is 2.6-3.4 (tt-inter), 5.4-6.1 (tu-inter), 4.4-4.6 (tt-final); parameter norms at 40k and 50k 25-35; tt-inter s51200 27.6 at 20k -> 25.5 at 50k',
          near(rng['tt-inter'], 2.6, 3.4) and near(rng['tu-inter'], 5.4, 6.1) and near(rng['tt-final'], 4.4, 4.6) and min(norms) >= 24.95 and max(norms) < 35.05
          and abs(peak[0] - 27.6) < .05 and abs(peak[-1] - 25.5) < .05,
          ', '.join(f'{a} {lo:.2f}-{hi:.2f}' for a, (lo, hi) in rng.items()) + f'; norms {min(norms):.1f}-{max(norms):.1f}; s51200 tt-inter ' + ' '.join(f'{x:.2f}' for x in peak))
    d = {(a, s): M[a, s][49950]['diag'] for a in ARMS for s in SEEDS}
    mg = {a: [d[a, s]['train_final_margin']['median'] for s in SEEDS] for a in ARMS}
    gn = [d[k]['grad_norm'] for k in d]; tce = [M[k][50000]['train']['final']['ce'] for k in M]
    print('INFO at 49,950: train margin median ' + '; '.join(f'{a} {min(v):.2f}-{max(v):.2f}' for a, v in mg.items())
          + f'; grad norm {min(gn):.1e}-{max(gn):.1e}; train final CE at 50k {min(tce):.1e}-{max(tce):.1e}')
    check('NOTE: margin median about 19.8 (tt-inter), 19.0 (tu-inter), 18.2-18.5 (tt-final); gradient norm 3.5-19e-8 at 50k',
          all(abs(x - 19.8) < .15 for x in mg['tt-inter']) and all(abs(x - 19.0) < .15 for x in mg['tu-inter']) and all(18.15 <= x <= 18.55 for x in mg['tt-final'])
          and abs(min(gn) - 3.5e-8) < .05e-8 and abs(max(gn) - 19e-8) < .5e-8, f'grad norm max {max(gn):.3e}')


if __name__ == '__main__':
    main(sys.argv[1])
    if len(sys.argv) > 2:
        f1(sys.argv[2])
        note_m4(sys.argv[2])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
