"""Time lane checks (QUESTION.md 2.1 and Revisions 04:11; REVIEW.md section 2). Analysis only: lane
files are read from git at a lane commit; r1 records from research/regimes; recovered checkpoints
from recovered/. Thresholds are recomputed here from the joined trajectories; checkpoints are
rescored with rev_pin.py's forward pass; then counterexamples CX-T1 and CX-T2.

uv run --locked python research/composition-c2/review/rev_time.py LANE_COMMIT
"""
import csv
import hashlib
import io
import json
import subprocess
import sys
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


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
