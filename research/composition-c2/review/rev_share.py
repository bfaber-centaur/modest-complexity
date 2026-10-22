"""Sharing lane checks (QUESTION.md 2.4 and Revisions 04:28, 04:38 and after; REVIEW.md section 5). Analysis only:
lane files are read from git at a lane commit; TT and TU are r1's runs (recovered checkpoints and
published metrics; TU is the reviewer's r1 arm). Rescoring uses rev_pin.py's forward pass and
rev_gen.py's neighbour recount, not the lane's code.

uv run --locked python research/composition-c2/review/rev_share.py LANE_COMMIT [FOLLOWUP_COMMIT]
"""
import csv
import io
import json
import subprocess
import sys
import numpy as np

from rev_pin import C2, FAIL, HELD, ROOT, TRAIN, check, my_forward, oracle_rows
from rev_gen import notn_mask
import c2rnn
import nets

SEEDS = (51200, 51201, 51202)
LANE = 'research/composition-c2/sharing/runs'
SCHEME = {'tt': (True, True), 'tu': (True, False), 'ut': (False, True), 'uu': (False, False)}
R1 = {'tt': ('composition', 'r1-comp-rnn-inter-s{}-a1'), 'tu': ('review', 'r1-rev-comp-rnn-untied-s{}-a1')}
KEYS = ('train', 'held', 'held_final_pred')


def git(commit, path):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{path}'])


def lane_metrics(commit, run):
    return [json.loads(l) for l in git(commit, f'{LANE}/{run}/metrics.jsonl').decode().splitlines()]


def r1_metrics(cell, seed):
    lane, name = R1[cell]
    return [json.loads(l) for l in (ROOT/'research'/'regimes'/lane/'runs'/name.format(seed)/'metrics.jsonl').read_text().splitlines()]


def main(commit):
    tr, he = oracle_rows(TRAIN), oracle_rows(HELD); notn = notn_mask(he)
    M = {(c, s): r1_metrics(c, s) for c in ('tt', 'tu') for s in SEEDS}
    M.update({(c, s): lane_metrics(commit, f'c2-share-{c}-s{s}-a1') for c in ('ut', 'uu') for s in SEEDS})

    # 1. Recipe of every lane run (D9), against what its ID and the M0 design say.
    ok, n = True, 0
    for kind in ('ut', 'uu', 'replay-tt', 'replay-tu'):
        for s in SEEDS:
            run = f'c2-share-{kind}-s{s}-a1'; c = json.loads(git(commit, f'{LANE}/{run}/config.json'))
            tt, tr_ = SCHEME[kind[-2:]]; n += 1
            ok &= (c['step_weights'] == [1., 1., 1.] and c['transition'] == ('tied' if tt else 'untied') and c['readout'] == ('tied' if tr_ else 'untied')
                   and c['lr'] == 1e-3 and c['wd'] == .1 and c['task_id'] == 'ops51000-ls51100' and c['init_seed'] == s and c['updates'] == 10000
                   and c['hidden'] == 128 and c['resume'] is None and c['scheme'] == dict(transition=tt, readout=tr_))
    check(f'recipe: all {n} lane runs (UT, UU, TT and TU replays) have step weights 1 1 1, their cell\'s scheme, lr 1e-3, wd 0.1, '
          f'the original task, the init seed in their ID, 10,000 updates, width 128, no resume', ok)

    # 2. Init identity: step-0 rows across cells per seed (and the replays) equal r1's.
    rep = {(c, s): lane_metrics(commit, f'c2-share-replay-{c}-s{s}-a1') for c in ('tt', 'tu') for s in SEEDS}
    ok = all(all(M[c, s][0][k] == M['tt', s][0][k] for k in KEYS) and M[c, s][0]['step'] == 0 for c in SCHEME for s in SEEDS)
    check('init identity: per seed, the step-0 rows (train, held, held_final_pred) of TT, TU, UT and UU are identical', ok)
    worst_rep = all(len(rep[c, s]) == len(M[c, s]) and all(all(a[k] == b[k] for k in ('step',) + KEYS) for a, b in zip(rep[c, s], M[c, s]))
                    for c in ('tt', 'tu') for s in SEEDS)
    check('replays: every logged row (201 per run) of the 6 TT/TU replays equals r1\'s published row in step, train, held, held_final_pred', worst_rep)

    # 3. Parameter counts; 4. gradient aggregation at init on the lane's seeds, full training data.
    ckpt = {}
    for c in SCHEME:
        for s in SEEDS:
            if c in R1:
                ckpt[c, s] = nets.load(C2/'recovered'/R1[c][1].format(s)/'ckpt-10000.npz')[0]
            else:
                ckpt[c, s] = nets.load(io.BytesIO(git(commit, f'{LANE}/c2-share-{c}-s{s}-a1/ckpt-10000.npz')))[0]
    counts = {c: {sum(v.size for v in ckpt[c, s].values()) for s in SEEDS} for c in SCHEME}
    check(f'parameter counts from the checkpoints: {counts}; every parameter is decayed (decoupled wd on all)',
          counts == {'tt': {19208}, 'tu': {21272}, 'ut': {53256}, 'uu': {55320}})
    ys = [tr['states'][:, t] for t in range(3)]; worst = 0.
    for s in SEEDS:
        g_t = c2rnn.loss_grad(c2rnn.init(128, s, True, True), tr['s'], tr['ops'], ys, [1, 1, 1])[2]
        for c in ('tu', 'ut', 'uu'):
            g = c2rnn.loss_grad(c2rnn.init(128, s, *SCHEME[c]), tr['s'], tr['ops'], ys, [1, 1, 1])[2]
            for k, v in g_t.items():
                tot = g[k] if k in g else g[f'{k}1'] + g[f'{k}2'] + g[f'{k}3']
                worst = max(worst, np.abs(tot - v).max() / max(np.abs(v).max(), 1e-300))
    check('gradient aggregation at init (seeds 51200-51202, all 384 training rows, weights 1 1 1): untied copies sum to the tied gradient',
          worst < 1e-12, f'worst relative diff {worst:.1e}')

    # 5. Endpoints from the checkpoints with my forward and my notN recount.
    ok, table = True, {}
    for (c, s), p in ckpt.items():
        ph = my_forward(p, he['s'], he['ops'], [1, 2, 3])[1][-1].argmax(1) == he['states'][:, -1]
        pt = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[1][-1].argmax(1) == tr['states'][:, -1]
        last = M[c, s][-1]
        ok &= last['step'] == 10000 and ph.mean() == last['held']['final']['acc'] and pt.mean() == last['train']['final']['acc']
        ok &= ''.join(map(str, my_forward(p, he['s'], he['ops'], [1, 2, 3])[1][-1].argmax(1))) == last['held_final_pred']
        if 'held_subset' in last:
            ok &= last['held_subset']['notN'] == ph[notn].mean() and last['held_subset']['N'] == ph[~notn].mean()
        table[c, s] = (ph.mean(), ph[~notn].mean(), ph[notn].mean(), pt.mean())
    check('12 ckpt-10000 rescored: held-out, train and the prediction string equal the last logged row; N/notN equal held_subset where logged', ok)
    for c in SCHEME:
        print(f'INFO {c.upper()} held | N | notN | train at 10k:', [tuple(round(float(x), 3) for x in table[c, s]) for s in SEEDS])

    # 6. CX-S1: sign stability over the last 2,000 updates.
    held = lambda c, s: {r['step']: r['held']['final']['acc'] for r in M[c, s] if r['step'] >= 8000}
    ok_uu, ok_int = True, True
    for s in SEEDS:
        h = {c: held(c, s) for c in SCHEME}
        steps = sorted(h['tt'])
        assert all(sorted(h[c]) == steps for c in SCHEME) and len(steps) == 41
        d_uu = [h['uu'][t] - h['tu'][t] for t in steps]
        inter = [(h['tt'][t] - h['tu'][t]) - (h['ut'][t] - h['uu'][t]) for t in steps]
        ok_uu &= min(d_uu) > 0; ok_int &= min(inter) > 0
        print(f'INFO s{s}, 41 evaluations 8,000-10,000: UU - TU min {min(d_uu):+.3f} max {max(d_uu):+.3f}, negative at {sum(x <= 0 for x in d_uu)}; '
              f'interaction min {min(inter):+.3f} max {max(inter):+.3f}; held-out change 8k->10k '
              + ', '.join(f'{c.upper()} {h[c][10000] - h[c][8000]:+.3f}' for c in SCHEME))
    check('CX-S1 prediction: UU - TU > 0 at every logged evaluation 8,000-10,000 in every seed', ok_uu)
    check('CX-S1 prediction: interaction (TT - TU) - (UT - UU) > 0 at every logged evaluation 8,000-10,000 in every seed', ok_int)



def wd0(commit):
    """Follow-up at wd 0 (QUESTION.md Revisions 04:38): recipes, pairing with the wd-0.1 runs, endpoints, CX-S2."""
    he = oracle_rows(HELD); notn = notn_mask(he); y = he['states'][:, -1]
    rec_ok, pair_ok, end_ok, flat, moving = True, True, True, True, 0
    ref = {(c, s): (r1_metrics(c, s) if c in R1 else lane_metrics(commit, f'c2-share-{c}-s{s}-a1')) for c in SCHEME for s in SEEDS}
    for c in SCHEME:
        for s in SEEDS:
            run = f'c2-share-wd0-{c}-s{s}-a1'; cfg = json.loads(git(commit, f'{LANE}/{run}/config.json')); tt, tr_ = SCHEME[c]
            rec_ok &= (cfg['wd'] == 0. and cfg['step_weights'] == [1., 1., 1.] and cfg['scheme'] == dict(transition=tt, readout=tr_) and cfg['lr'] == 1e-3
                       and cfg['task_id'] == 'ops51000-ls51100' and cfg['init_seed'] == s and cfg['updates'] == 10000 and cfg['hidden'] == 128 and cfg['resume'] is None)
            m = lane_metrics(commit, run)
            pair_ok &= all(m[0][k] == ref[c, s][0][k] for k in ('step',) + KEYS)
            last = m[-1]; ok = np.array([int(ch) for ch in last['held_final_pred']]) == y
            end_ok &= last['step'] == 10000 and ok.mean() == last['held']['final']['acc'] and ok[notn].mean() == last['held_subset']['notN']
            by = {r['step']: r for r in m}
            preds = [np.array([int(ch) for ch in by[t]['held_final_pred']]) for t in range(5000, 10001, 50)]
            churn = [int((a != b).sum()) for a, b in zip(preds, preds[1:])]
            dh = by[10000]['held']['final']['acc'] - by[5000]['held']['final']['acc']
            ce = (by[5000]['held']['final']['ce'], by[10000]['held']['final']['ce'])
            mg = (by[5000]['diag']['train_final_margin']['median'], by[9950]['diag']['train_final_margin']['median'])   # no diag in the 10k row
            flat &= abs(dh) <= .03; moving += mg[1] > 1.1 * mg[0] and ce[1] > ce[0]
            print(f'INFO wd0 {c.upper()} s{s}: held at 10k {ok.mean():.3f} (notN {ok[notn].mean():.3f}); 5k->10k held {dh:+.3f}, held CE {ce[0]:.2f}->{ce[1]:.2f}, '
                  f'train margin median 5k->9,950 {mg[0]:.1f}->{mg[1]:.1f}; changed held-out predictions per evaluation: mean {np.mean(churn):.2f}, max {max(churn)}')
    check('wd-0 recipes: 12 runs with wd 0 and otherwise the wd-0.1 recipe (step weights 1 1 1, cell scheme, lr, task, seed, 10,000 updates, width, no resume)', rec_ok)
    check('wd-0 pairing: each run\'s step-0 row equals its wd-0.1 counterpart\'s (r1\'s for TT/TU): same initial function', pair_ok)
    check('wd-0 endpoints: held-out and notN from the logged prediction string with my recount equal the last logged row', end_ok)
    check('CX-S2 prediction, first half: held-out changes by at most 0.03 from 5k to 10k in every run', flat)
    check(f'CX-S2 prediction, second half: train margin grows > 10% (5k->9,950, the last row with diag) and held-out CE rises 5k->10k (in {moving} of 12 runs; predicted all)', moving == 12)



def wd0_note(commit):
    """The follow-up section of the sharing NOTE, recomputed from the logged rows (QUESTION.md Revisions, after CX-S2)."""
    he = oracle_rows(HELD); notn = notn_mask(he); y = he['states'][:, -1]
    ok_s = lambda r: np.array([int(ch) for ch in r['held_final_pred']]) == y
    R = {(w, c, s): (lane_metrics(commit, f'c2-share-wd0-{c}-s{s}-a1') if w == 0 else
                     r1_metrics(c, s) if c in R1 else lane_metrics(commit, f'c2-share-{c}-s{s}-a1')) for w in (0, .1) for c in SCHEME for s in SEEDS}
    by = {k: {r['step']: r for r in v} for k, v in R.items()}
    H = {k: v[10000]['held']['final']['acc'] for k, v in by.items()}
    near = lambda a, b: abs(a - b) <= .005 + 1e-9
    note = {'tt': ((.398, .383, .453), (.048, .071, .024), (-.60, -.58, -.55)), 'tu': ((.297, .297, .320), (0, 0, .048), (-.34, -.23, -.20)),
            'ut': ((.336, .336, .305), (0, 0, 0), (-.50, -.34, -.32)), 'uu': ((.297, .312, .305), (.024, 0, 0), (-.47, -.30, -.40))}
    ok = all(abs(H[0, c, s] - note[c][0][n]) <= .0005 + 1e-9 and abs(ok_s(by[0, c, s][10000])[notn].mean() - note[c][1][n]) <= .0005 + 1e-9
             and near(H[0, c, s] - H[.1, c, s], note[c][2][n]) for c in SCHEME for n, s in enumerate(SEEDS))
    check('NOTE follow-up table: held-out, notN and wd 0 - wd 0.1 at 10k, per cell and seed', ok)
    con = lambda w, s: {'TT - UT': H[w, 'tt', s] - H[w, 'ut', s], 'TU - UU': H[w, 'tu', s] - H[w, 'uu', s], 'TT - TU': H[w, 'tt', s] - H[w, 'tu', s],
                        'UT - UU': H[w, 'ut', s] - H[w, 'uu', s], 'interaction': H[w, 'tt', s] - H[w, 'tu', s] - H[w, 'ut', s] + H[w, 'uu', s]}
    nc = {'TT - UT': (.06, .05, .15), 'TU - UU': (0, -.02, .02), 'TT - TU': (.10, .09, .13), 'UT - UU': (.04, .02, 0), 'interaction': (.06, .06, .13)}
    check('NOTE paired differences within wd 0 (five contrasts x three seeds)', all(near(con(0, s)[k], nc[k][n]) for k in nc for n, s in enumerate(SEEDS)))
    i1 = [con(.1, s)['interaction'] for s in SEEDS]
    check('NOTE: interaction at wd 0.1 is +0.30 to +0.56', near(min(i1), .30) and near(max(i1), .56), f'{min(i1):+.3f} to {max(i1):+.3f}')
    f2 = [H[0, 'tt', s] - H[.1, 'tt', s] for s in SEEDS]
    sp = {w: [max(H[w, c, s] for c in SCHEME) - min(H[w, c, s] for c in SCHEME) for s in SEEDS] for w in (0, .1)}
    lead = [H[0, 'tt', s] - max(H[0, c, s] for c in ('tu', 'ut', 'uu')) for s in SEEDS]
    check('NOTE F2: TT at wd 0 below wd 0.1 by >= 0.2 on every seed (-0.55 to -0.60)', max(f2) <= -.2 and near(max(f2), -.55) and near(min(f2), -.60))
    check('NOTE F3: spread across cells smaller at wd 0 on every seed (0.09-0.15 against 0.37-0.48)', all(a < b for a, b in zip(sp[0], sp[.1]))
          and near(min(sp[0]), .09) and near(max(sp[0]), .15) and near(min(sp[.1]), .37) and near(max(sp[.1]), .48),
          f'{min(sp[0]):.3f}-{max(sp[0]):.3f} against {min(sp[.1]):.3f}-{max(sp[.1]):.3f}')
    check('NOTE F4: TT highest at wd 0 on 3 of 3 seeds, "by 0.05-0.15" read as its lead over the next-best cell', min(lead) > 0 and near(min(lead), .05) and near(max(lead), .15),
          f'{min(lead):.3f}-{max(lead):.3f}')

    k0 = [(c, s) for c in SCHEME for s in SEEDS]
    at1k = [by[0, c, s][1000]['held']['final']['acc'] for c, s in k0]
    last2 = {w: [H[w, c, s] - by[w, c, s][8000]['held']['final']['acc'] for c, s in k0] for w in (0, .1)}
    rise = [H[0, c, s] - by[0, c, s][1000]['held']['final']['acc'] for c, s in k0]
    nn = [ok_s(by[0, c, s][10000])[~notn].mean() for c, s in k0]; nt = [ok_s(by[0, c, s][10000])[notn].mean() for c, s in k0]
    fit = [next(r['step'] for r in R[0, c, s] if all(r['train'][h]['acc'] == 1. for h in ('step1', 'step2', 'final'))) for c, s in k0]
    check('NOTE trajectories: held-out at 1,000 is 0.23-0.34', near(min(at1k), .23) and near(max(at1k), .34), f'{min(at1k):.3f}-{max(at1k):.3f}')
    check('NOTE trajectories: change over the last 2,000 updates -0.01 to +0.03 at wd 0, +0.03 to +0.14 at wd 0.1',
          near(min(last2[0]), -.01) and near(max(last2[0]), .03) and near(min(last2[.1]), .03) and near(max(last2[.1]), .14),
          f'{min(last2[0]):+.3f} to {max(last2[0]):+.3f}; {min(last2[.1]):+.3f} to {max(last2[.1]):+.3f}')
    check('NOTE trajectories: N 0.43-0.66 and notN <= 0.07 (to two decimals) at 10k', near(min(nn), .43) and near(max(nn), .66) and max(nt) < .075,
          f'N {min(nn):.3f}-{max(nn):.3f}, notN max {max(nt):.3f}')
    check('NOTE fit: every wd 0 run has all three training heads at 1.0 by 350-600 updates', min(fit) == 350 and max(fit) == 600, f'{min(fit)}-{max(fit)}')
    print(f'INFO wd 0 held-out change from 1,000 to 10,000: {min(rise):+.3f} to {max(rise):+.3f} (median {np.median(rise):+.3f}); '
          f'runs with a net rise: {sum(r > 0 for r in rise)} of 12')

    # Figure CSVs against the logged rows.
    rows = list(csv.DictReader(io.StringIO(git(commit, 'research/composition-c2/sharing/figures/followup_trajectories.csv').decode())))
    ok = len(rows) == 12 * 201
    for r in rows:
        m = by[float(r['wd']), r['cell'].lower(), int(r['seed'])][int(r['step'])]; o = ok_s(m)
        ok &= (r['run_id'] == f'c2-share-wd0-{r["cell"].lower()}-s{r["seed"]}-a1' and float(r['held_final']) == m['held']['final']['acc']
               and all(float(r[f'train_{h}']) == m['train'][h]['acc'] for h in ('final', 'step1', 'step2'))
               and float(r['N']) == o[~notn].mean() and float(r['notN']) == o[notn].mean())
    check('followup_trajectories.csv: all 2,412 rows (12 wd 0 runs x 201 steps) equal the logged rows and my N/notN recount', ok)
    cnt = {'tt': 19208, 'tu': 21272, 'ut': 53256, 'uu': 55320}
    def x5(rr, need=.95):
        v = [q['held']['final']['acc'] for q in rr]
        return next((rr[n]['step'] for n in range(len(v) - 4) if min(v[n:n+5]) >= need), None)
    rows = list(csv.DictReader(io.StringIO(git(commit, 'research/composition-c2/sharing/figures/followup_endpoints.csv').decode())))
    ok = len(rows) == 24
    for r in rows:
        k = (float(r['wd']), r['cell'].lower(), int(r['seed'])); m = by[k][10000]; o = ok_s(m); f5 = x5(R[k])
        fs = next(q['step'] for q in R[k] if all(q['train'][h]['acc'] == 1. for h in ('step1', 'step2', 'final')))
        ok &= (int(r['parameters']) == cnt[k[1]] and int(r['last_step']) == 10000 and float(r['train_final']) == m['train']['final']['acc']
               and float(r['held_final']) == o.mean() and float(r['N']) == o[~notn].mean() and float(r['notN']) == o[notn].mean()
               and int(r['first_step_train_all_heads_1']) == fs and r['first_step_held_ge_095_x5'] == ('' if f5 is None else str(f5)))
    check('followup_endpoints.csv: all 24 rows (both recipes) equal the logged rows, my recount, the parameter counts, the fit step and '
          'the first step of five consecutive held-out >= 0.95', ok)

    # Optimizer, from the last row with diag (9,950); the NOTE's 10k values are from local-only checkpoints.
    d = [by[0, c, s][9950]['diag'] for c, s in k0]
    fr = [r['diag']['adam_sqrt_vhat']['frac_below_eps'] for c, s in k0 for r in R[0, c, s] if 'diag' in r and r['step'] >= 500]
    med = [q['adam_sqrt_vhat']['median'] for q in d]; un = [q['update_norm'] for q in d]; mm = [q['train_final_margin']['min'] for q in d]
    print(f'INFO wd 0 at 9,950: median sqrt(vhat) {min(med):.1e}-{max(med):.1e}; update norm {min(un):.1e}-{max(un):.1e}; train margin min {min(mm):.1f}-{max(mm):.1f}; '
          f'fraction below eps over 500-9,950 {min(fr):.3f}-{max(fr):.3f}')
    check('NOTE optimizer, against the 9,950 rows: median sqrt(vhat) within 6e-6 to 2e-5, fraction below eps 2-9% from 500 on, update norm within 8e-4 to 5e-3, '
          'train margin min within 8.7-10.5 (allowing for one update and rounding)', min(med) >= 5.5e-6 and max(med) <= 2.5e-5 and min(fr) >= .015 and max(fr) <= .095
          and min(un) >= 7.5e-4 and max(un) <= 5.5e-3 and min(mm) >= 8.5 and max(mm) <= 10.6)
    g0 = [r for r in csv.DictReader(io.StringIO(git(commit, 'research/composition-c2/sharing/out/groups_wd0.csv').decode()))
          if r['cell'] == 'TT' and r['seed'] == '51200' and r['step'] == '10000' and r['group'] == 'transition']
    p = nets.load(C2/'recovered'/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz')[0]
    mine = float(np.sqrt(sum((p[k]**2).sum() for k in ('A', 'B', 'bh'))))
    check('NOTE: TT s51200 transition norm 29.0 at wd 0 (groups_wd0.csv) against 19.1 at wd 0.1 (mine, from the recovered checkpoint: A, B, bh)',
          abs(float(g0[0]['param_norm']) - 29.0) < .05 and abs(mine - 19.1) < .05, f'{float(g0[0]["param_norm"]):.3f} and {mine:.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
    if len(sys.argv) > 2:
        wd0(sys.argv[2])
        wd0_note(sys.argv[2])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
