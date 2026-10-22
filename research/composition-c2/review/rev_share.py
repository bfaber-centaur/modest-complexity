"""Sharing lane checks (QUESTION.md 2.4 and Revisions 04:28; REVIEW.md section 5). Analysis only:
lane files are read from git at a lane commit; TT and TU are r1's runs (recovered checkpoints and
published metrics; TU is the reviewer's r1 arm). Rescoring uses rev_pin.py's forward pass and
rev_gen.py's neighbour recount, not the lane's code.

uv run --locked python research/composition-c2/review/rev_share.py LANE_COMMIT
"""
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


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
