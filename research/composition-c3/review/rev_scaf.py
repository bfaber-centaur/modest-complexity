"""c3 scaffolding-lane review (QUESTION.md section 3: S1-S7, CX-S1, CX-S2, then the NOTE's numbers).
The forward pass and oracle are the c2 reviewer's own (../../composition-c2/review/rev_pin.py), and the
schedule and exposure recounts are mine (rev_c3pin.py). The lane's records are read from its commit
with git archive; the reused keep and final-only arms are r1's and the c2 time lane's committed records.

uv run --locked python research/composition-c3/review/rev_scaf.py --out OUT --lane 4512bce
"""
import argparse
import csv
import io
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from rev_c3pin import C2, R1, REC, ROOT, SEEDS, my_exposure, my_w, rows  # noqa: E402
from rev_pin import FAIL, HELD, P, TRAIN, check, my_forward, oracle_rows  # noqa: E402
import c2task  # noqa: E402  (only to confirm my row order and N mask equal the pinned ones)
import nets  # noqa: E402

S_ALL, KINDS, TAG = (1000, 10000), {'drop': 1, 'taper': 2000}, {1000: '1k', 10000: '10k'}
FIXED, END = (1000, 10000, 20000, 30000, 40000, 50000), 50000
HEADS = ('step1', 'step2', 'final')
MEASURES = ('held_L3', 'held_notN', 'L1', 'L2', 'L4', 'L5', 'L6')
EPS = 1e-8
TR, HR = oracle_rows(TRAIN), oracle_rows(HELD)
LROWS = {L: oracle_rows(list(product(range(4), repeat=L))) for L in range(1, 7)}


def neighbour_sets(prefix_only=False):
    """Per held row: final labels of training rows with the same start that differ in exactly one op slot
    (prefix_only: in the last slot only)."""
    lab = {(int(s), *map(int, q)): int(y) for s, q, y in zip(TR['s'], TR['ops'], TR['states'][:, -1])}
    out = []
    for s, q in zip(HR['s'], HR['ops']):
        q, got = list(map(int, q)), set()
        for j in ((2,) if prefix_only else range(3)):
            for v in range(4):
                if v != q[j] and (int(s), *q[:j], v, *q[j+1:]) in lab:
                    got.add(lab[(int(s), *q[:j], v, *q[j+1:])])
        out.append(got)
    return out


NB_ANY, NB_PRE = neighbour_sets(), neighbour_sets(True)
Y = HR['states'][:, -1]
NMASK = np.array([int(y) in nb for y, nb in zip(Y, NB_ANY)])


def rid(kind, S, seed):
    return f'c3-scaf-{kind}-at{TAG[S]}-s{seed}-a1'


def ce(z, y):
    return float(np.mean(np.logaddexp.reduce(z, axis=1) - z[np.arange(len(y)), y]))


def score(p):
    """A logged row's r1 keys, recomputed with my forward on the training and held-out rows."""
    out = {}
    for name, d in (('train', TR), ('held', HR)):
        Z = my_forward(p, d['s'], d['ops'], [1, 2, 3])[1]
        out[name] = {h: dict(acc=float(np.mean(Z[t].argmax(1) == d['states'][:, t])), ce=ce(Z[t], d['states'][:, t])) for t, h in enumerate(HEADS)}
        if name == 'held':
            out['held_final_pred'] = ''.join(map(str, Z[2].argmax(1)))
    return out


def subsets(pred):
    ok = np.array([int(c) for c in pred]) == Y
    return dict(N=float(ok[NMASK].mean()), notN=float(ok[~NMASK].mean()))


def frozen(p):
    out = {f'L{L}': float(np.mean(my_forward(p, d['s'], d['ops'], [1]*L)[1][-1].argmax(1) == d['states'][:, -1])) for L, d in LROWS.items()}
    pred = ''.join(map(str, my_forward(p, HR['s'], HR['ops'], [1, 2, 3])[1][-1].argmax(1)))
    ok = np.array([int(c) for c in pred]) == Y
    out.update(held_L3=float(ok.mean()), held_N=float(ok[NMASK].mean()), held_notN=float(ok[~NMASK].mean()), pred=pred)
    return out


def crossing(steps, acc):
    """First of five consecutive logged evaluations (50 apart) at >= 0.95; None if there is none."""
    for i in range(len(steps) - 4):
        if min(acc[i:i+5]) >= .95 and steps[i+4] - steps[i] == 200:
            return steps[i]
    return None


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def num(x):
    return None if x == '' else float(x)


def close(a, b, tol=1e-12):
    if a is None or b is None or a == '' or b == '':
        return (a in (None, '')) == (b in (None, ''))
    return abs(float(a) - float(b)) <= tol * max(1., abs(float(b)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--lane', required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    tar = subprocess.run(['git', 'archive', a.lane, 'research/composition-c3/scaffolding'], cwd=ROOT, capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(a.out/'lane', filter='data')
    LANE = a.out/'lane'/'research'/'composition-c3'/'scaffolding'
    FIG = LANE/'figures'

    # My rows and N mask against the pinned ones (row order matters for held_final_pred).
    T = c2task.load()
    for name, mine, ref in (('train', TR, c2task.rows(T['P'], T['train'])), ('held', HR, c2task.rows(T['P'], T['held']))):
        check(f'S0 my {name} rows equal c2task.rows in order (s, ops, states)', all(np.array_equal(mine[k], ref[k]) for k in ('s', 'ops', 'states')))
    check('S0 my N mask (true label among one-slot training neighbours) equals c2task.neighbour_answerable: 86 N, 42 notN',
          np.array_equal(NMASK, c2task.neighbour_answerable(T['P'], T['train'], T['held'])[0]) and NMASK.sum() == 86)

    # Records of every arm. keep/final: r1 from init, then the c2 time continuation from 10,000.
    r1 = {(c, s): rows(R1/'composition'/'runs'/f'r1-comp-rnn-{c}-s{s}-a1'/'metrics.jsonl') for c in ('inter', 'final') for s in SEEDS}
    traj, ck = {}, {}
    for arm, c in (('keep', 'inter'), ('final', 'final')):
        for s in SEEDS:
            cont = rows(C2/'time'/'runs'/f'c2-time-tt-{c}-s{s}-a1'/'metrics.jsonl')
            assert all(cont[0][k] == r1[(c, s)][-1][k] for k in r1[(c, s)][-1]) and cont[0]['step'] == 10000
            traj[(arm, None, s)] = [r for r in r1[(c, s)] if r['step'] < 10000] + cont
            for t in FIXED:
                ck[(arm, None, s, t)] = REC/f'r1-comp-rnn-{c}-s{s}-a1'/f'ckpt-{t}.npz' if t <= 10000 else C2/'time'/'runs'/f'c2-time-tt-{c}-s{s}-a1'/f'ckpt-{t}.npz'
    cfgs = {}
    for S in S_ALL:
        for kind in KINDS:
            for s in SEEDS:
                d = LANE/'runs'/rid(kind, S, s)
                traj[(kind, S, s)] = rows(d/'metrics.jsonl'); cfgs[(kind, S, s)] = json.loads((d/'config.json').read_text())
                for t in sorted({S, S + 2000, *[x for x in FIXED if x >= S]}):
                    ck[(kind, S, s, t)] = REC/f'r1-comp-rnn-inter-s{s}-a1'/f'ckpt-{t}.npz' if t == S else d/f'ckpt-{t}.npz'
    branches = [(kind, S, s) for S in S_ALL for kind in KINDS for s in SEEDS]

    # S1. Branch points.
    recovery = {(e['run_id'], e['file']): e['sha256'] for e in json.loads((C2/'recovered'/'RECOVERY.json').read_text())['files']}
    plan = {e['run_id']: e['argv'] for e in json.loads((LANE/'plan.json').read_text())['runs']}
    bad1, badrow = [], []
    for kind, S, s in branches:
        c, ID, src = cfgs[(kind, S, s)], rid(kind, S, s), f'r1-comp-rnn-inter-s{s}-a1'
        rf = c['resumed_from']
        ok = (rf['path'] == f'research/composition-c2/recovered/{src}/ckpt-{S}.npz' and rf['sha256'] == recovery[(src, f'ckpt-{S}.npz')]
              == nets.sha256_file(REC/src/f'ckpt-{S}.npz') and rf['step'] == S and rf['source']['run_id'] == src and rf['source']['manifest_listed']
              and c['run_id'] == ID and c['task_id'] == 'ops51000-ls51100' and c['transition'] == c['readout'] == 'tied'
              and c['step_weights'] == [1., 1., 1.] and c['hidden'] == 128 and c['lr'] == 1e-3 and c['wd'] == .1 and c['eval_every'] == 50
              and c['updates'] == END and c['switch_step'] == S and c['switch_weights'] == [0., 0., 1.] and c['taper_updates'] == KINDS[kind]
              and all(f in plan[ID] for f in ('--task-id', '--transition', '--readout', '--step-weights', '--hidden', '--lr', '--wd', '--eval-every')))
        if not ok:
            bad1.append(ID)
        r1row = next(r for r in r1[('inter', s)] if r['step'] == S)
        if not (traj[(kind, S, s)][0]['step'] == S and all(traj[(kind, S, s)][0][k] == r1row[k] for k in r1row)):
            badrow.append(ID)
    check('S1 all 12 configs resume r1 tied/inter ckpt-S at its RECOVERY.json hash (manifest-listed, source run ID) with the right run ID and explicit recipe flags', not bad1, str(bad1))
    check("S1 each branch's first row (step S) equals r1's published row on every r1 key", not badrow, str(badrow))

    # S2. Schedule and exposure.
    bad_w, bad_e, n_rows = [], [], 0
    for kind, S, s in branches:
        D = KINDS[kind]
        for r in traj[(kind, S, s)]:
            n_rows += 1
            if r['step_weights'] != my_w(r['step'], S, D) or r['step_weights'][2] != 1.:
                bad_w.append((rid(kind, S, s), r['step']))
        ex = cfgs[(kind, S, s)]['exposure']
        for key, start in (('labels_this_run', S), ('labels_from_init', 0)):
            m = my_exposure(384, start, END, S, D)
            if ex[key]['updates_with_weight'] != m['updates_with_weight'] or not np.allclose(ex[key]['weighted_row_presentations'], m['weighted_row_presentations'], rtol=1e-12, atol=0):
                bad_e.append((rid(kind, S, s), key))
    check(f'S2 step_weights equal my w(t) on every logged row of the 12 branches ({n_rows} rows); final weight exactly 1', not bad_w, str(bad_w[:3]))
    check('S2 config exposure (this run and from init) equals my recount in all 12', not bad_e, str(bad_e))
    pts = {(S, D): [my_w(S + k, S, D)[0] for k in (-1, 0, 1, 1998, 1999, 2000)] for S in S_ALL for D in (1, 2000)}
    print('     S2 my intermediate weight at S-1, S, S+1, S+1998, S+1999, S+2000:', {f'S={S} D={D}': v for (S, D), v in pts.items()})

    # S3. Records: manifests, rescoring, registry.
    bad_m, bad_s, n_ck = [], [], 0
    for kind, S, s in branches:
        d, ID = LANE/'runs'/rid(kind, S, s), rid(kind, S, s)
        for e in json.loads((d/'MANIFEST.json').read_text()):
            if e['committed'] and nets.sha256_file(d/e['file']) != e['sha256']:
                bad_m.append((ID, e['file']))
        logged = {r['step']: r for r in traj[(kind, S, s)]}
        for f in sorted(d.glob('ckpt-*.npz')):
            n_ck += 1
            p, st, meta = nets.load(f)
            t = int(f.stem.split('-')[1]); mine, row = score(p), logged[t]
            ok = st['t'] == t == meta['step'] and meta['run_id'] == ID and mine['held_final_pred'] == row['held_final_pred'] and all(
                mine[x][h]['acc'] == row[x][h]['acc'] and close(mine[x][h]['ce'], row[x][h]['ce'], 1e-9) for x in ('train', 'held') for h in HEADS)
            if not ok:
                bad_s.append((ID, t))
    check('S3 every committed file of the 12 branches matches its MANIFEST.json hash', not bad_m, str(bad_m))
    check(f'S3 every committed branch checkpoint ({n_ck}) rescores with my forward to its logged row (train and held acc exact, CE 1e-9, held_final_pred); count, meta step and run ID', not bad_s and n_ck == 66, f'{n_ck} checkpoints; {bad_s[:3]}')
    reg = [json.loads(x) for x in (LANE/'registry.jsonl').read_text().splitlines()]
    ev = {}
    for e in reg:
        ev.setdefault(e['run_id'], []).append(e['status'])
    check('S3 registry: every plan entry has exactly one running and one complete event (no failure, cap or silent retry)',
          set(ev) == set(plan) and all(v == ['running', 'complete'] for v in ev.values()), f'{len(plan)} plan entries; {dict((k, v) for k, v in ev.items() if v != ["running", "complete"])}')

    # S4. Frozen lengths 1-6, exhaustively, at every checkpoint the lane evaluated.
    fz = {key: frozen(nets.load(path)[0]) for key, path in ck.items()}
    lane_fz = read_csv(FIG/'scaf_frozen.csv')
    keyof = lambda r: (r['arm'], int(r['S']) if r['S'] else None, int(r['seed']), int(r['step']))
    bad4 = [keyof(r) for r in lane_fz if any(float(r[m]) != fz[keyof(r)][m] for m in ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'held_L3', 'held_N', 'held_notN'))]
    check(f'S4 every value of scaf_frozen.csv ({len(lane_fz)} rows x 9) equals my exhaustive evaluation; same set of checkpoints',
          not bad4 and {keyof(r) for r in lane_fz} == set(fz), f'{bad4[:3]}; mine {len(fz)}')
    for name in ('c2-gen-frozen-orig-a2', 'c2-gen-frozen-50k-a1'):
        pub = [r for r in read_csv(C2/'generalization'/'runs'/name/'length.csv') if r['subset'] == 'all' and r['task_id'] == 'ops51000-ls51100']
        badp = [(r['condition'], r['init_seed'], r['ckpt_step'], r['length']) for r in pub
                if float(r['accuracy']) != fz[('keep' if r['condition'] == 'inter' else 'final', None, int(r['init_seed']), int(r['ckpt_step']))][f"L{r['length']}"]]
        check(f'S4 my frozen L1-6 equals c2 generalization {name}/length.csv for keep and final-only ({len(pub)} values)', not badp and len(pub) == 36, str(badp[:3]))
    badpred = [k for k, v in fz.items() if (lg := {r['step']: r for r in traj[k[:3]]}.get(k[3])) and lg['held_final_pred'] != v['pred']]
    check('S4 at every evaluated checkpoint with a logged row, my held-out prediction string equals the logged one', not badpred, str(badpred[:3]))

    # S5. Retention arithmetic, labels, thresholds.
    lane_ret = read_csv(FIG/'scaf_retention.csv')
    mine_ret = {}
    for kind, S, s in branches:
        for m in MEASURES:
            for t in [x for x in FIXED if x >= S]:
                b, k, f, sw = fz[(kind, S, s, t)][m], fz[('keep', None, s, t)][m], fz[('final', None, s, t)][m], fz[('keep', None, s, S)][m]
                st = '' if t != END else ('degraded' if b - k <= -.05 else 'above keep' if b - k >= .05 else 'retained')
                mine_ret[(kind, S, s, m, t)] = dict(value=b, at_switch=sw, keep=k, final_only=f, d_branch=b - sw, d_keep=b - k, d_final=b - f, status=st)
    bad5 = [r for r in lane_ret if not (all(close(r[x], mine_ret[(r['arm'], int(r['S']), int(r['seed']), r['measure'], int(r['step']))][x]) for x in
            ('value', 'at_switch', 'keep', 'final_only', 'd_branch', 'd_keep', 'd_final')) and r['status'] == mine_ret[(r['arm'], int(r['S']), int(r['seed']), r['measure'], int(r['step']))]['status'])]
    ends = [v for (kind, S, s, m, t), v in mine_ret.items() if t == END]
    margin = min(abs(abs(v['d_keep']) - .05) for v in ends)
    check(f'S5 every row of scaf_retention.csv ({len(lane_ret)}) recomputes from my frozen values: deltas to 1e-12 and the endpoint labels', not bad5 and len(lane_ret) == len(mine_ret), f'{len(bad5)} differ')
    check(f"S5 endpoint labels: {sum(v['status'] == 'retained' for v in ends)} of {len(ends)} retained; nearest |d_keep| to the 0.05 boundary is {margin:.4f} away (the 1e-12 tolerance never matters)",
          margin > 1e-9)
    lane_thr = read_csv(FIG/'scaf_threshold.csv')
    mine_thr = {}
    for arm_key, rs in traj.items():
        arm, S, s = arm_key
        full = rs if S is None else [r for r in r1[('inter', s)] if r['step'] < S] + rs
        steps = [r['step'] for r in full]
        for name, acc in (('held_final', [r['held']['final']['acc'] for r in full]), ('held_notN', [subsets(r['held_final_pred'])['notN'] for r in full])):
            c = crossing(steps, acc)
            mine_thr[(arm, S, s, name)] = dict(crossing_step=c, censored=c is None, before_switch=c is not None and S is not None and c < S,
                                               value_at_switch=acc[steps.index(S)] if S else None, value_at_end=acc[-1])
    bad5t = [r for r in lane_thr if (lambda m: num(r['crossing_step']) != m['crossing_step'] or r['censored'] != str(m['censored'])
             or r['before_switch'] != str(m['before_switch']) or not close(num(r['value_at_switch']), m['value_at_switch']) or not close(float(r['value_at_end']), m['value_at_end']))(
             mine_thr[('final' if r['arm'] == 'final' else r['arm'], int(r['S']) if r['S'] else None, int(r['seed']), r['measure'])])]
    check(f'S5 every row of scaf_threshold.csv ({len(lane_thr)}) recomputes from the joined trajectories (crossing, censoring, before-switch, values)', not bad5t and len(lane_thr) == len(mine_thr), str(bad5t[:2]))
    bad_sub = [(k, r['step']) for k, rs in traj.items() for r in rs if 'held_subset' in r and (lambda x: (x['N'], x['notN']) != (r['held_subset']['N'], r['held_subset']['notN']))(subsets(r['held_final_pred']))]
    check('S5 every logged held_subset equals my N / notN split of its prediction string', not bad_sub, str(bad_sub[:3]))

    # S6. Parameters keep moving; the Adam count carries.
    bad6, small = [], []
    for kind, S, s in branches:
        (p0, _, _), (p1, s1, _), (p2, s2, _) = (nets.load(ck[(kind, S, s, t)]) for t in (S, S + 2000, END))
        mv = [float(np.abs(y[n] - x[n]).max()) for x, y in ((p0, p1), (p1, p2)) for n in p0]
        small.append(min(mv))
        if min(mv) <= 0 or s1['t'] != S + 2000 or s2['t'] != END:
            bad6.append(rid(kind, S, s))
    check('S6 in every branch every parameter array changes over S -> S+2000 and S+2000 -> 50000; the count is S+2000 and 50000', not bad6, f'smallest max|change| {min(small):.1e}; {bad6}')

    # S7. Figure CSVs.
    lane_main = read_csv(FIG/'scaf_main.csv')
    idx = {(arm, S, s, r['step']): (arm, S, s, r) for (arm, S, s), rs in traj.items() for r in rs}
    bad7, n7 = [], 0
    for r in lane_main:
        arm, S, s, row = idx[(r['arm'], int(r['S']) if r['S'] else None, int(r['seed']), int(r['step']))]
        w = row.get('step_weights') or ([1., 1., 1.] if arm == 'keep' else [0., 0., 1.])
        sub, dg = subsets(row['held_final_pred']), row.get('diag', {})
        v = dg.get('adam_sqrt_vhat', {}); mg = dg.get('train_final_margin', {})
        mine = dict(w_inter=w[0], weighted_train_loss=sum(wk*row['train'][h]['ce'] for wk, h in zip(w, HEADS)),
                    held_final_acc=row['held']['final']['acc'], held_final_ce=row['held']['final']['ce'], held_N_acc=sub['N'], held_notN_acc=sub['notN'],
                    grad_norm=dg.get('grad_norm'), update_norm=dg.get('update_norm'), sqrt_vhat_median=v.get('median'), sqrt_vhat_max=v.get('max'),
                    frac_below_eps=v.get('frac_below_eps'), train_margin_min=mg.get('min'), train_margin_median=mg.get('median'),
                    **{f'train_{h}_{m}': row['train'][h][m] for h in HEADS for m in ('acc', 'ce')})
        n7 += len(mine)
        if not all(close(num(r[k]), x) for k, x in mine.items()):
            bad7.append((r['arm'], r['S'], r['seed'], r['step']))
    check(f'S7 every value of scaf_main.csv ({len(lane_main)} rows, {n7} values) recomputes from the logged rows', not bad7 and len(lane_main) == len(idx), f'{bad7[:3]}; mine {len(idx)} rows')
    lane_opt = read_csv(FIG/'scaf_optimizer.csv')
    bad7o = []
    for r in lane_opt:
        key = (r['arm'], int(r['S']) if r['S'] else None, int(r['seed']))
        dg = [(x['step'], x['diag']) for x in traj[key] if 'diag' in x]
        mine = dict(first_diag_step=dg[0][0], median_below_eps_step=next((t for t, x in dg if x['adam_sqrt_vhat']['median'] < EPS), None),
                    all_below_eps_step=next((t for t, x in dg if x['adam_sqrt_vhat']['frac_below_eps'] >= 1.), None))
        if key[1]:
            u = dict((t, x['update_norm']) for t, x in dg)
            mine.update({f'update_norm_ratio_S+{k}': u[key[1] + k] / u[key[1]] for k in (50, 500, 2000, 5000)})
        if not all(close(num(r[k]), x) for k, x in mine.items()):
            bad7o.append(key)
    check(f'S7 every value of scaf_optimizer.csv ({len(lane_opt)} rows) recomputes from the logged diag', not bad7o, str(bad7o))
    lane_cmp = read_csv(FIG/'scaf_compatibility.csv')
    bad7c = []
    for r in lane_cmp:
        key = (r['arm'], int(r['S']) if r['S'] else None, int(r['seed']))
        pred = np.array([int(c) for c in {x['step']: x for x in traj[key]}[int(r['step'])]['held_final_pred']])
        wrong = np.flatnonzero(pred != Y)
        f = lambda sets: (float(np.mean([pred[i] in sets[i] for i in wrong])), float(np.mean([len(sets[i] - {int(Y[i])})/7 for i in wrong]))) if len(wrong) else (None, None)
        (x1, u1), (x2, u2) = f(NB_ANY), f(NB_PRE)
        mine = dict(n_wrong=len(wrong), any_nbr=x1, any_nbr_uniform=u1, same_prefix=x2, same_prefix_uniform=u2)
        if not all(close(num(r[k]), x) for k, x in mine.items()):
            bad7c.append((key, r['step']))
    check(f'S7 every value of scaf_compatibility.csv ({len(lane_cmp)} rows) recomputes with my own neighbour sets', not bad7c, str(bad7c[:3]))

    # CX-S1. At S = 10k, does length-3 retention hide a length-6 or h1/h2 loss?
    print('     CX-S1 at S=10k, ckpt-50000 (mine):')
    cx1 = {}
    for kind in KINDS:
        ret = [abs(fz[(kind, 10000, s, END)]['held_L3'] - fz[('keep', None, s, END)]['held_L3']) < .05 for s in SEEDS]
        gap6 = [fz[('keep', None, s, END)]['L6'] - fz[(kind, 10000, s, END)]['L6'] for s in SEEDS]
        sc = [score(nets.load(ck[(kind, 10000, s, END)])[0])['train'] for s in SEEDS]
        h12 = [min(x['step1']['acc'], x['step2']['acc']) for x in sc]
        low = [min(min(r['train']['step1']['acc'], r['train']['step2']['acc']) for r in traj[(kind, 10000, s)]) for s in SEEDS]
        cx1[kind] = (all(ret), sum(g >= .05 for g in gap6), sum(x < .9 for x in h12))
        print(f'       {kind}: L3 retained {ret}; keep L6 - branch L6 {[round(g, 4) for g in gap6]}; train min(step1, step2) acc {h12}; lowest logged after S {low}')
    check('CX-S1 (prediction) if B or C is retained at L3 at S=10k, its L6 at 50k is >= 0.05 below keep in >= 1 seed', all(r[1] >= 1 for r in cx1.values() if r[0]), str(cx1))
    check('CX-S1 (prediction) ... and its train step-1 or step-2 accuracy is < 0.9 in >= 2 seeds', all(r[2] >= 2 for r in cx1.values() if r[0]), str(cx1))

    # CX-S2. Decay without the intermediate loss.
    print('     CX-S2 parameter norms at ckpt-50000, branch / keep, S=10k:')
    cx2 = {}
    for kind in KINDS:
        for s in SEEDS:
            pb, pk = nets.load(ck[(kind, 10000, s, END)])[0], nets.load(ck[('keep', None, s, END)])[0]
            nb, nk = (float(np.sqrt(sum(float(np.sum(x*x)) for x in q.values()))) for q in (pb, pk))
            grp = {n: float(np.linalg.norm(pb[n]) / np.linalg.norm(pk[n])) for n in pk}
            cx2[(kind, s)] = (nb / nk, grp)
            print(f"       {kind} s{s}: total {nb:.4f} / {nk:.4f} = {nb/nk:.4f}; " + ', '.join(f'{n} {x:.3f}' for n, x in grp.items()))
    check('CX-S2 (prediction) every drop branch at S=10k has a smaller parameter norm than keep at 50k, in all three seeds', all(cx2[('drop', s)][0] < 1 for s in SEEDS))
    check('CX-S2 (prediction) ... and V has the smallest branch/keep norm ratio of any parameter group', all(min(cx2[('drop', s)][1], key=cx2[('drop', s)][1].get) == 'V' for s in SEEDS))

    (a.out/'rev_scaf.json').write_text(json.dumps(dict(
        frozen={'|'.join(map(str, k)): {m: v for m, v in x.items() if m != 'pred'} for k, x in fz.items()},
        threshold={'|'.join(map(str, k)): v for k, v in mine_thr.items()},
        retention={'|'.join(map(str, k)): v for k, v in mine_ret.items()},
        cx2={f'{k}|{s}': v for (k, s), v in cx2.items()}), indent=1))


if __name__ == '__main__':
    main()
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
