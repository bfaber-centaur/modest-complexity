"""Generalization lane checks (QUESTION.md 2.2 and Revisions 04:14; REVIEW.md section 3).
Part A: task facts and the frozen length 1-6 evaluation of the six recovered r1 tied/tied
checkpoints, recomputed with the oracle and forward pass of rev_pin.py (not the lane's code),
against the lane's published CSVs at a lane commit; then counterexample CX-G1.

uv run --locked python research/composition-c2/review/rev_gen.py LANE_COMMIT [PART_B_COMMIT [TIME_COMMIT [ANATOMY_COMMIT]]]
"""
import csv
import io
from itertools import permutations, product
import json
import subprocess
import sys
import numpy as np

from rev_pin import C2, FAIL, HELD, P, ROOT, TRAIN, check, my_forward, oracle_rows
import io
import nets

RUNS = [f'r1-comp-rnn-{c}-s{s}-a1' for c in ('final', 'inter') for s in (51200, 51201, 51202)]
LANE = 'research/composition-c2/generalization'


def lane_csv(commit, path):
    text = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{LANE}/{path}'], text=True)
    return list(csv.DictReader(io.StringIO(text)))


def compose(q):
    f = tuple(range(8))
    for k in q:
        f = tuple(P[k][x] for x in f)
    return f


def notn_mask(he):
    """True where no one-op-slot training neighbour (same start) has the row's final label."""
    lab = {(s, *q): y for q in TRAIN for s in range(8) for y in [compose(q)[s]]}
    out = []
    for s, q, y in zip(he['s'], he['ops'], he['states'][:, -1]):
        nb = {lab[k] for j in range(3) for v in range(4) if v != q[j]
              for k in [(int(s), *[int(x) for x in q[:j]], v, *[int(x) for x in q[j+1:]])] if k in lab}
        out.append(int(y) not in nb)
    return np.array(out)


def main(commit):
    # 1. Task facts (lane QUESTION.md table), with permutation tuples.
    seen, frontier = {tuple(range(8))}, [tuple(range(8))]
    while frontier:
        frontier = [g2 for g in frontier for k in range(4) for g2 in [tuple(P[k][x] for x in g)] if g2 not in seen and not seen.add(g2)]
    short = {compose(q) for L in (1, 2) for q in product(range(4), repeat=L)}
    f3tr, f3he = {compose(q) for q in TRAIN}, {compose(q) for q in HELD}
    distinct84 = len(short | f3tr | f3he) == 84
    mine = {}
    for L in (4, 5, 6):
        fs = [compose(q) for q in product(range(4), repeat=L)]
        cnt = {}
        for f in fs:
            cnt[f] = cnt.get(f, 0) + 1
        mine[L] = (len(fs), len(cnt), sum(cnt[f] > 1 for f in fs), sum(f in short for f in fs), sum(f in f3tr for f in fs),
                   sum(f in f3he for f in fs), sum(f not in short | f3tr | f3he for f in fs))
    lane_table = {4: (256, 256, 0, 0, 0, 0, 256), 5: (1024, 1015, 18, 0, 0, 0, 1024), 6: (4096, 3907, 372, 8, 1, 1, 4086)}
    check(f'task facts: group order {len(seen)} (S8 = 40320); 84 products of lengths 1-3 distinct; length 4-6 function-class table equals the lane\'s',
          len(seen) == 40320 and distinct84 and mine == lane_table, str(mine))

    # 2. Part A, every row at lengths 1-6, my forward and oracle, against the lane's length.csv and steps.csv.
    rows_a = lane_csv(commit, 'runs/c2-gen-frozen-orig-a2/length.csv'); rows_s = lane_csv(commit, 'runs/c2-gen-frozen-orig-a2/steps.csv')
    he = oracle_rows(HELD); notn = notn_mask(he)
    worst, n_cmp, acc, steps_ok, windows = 0., 0, {}, True, {}
    for run in RUNS:
        p = nets.load(C2/'recovered'/run/'ckpt-10000.npz')[0]
        lane = {(int(r['length']), r['subset']): r for r in rows_a if r['run_id'] == run}
        for L in range(1, 7):
            progs = list(product(range(4), repeat=L)); d = oracle_rows(progs)
            pred = my_forward(p, d['s'], d['ops'], [1]*L)[1][-1].argmax(1); ok = pred == d['states'][:, -1]
            acc[run, L] = ok.mean()
            sub = {'all': np.ones(len(ok), bool)}
            q3 = [tuple(map(int, q[:3])) for q in d['ops']]
            if L == 3:
                held = np.array([q in set(HELD) for q in q3]); sub.update({'train programs': ~held, 'held-out programs': held})
                order = {q: i for i, q in enumerate(HELD)}  # held rows in oracle_rows(HELD) order
                idx = np.array([order[q]*8 + int(s) for q, s in zip(q3, d['s']) if q in order])
                hn = np.zeros(len(ok), bool); hn[held] = notn[idx]
                sub.update({'held-out N': held & ~hn, 'held-out notN': hn})
            if L >= 4:
                held = np.array([q in set(HELD) for q in q3])
                sub.update({'length-3 prefix is a training program': ~held, 'length-3 prefix is a held-out program': held})
                fs = [compose(q) for q in d['ops'][::8]]  # oracle_rows: programs in order, 8 starts each
                cls = np.repeat([('F12' if f in short else 'F3train' if f in f3tr else 'F3held' if f in f3he else 'novel') for f in fs], 8)
                sub.update({f'function class {c}': cls == c for c in ('F12', 'F3train', 'F3held', 'novel')})
                # CX-G1: any 3-op window a held-out program
                anyw = np.array([any(tuple(map(int, q[i:i+3])) in set(HELD) for i in range(L-2)) for q in d['ops']])
                windows[run, L] = (ok[anyw].mean(), ok[~anyw].mean(), int(anyw.sum()), int((~anyw).sum()))
            for name, m in sub.items():
                r = lane[L, name]; n_cmp += 1
                if int(r['n_rows']) != int(m.sum()):
                    worst = np.inf
                elif m.any():
                    worst = max(worst, abs(float(r['accuracy']) - ok[m].mean()))
        # steps.csv: per-step accuracy, first error, and the step after it, on all length-6 rows
        d = oracle_rows(list(product(range(4), repeat=6))); Z = my_forward(p, d['s'], d['ops'], [1]*6)[1]
        yhat = np.stack([z.argmax(1) for z in Z], 1); wrong = yhat != d['states']
        first = np.where(wrong.any(1), wrong.argmax(1), -1)
        for r in (r for r in rows_s if r['run_id'] == run and r['step'] != '0'):
            if not r['step'].isdigit():
                steps_ok &= r['step'] == 'never wrong in 6 steps' and int(r['first_error_here']) == (first == -1).sum(); continue
            t = int(r['step']) - 1; at = first == t
            steps_ok &= float(r['accuracy']) == (~wrong[:, t]).mean() and int(r['first_error_here']) == at.sum()
            if t < 5:
                nxt, rows_ = yhat[at, t+1], np.where(at)[0]
                follows = nxt == np.array([P[d['ops'][i, t+1]][yhat[i, t]] for i in rows_])
                rec = nxt == d['states'][at, t+1]
                steps_ok &= (int(r['after_first_error_n']), int(r['after_recovered']), int(r['after_follows_own_error']), int(r['after_other'])) == \
                    (int(at.sum()), int(rec.sum()), int(follows.sum()), int((~rec & ~follows).sum()))
    check(f'Part A: {n_cmp} (run, length, subset) accuracies and row counts in length.csv equal my forward and oracle on every row', worst < 1e-12, f'max |diff| {worst:.1e}')
    check('Part A: steps.csv accuracy, first-error counts and the after-first-error split (recovered / follows own error / other) equal mine', steps_ok)
    for run in RUNS:
        print(f'INFO {run}: accuracy at L = 1..6:', [round(float(acc[run, L]), 3) for L in range(1, 7)])
    # references.csv: oracle, and each checkpoint's own one-step map re-encoded (argmax readout of one step from start(s), iterated)
    refs = {(r['system'], int(r['length'])): float(r['accuracy']) for r in lane_csv(commit, 'runs/c2-gen-frozen-orig-a2/references.csv')}
    ref_ok = all(refs['oracle (per-row loop)', L] == 1. for L in range(1, 7))
    for run in RUNS:
        p = nets.load(C2/'recovered'/run/'ckpt-10000.npz')[0]
        step1 = np.array([[my_forward(p, np.array([s]), np.array([[k]]), [1])[1][0].argmax() for k in range(4)] for s in range(8)])
        for L in range(1, 7):
            d = oracle_rows(list(product(range(4), repeat=L))); y = d['s'].copy()
            for t in range(L):
                y = step1[y, d['ops'][:, t]]
            ref_ok &= refs[f'own one-step map re-encoded ({run})', L] == (y == d['states'][:, -1]).mean()
    check('references.csv: oracle 1.0 at every length; each checkpoint\'s own one-step map re-encoded equals mine at L = 1-6 (P2 line NOT CHECKED)', ref_ok)
    fig = lane_csv(commit, 'figures/length.csv')
    in_runs = lambda f: any(all(r[k] == f[k] for k in ('run_id', 'length', 'subset', 'n_rows', 'accuracy')) for r in rows_a)
    in_refs = lambda f: f['source'].endswith('references.csv') and f['subset'] == 'all' and float(f['accuracy']) == refs[f['run_id'], int(f['length'])]
    check('figures/length.csv: every row equals its source row (run length.csv or references.csv)', all(in_runs(f) or in_refs(f) for f in fig), f'{len(fig)} rows')

    # 3. CX-G1 (success branch: tied/inter L = 4 accuracy >= 0.5 in a seed).
    print('CX-G1 accuracy with any 3-op window a held-out program | with none (rows): ')
    held_ok = True
    for run in RUNS:
        if 'inter' in run and acc[run, 4] >= .5:
            for L in (4, 5, 6):
                a, b, na, nb = windows[run, L]
                held_ok &= abs(a - b) < .05
                print(f'  {run} L={L}: {a:.3f} ({na}) | {b:.3f} ({nb}); diff {a - b:+.3f}')
    check('CX-G1 prediction: tied/inter accuracy differs by < 0.05 between rows with and without a held-out window, at every L = 4-6 and seed', held_ok)



def rows_p(Pm, progs):
    s_, q_, y_ = [], [], []
    for q in progs:
        for s in range(8):
            x, st = s, []
            for k in q:
                x = Pm[k][x]; st.append(x)
            s_.append(s); q_.append(list(q)); y_.append(st)
    return dict(s=np.array(s_), ops=np.array(q_), states=np.array(y_))


def compose_p(Pm, q):
    f = tuple(range(8))
    for k in q:
        f = tuple(Pm[k][x] for x in f)
    return f


def draws_and_splits():
    """Part B's draws (from 63000) and splits (from 63100), recomputed: r1's permutation draw and acceptance plus
    84 distinct products; r1's split, Latin, distinct squares per draw. Returns the decision log, draws, tasks."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('r1_task_rev', ROOT/'research'/'regimes'/'composition'/'task.py')
    r1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r1)  # r1's split() is the definition of a split
    draw = lambda seed: [[int(v) for v in np.random.default_rng([seed, k]).permutation(8)] for k in range(4)]
    log, draws, seed = [], [], 63000
    while len(draws) < 2:
        Pm = draw(seed)
        ok = (not any(p == list(range(8)) for p in Pm) and len({tuple(p) for p in Pm}) == 4
              and len({compose_p(Pm, q) for L in (1, 2, 3) for q in product(range(4), repeat=L)}) == 84)
        log.append(('draw', seed, ok)); draws += [(seed, Pm)] * ok; seed += 1
    tasks, seed = [], 63100
    for dseed, Pm in draws:
        squares = []
        while len(squares) < 2:
            L = r1.split(seed)[0].tolist()
            latin = all(sorted(r) == [0, 1, 2, 3] for r in L) and all(sorted(c) == [0, 1, 2, 3] for c in zip(*L))
            ok = latin and L not in squares
            log.append(('split', seed, ok))
            if ok:
                squares.append(L); tasks.append((f'ops{dseed}-ls{seed}', Pm, L))
            seed += 1
    return log, draws, tasks


def partb(commit):
    """Part B (QUESTION.md 2.2 item 3 and CX-G2): new draws and splits, recipes, init pairing, rescoring."""
    import c2task
    T = json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{LANE}/tasks.json']))
    log, draws, tasks = draws_and_splits()
    lane_log = [(e['kind'], e['seed'], e['accepted']) for e in T['decision_log']]
    struct = {e['task_id']: e for e in T['structure']}
    same = lane_log == log and [t_[0] for t_ in tasks] == T['accepted_tasks']
    for tid, Pm, L in tasks:
        held = sorted((a, b, L[a][b]) for a in range(4) for b in range(4)); C = c2task.load(tid)
        same &= struct[tid]['ops'] == Pm and struct[tid]['latin_square'] == L and [tuple(h) for h in struct[tid]['held_out_programs']] == held
        same &= C['P'].tolist() == Pm and C['L'].tolist() == L and [tuple(h) for h in C['held']] == held
    check(f'Part B draws and splits recomputed from their seeds (r1 acceptance, 84 distinct products, Latin, distinct squares): '
          f'decision log, accepted tasks, permutations, squares and held-out programs equal tasks.json and c2task.load', same, str([t_[0] for t_ in tasks]))
    for dseed, Pm in draws:
        fixed = [(x, Pm[0][x]) for x in range(8) if len({Pm[k][x] for k in range(4)}) == 1]
        print(f'INFO draw {dseed}: symbols sent to one symbol by all four ops: {fixed}')

    # Recipes, init pairing, rescoring of all 16 runs.
    rec_ok, init_ok, score_ok, table = True, True, True, {}
    for tid, Pm, L in tasks:
        held = sorted((a, b, L[a][b]) for a in range(4) for b in range(4)); train = [q for q in product(range(4), repeat=3) if q not in held]
        tr, he = rows_p(Pm, train), rows_p(Pm, held)
        lab = {(s, *q): compose_p(Pm, q)[s] for q in train for s in range(8)}
        notn = np.array([int(y) not in {lab[k] for j in range(3) for v in range(4) if v != q[j]
                                        for k in [(int(s), *[int(x) for x in q[:j]], v, *[int(x) for x in q[j+1:]])] if k in lab}
                         for s, q, y in zip(he['s'], he['ops'], he['states'][:, -1])])
        for init in (63200, 63201):
            first = {}
            for cond, w in (('inter', [1., 1., 1.]), ('final', [0., 0., 1.])):
                run = f'c2-gen-{cond}-{tid}-s{init}-a1'; base = f'{commit}:{LANE}/runs/{run}'
                c = json.loads(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{base}/config.json']))
                rec_ok &= (c['task_id'] == tid and c['step_weights'] == w and c['init_seed'] == init and c['updates'] == 10000 and c['lr'] == 1e-3
                           and c['wd'] == .1 and c['hidden'] == 128 and c['resume'] is None and c['scheme'] == dict(transition=True, readout=True))
                m = [json.loads(l) for l in subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{base}/metrics.jsonl']).decode().splitlines()]
                first[cond] = {k: m[0][k] for k in ('step', 'train', 'held', 'held_final_pred')}
                p = nets.load(io.BytesIO(subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{base}/ckpt-10000.npz'])))[0]
                ph = my_forward(p, he['s'], he['ops'], [1, 2, 3])[1][-1].argmax(1); ok = ph == he['states'][:, -1]
                pt = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[1][-1].argmax(1) == tr['states'][:, -1]
                last = m[-1]
                score_ok &= (last['step'] == 10000 and ''.join(map(str, ph)) == last['held_final_pred'] and ok.mean() == last['held']['final']['acc']
                             and pt.mean() == last['train']['final']['acc'] and last['held_subset']['notN'] == ok[notn].mean()
                             and last['held_subset']['n_notN'] == int(notn.sum()))
                table[tid, init, cond] = (round(float(ok.mean()), 3), round(float(ok[notn].mean()), 3), int(notn.sum()), round(float(pt.mean()), 3))
            init_ok &= first['inter'] == first['final']
    check('Part B recipes: 16 runs, tied/tied, inter 1 1 1 / final 0 0 1, task and init seed as in the ID, lr 1e-3, wd 0.1, 10,000 updates, no resume', rec_ok)
    check('Part B init pairing: within each (task, init seed), inter and final have identical step-0 rows (same initial function)', init_ok)
    check('Part B: 16 ckpt-10000 rescored on their own task: held-out string, held-out, train, notN (my recount) equal the last logged row', score_ok)
    for k, v in table.items():
        print(f'INFO {k}: held | notN (n) | train = {v}')

    # CX-G2: is any draw a relabelling of another (symbols by sigma in S8, ops by pi in S4)?
    S = np.array(list(permutations(range(8))))
    D = [('orig 51000', np.array(P))] + [(f'draw {s}', np.array(Pm)) for s, Pm in draws]
    def relabellings(A, B):
        n = 0
        for pi in permutations(range(4)):
            ok = np.ones(len(S), bool)
            for k in range(4):
                ok &= (S[:, A[k]] == B[pi[k]][S]).all(1)  # sigma(A_k(y)) = B_pi(k)(sigma(y)) for all y
            n += int(ok.sum())
        return n
    res = {(a, b): relabellings(A, B) for i, (a, A) in enumerate(D) for b, B in D[i:]}
    print('INFO number of (sigma, pi) with sigma A_k sigma^-1 = B_pi(k) for all k:', res)
    check('CX-G2 prediction: no two of the original and the two new draws are relabellings of each other (each draw relates only to itself)',
          all(v == 0 for (a, b), v in res.items() if a != b))
    print('INFO automorphisms per draw (incl. identity):', {a: v for (a, b), v in res.items() if a == b},
          '-> with only the identity, two distinct squares on one draw are not relabellings of each other')



def a5(time_commit):
    """NOTE A5 (time lane ckpt-20000..50000 of tt-inter and tt-final), plus the per-row baseline for
    'another op applied to the true state' on the original draw. Post hoc, descriptive (REVIEW.md G5, G6)."""
    git = lambda path: subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{time_commit}:research/composition-c2/time/runs/{path}'])
    D = {L: oracle_rows(list(product(range(4), repeat=L))) for L in range(1, 7)}
    ok = True
    for arm in ('tt-inter', 'tt-final'):
        for s in (51200, 51201, 51202):
            acc = {}
            for step in (20000, 30000, 40000, 50000):
                p = nets.load(io.BytesIO(git(f'c2-time-{arm}-s{s}-a1/ckpt-{step}.npz')))[0]
                acc[step] = [float((my_forward(p, d['s'], d['ops'], [1]*L)[1][-1].argmax(1) == d['states'][:, -1]).mean()) for L, d in D.items()]
            if arm == 'tt-inter':
                ok &= all(a == 1. for a in acc[50000])
            print(f'INFO {arm} s{s}: L = 1..6 at 50k {[round(a, 3) for a in acc[50000]]}; L = 6 at 20k/30k/40k/50k {[round(acc[k][5], 4) for k in sorted(acc)]}')
    check('A5: the time lane\'s tied/inter ckpt-50000 answers every row at every length 1-6 in all three seeds', ok)
    b = [len({P[k2][x] for k2 in range(4) if k2 != k} - {P[k][x]}) / 7 for x in range(8) for k in range(4)]
    print(f'INFO original draw: share of the 7 wrong symbols that are another op applied to the true state, mean over (state, op) {np.mean(b):.3f} '
          f'(min {min(b):.3f}); the NOTE uses 3/7 = 0.429')


def avg_ranks(x):
    """Ranks 0..n-1 with ties given their mean rank."""
    x = np.asarray(x, float); r = np.empty(len(x)); r[np.argsort(x, kind='mergesort')] = np.arange(len(x))
    _, inv = np.unique(x, return_inverse=True)
    return (np.bincount(inv, r) / np.bincount(inv))[inv]


def anatomy(commit):
    """Follow-up anatomy, drift columns and CX-G3 (QUESTION.md Revisions, generalization follow-up anatomy and drift)."""
    _, _, tasks = draws_and_splits()
    git = lambda path: subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{LANE}/{path}'])
    models = [(f'r1-comp-rnn-{c}-s{s}-a1', 'ops51000-ls51100', P, [tuple(q) for q in HELD], c,
               lambda r: nets.load(C2/'recovered'/r/'ckpt-10000.npz')[0]) for c in ('inter', 'final') for s in (51200, 51201, 51202)]
    models += [(f'c2-gen-{c}-{tid}-s{i}-a1', tid, Pm, [(a, b, L[a][b]) for a in range(4) for b in range(4)], c,
                lambda r: nets.load(io.BytesIO(git(f'runs/{r}/ckpt-10000.npz')))[0]) for tid, Pm, L in tasks for i in (63200, 63201) for c in ('inter', 'final')]
    steps = {(r['run_id'], r['step']): r for f in ('c2-gen-frozen-orig-a2', 'c2-gen-frozen-rep-a1')
             for r in csv.DictReader(io.StringIO(git(f'runs/{f}/steps.csv').decode()))}
    summ = json.loads(git('runs/c2-gen-followup-anatomy-a1/summary.json'))
    anat = {(r['run_id'], int(r['state_before']), int(r['op'])): r for r in csv.DictReader(io.StringIO(git('runs/c2-gen-followup-anatomy-a1/anatomy.csv').decode()))}
    wins = {(r['run_id'], int(r['length']), r['contains_held_out_window'] == 'True'): r
            for r in csv.DictReader(io.StringIO(git('runs/c2-gen-followup-anatomy-a1/windows.csv').decode()))}
    fig = {r['run_id']: r for r in csv.DictReader(io.StringIO(git('figures/followup.csv').decode()))}
    drift_ok, anat_ok, sum_ok, win_ok, fig_ok, n_steps = True, True, True, True, True, 0
    S, D6, AFTER, PRE, GAP, CX = {}, {}, {}, {}, {}, {}
    for run, tid, Pm, held, cond, load in models:
        p = load(run); hs = set(held); Pa = np.array(Pm)
        train = [q for q in product(range(4), repeat=3) if q not in hs]
        tr, d = rows_p(Pm, train), rows_p(Pm, list(product(range(4), repeat=6)))
        h3 = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[0][3]
        cent = np.stack([h3[tr['states'][:, -1] == y].mean(0) for y in range(8)])
        rn, rd = np.median(np.linalg.norm(h3, axis=1)), np.median(np.linalg.norm(h3 - cent[tr['states'][:, -1]], axis=1))
        H, Z = my_forward(p, d['s'], d['ops'], [1] * 6)
        sym = np.concatenate([d['s'][:, None], d['states']], 1)
        dist = [np.linalg.norm(H[t] - cent[sym[:, t]], axis=1) for t in range(7)]
        for t in range(7):
            r = steps[run, str(t)]; n_steps += 1
            mine = (np.median(np.linalg.norm(H[t], axis=1)) / rn, np.median(dist[t]) / rd)
            drift_ok &= abs(mine[0] - float(r['norm_ratio'])) < 1e-9 and abs(mine[1] - float(r['centroid_distance_ratio'])) < 1e-9
        D6[run] = (np.median(np.linalg.norm(H[6], axis=1)) / rn, np.median(dist[6]) / rd)
        pred = np.stack([z.argmax(1) for z in Z], 1); ok = pred == d['states']
        first = np.where(ok.all(1), 0, (~ok).argmax(1) + 1)
        AFTER[run] = [(int((first == t).sum()), int((pred[first == t, t] == d['states'][first == t, t]).sum()),
                       int((pred[first == t, t] == Pa[d['ops'][first == t, t], pred[first == t, t-1]]).sum())) for t in (4, 5)]
        if cond != 'inter':
            continue
        err, risk, wrong_op, win, win_r, nf, nr = np.zeros((32, 3), int), np.zeros((32, 3), int), 0, 0, 0, 0, 0
        cx = {}
        for j, t in enumerate((4, 5, 6)):
            at = (first == 0) | (first >= t); fe = first == t
            tr_id = sym[:, t-1] * 4 + d['ops'][:, t-1]
            err[:, j] = np.bincount(tr_id[fe], minlength=32); risk[:, j] = np.bincount(tr_id[at], minlength=32)
            x, k, yh = sym[fe, t-1], d['ops'][fe, t-1], pred[fe, t-1]
            wrong_op += int(sum(any(Pa[kk, xx] == yy for kk in range(4) if kk != kx) for xx, kx, yy in zip(x, k, yh)))
            hw = np.array([tuple(int(v) for v in q[t-3:t]) in hs for q in d['ops']])
            win += int(hw[fe].sum()); win_r += int(hw[at].sum()); nf += int(fe.sum()); nr += int(at.sum())
            if t >= 5:   # CX-G3: drift quartile of h_{t-1} within each transition, among rows at risk
                q = np.zeros(len(first), int)
                for c in range(32):
                    idx = np.where(at & (tr_id == c))[0]
                    q[idx[np.argsort(dist[t-1][idx], kind='mergesort')]] = np.arange(len(idx)) * 4 // len(idx)
                cx[t] = [fe[at & (q == b)].mean() for b in range(4)]
        CX[run] = cx
        for x in range(8):
            for k in range(4):
                a = anat[run, x, k]
                anat_ok &= all(int(a[f'first_errors_step{t}']) == err[x*4+k, j] and int(a[f'at_risk_step{t}']) == risk[x*4+k, j] for j, t in enumerate((4, 5, 6)))
        pooled = np.sort(err.sum(1))[::-1]; sh = lambda n: pooled[:n].sum() / pooled.sum()
        rate = err / risk
        rho = lambda a, b: float(np.corrcoef(avg_ranks(a), avg_ranks(b))[0, 1])
        m = dict(first_errors_steps_4_6=int(err.sum()), top1_share=sh(1), top4_share=sh(4), top8_share=sh(8),
                 spearman_rate_step5_vs_step6=rho(rate[:, 1], rate[:, 2]), spearman_rate_step4_vs_step5=rho(rate[:, 0], rate[:, 1]) if err[:, 0].sum() >= 20 else None,
                 wrong_answer_is_other_op_on_true_state=wrong_op / nf, first_error_window_is_held_out=win / nf, at_risk_window_is_held_out=win_r / nr)
        S[run] = m; L_ = summ['per_model'][run]
        sum_ok &= all((m[k_] is None and L_[k_] is None) or (m[k_] is not None and L_[k_] is not None and abs(m[k_] - L_[k_]) < 1e-12) for k_ in m)
        sum_ok &= (L_['first_errors_by_step'] == {str(t): int(err[:, j].sum()) for j, t in enumerate((4, 5, 6))} and L_['F1_top4_ge_0p5'] == (m['top4_share'] >= .5)
                   and L_['F2_rho_gt_0p5'] == (m['spearman_rate_step5_vs_step6'] > .5) and L_['F3_window_share_ge_0p375'] == (m['first_error_window_is_held_out'] >= .375))
        acc = {}
        for L in (4, 5, 6):
            dl = rows_p(Pm, list(product(range(4), repeat=L)))
            okl = my_forward(p, dl['s'], dl['ops'], [1] * L)[1][-1].argmax(1) == dl['states'][:, -1]
            anyw = np.array([any(tuple(int(v) for v in q[j:j+3]) in hs for j in range(L - 2)) for q in dl['ops']])
            pre = np.array([tuple(int(v) for v in q[:3]) in hs for q in dl['ops']])
            for flag in (True, False):
                w = wins[run, L, flag]; acc[L, flag] = okl[anyw == flag].mean()
                win_ok &= int(w['n_rows']) == int((anyw == flag).sum()) and abs(float(w['accuracy']) - acc[L, flag]) < 1e-12
            GAP[run, L] = acc[L, False] - acc[L, True]
            if L == 6:
                PRE[run] = (okl[pre].mean(), okl[~pre].mean())
        g = fig[run]
        fig_ok &= (abs(float(g['top4_share']) - m['top4_share']) < 1e-12 and abs(float(g['first_error_window_is_held_out']) - m['first_error_window_is_held_out']) < 1e-12
                   and abs(float(g['at_risk_window_is_held_out']) - m['at_risk_window_is_held_out']) < 1e-12
                   and abs(float(g['length6_acc_with_held_window']) - acc[6, True]) < 1e-12 and abs(float(g['length6_acc_without_held_window']) - acc[6, False]) < 1e-12)
    check(f'drift: norm_ratio and centroid_distance_ratio of steps.csv ({n_steps} rows: 22 frozen models x steps 0-6) equal mine', drift_ok)
    check('follow-up anatomy.csv: first errors and at-risk counts at steps 4, 5, 6 for 11 models x 32 transitions equal mine', anat_ok)
    check('follow-up summary.json: counts by step, top-1/4/8 shares, Spearman rho (5 vs 6; 4 vs 5 where >= 20 step-4 errors), other-op share, '
          'window shares at the error and at risk, and the F1-F3 flags equal mine for all 11 models', sum_ok)
    check('follow-up windows.csv: row counts and accuracies with and without a held-out window at L = 4-6 equal mine (11 models)', win_ok)
    check('figures/followup.csv: all 11 rows equal mine', fig_ok and len(fig) == 11)
    rng = lambda key, runs=S: (min(S[r][key] for r in runs if S[r][key] is not None), max(S[r][key] for r in runs if S[r][key] is not None))
    near = lambda v, a, b: abs(v[0] - a) <= .005 + 1e-9 and abs(v[1] - b) <= .005 + 1e-9
    nums = {'top4_share': (.20, .29), 'top8_share': (.36, .50), 'spearman_rate_step5_vs_step6': (.62, .96), 'spearman_rate_step4_vs_step5': (.08, .80),
            'first_error_window_is_held_out': (.31, .41), 'wrong_answer_is_other_op_on_true_state': (.62, .71)}
    for k_, (a, b) in nums.items():
        v = rng(k_); check(f'NOTE follow-up: {k_} over the 11 models is {a:.2f}-{b:.2f}', near(v, a, b), f'{v[0]:.3f}-{v[1]:.3f}')
    check('NOTE follow-up: F1 met in 0 of 11, F2 in 11 of 11, F3 in 3 of 11; at-risk window share exactly 0.25 in every model (Latin square)',
          sum(m['top4_share'] >= .5 for m in S.values()) == 0 and sum(m['spearman_rate_step5_vs_step6'] > .5 for m in S.values()) == 11
          and sum(m['first_error_window_is_held_out'] >= .375 for m in S.values()) == 3 and all(m['at_risk_window_is_held_out'] == .25 for m in S.values()))
    orig = [f'r1-comp-rnn-inter-s{s}-a1' for s in (51200, 51201, 51202)]
    g4 = [GAP[r, 4] for r in orig]; g56 = [GAP[r, L] for r in orig for L in (5, 6)]
    check('NOTE held-out-window bullet (original three): with-window programs less accurate by 0.01-0.10 at L = 4 and 0.05-0.14 at L = 5-6',
          min(g4 + g56) > 0 and near((min(g4), max(g4)), .01, .10) and near((min(g56), max(g56)), .05, .14), f'{min(g4):.3f}-{max(g4):.3f}; {min(g56):.3f}-{max(g56):.3f}')
    pre_note = [(.81, .87), (.50, .58), (.65, .74)]
    check('NOTE: L = 6 accuracy, held-out prefix vs not (original three): 0.81 vs 0.87, 0.50 vs 0.58, 0.65 vs 0.74',
          all(near(PRE[r], *v) for r, v in zip(orig, pre_note)), str([tuple(round(float(x), 3) for x in PRE[r]) for r in orig]))
    wo = [fig[r]['length6_acc_without_held_window'] for r in S]
    check('NOTE: programs without a held-out window fall to 0.64-0.93 at L = 6 (11 models)', near((min(map(float, wo)), max(map(float, wo))), .64, .93),
          f'{min(map(float, wo)):.3f}-{max(map(float, wo)):.3f}')
    for cond, note in (('inter', ((1.02, 1.17), (1.03, 1.19))), ('final', ((1.4, 1.6), (1.5, 1.7)))):
        rs = [f'r1-comp-rnn-{cond}-s{s}-a1' for s in (51200, 51201, 51202)]; allr = [r for r in D6 if f'-{cond}-' in r]
        v = [(min(D6[r][i] for r in rs), max(D6[r][i] for r in rs)) for i in (0, 1)]
        tol = .005 if cond == 'inter' else .05
        check(f'NOTE drift at step 6, tied/{cond} (original three): norm {note[0][0]}-{note[0][1]}x, centroid distance {note[1][0]}-{note[1][1]}x the training-h3 medians',
              all(abs(v[i][0] - note[i][0]) <= tol + 1e-9 and abs(v[i][1] - note[i][1]) <= tol + 1e-9 for i in (0, 1)),
              f'{v[0][0]:.3f}-{v[0][1]:.3f}, {v[1][0]:.3f}-{v[1][1]:.3f}; all {len(allr)} {cond} models: norm {min(D6[r][0] for r in allr):.2f}-{max(D6[r][0] for r in allr):.2f}, '
              f'distance {min(D6[r][1] for r in allr):.2f}-{max(D6[r][1] for r in allr):.2f}')
    fr = [(rc / n, fo / n, (n - rc - fo) / n) for r in orig for n, rc, fo in AFTER[r]]   # per seed and step (4, 5); pooling does not give the NOTE's ranges
    check('NOTE: after a first error at step 4 or 5 (original three, per seed and step): next step right again 27-38%, follows own error 38-41%, neither 21-34%',
          all(near((min(f_[i] for f_ in fr), max(f_[i] for f_ in fr)), a, b) for i, (a, b) in enumerate(((.27, .38), (.38, .41), (.21, .34)))),
          '; '.join(f'{a:.3f}/{b:.3f}/{c:.3f}' for a, b, c in fr))

    print('CX-G3: first-error rate at step t among rows at risk, by within-transition drift quartile of h_{t-1} (q1 lowest .. q4 highest); ratio q4/q1')
    hold = 0
    for r, cx in CX.items():
        ratios = {t: (cx[t][3] / cx[t][0] if cx[t][0] > 0 else np.inf) for t in (5, 6)}
        hold += all(v >= 2 for v in ratios.values())
        print(f'  {r}: ' + '; '.join(f't={t}: ' + ' '.join(f'{v:.3f}' for v in cx[t]) + f' (x{ratios[t]:.1f})' for t in (5, 6)))
    check(f'CX-G3 prediction: top-quartile rate >= 2x bottom-quartile at both t = 5 and 6 in >= 9 of 11 models ({hold} of 11)', hold >= 9)


if __name__ == '__main__':
    main(sys.argv[1])
    if len(sys.argv) > 2:
        partb(sys.argv[2])
    if len(sys.argv) > 3:
        a5(sys.argv[3])
    if len(sys.argv) > 4:
        anatomy(sys.argv[4])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
