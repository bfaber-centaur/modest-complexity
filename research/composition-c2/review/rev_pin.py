"""c2 pin check (QUESTION.md section 1; REVIEW.md section 1). Independent of the pin's own
tests: the task oracle, the forward pass and the loss are written here from the r1 tables
(copied as data from research/regimes/composition/QUESTION.md) and the model's definition.
Code under test is imported read-only; c2train.py is also run as a subprocess in a temporary
directory for the resume-guard checks (one update each, not research runs).

uv run --locked python research/composition-c2/review/rev_pin.py RUN_ROOT
(RUN_ROOT holds the reviewer run c2-rev-verify-final-s51200-a1)
"""
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np

HERE = Path(__file__).resolve().parent
C2, ROOT = HERE.parent, HERE.parents[2]
sys.path[:0] = [str(C2/'common'), str(ROOT/'research'/'regimes'/'common')]
import c2probe  # noqa: E402
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import c2train  # noqa: E402
import nets  # noqa: E402

FAIL = []
P = [[7, 0, 1, 2, 4, 6, 5, 3], [5, 0, 7, 2, 6, 1, 3, 4], [6, 3, 0, 5, 1, 2, 7, 4], [6, 1, 4, 3, 0, 5, 7, 2]]
LSQ = [[0, 1, 3, 2], [3, 2, 0, 1], [2, 0, 1, 3], [1, 3, 2, 0]]
HELD = sorted((a, b, LSQ[a][b]) for a in range(4) for b in range(4))
TRAIN = [q for q in product(range(4), repeat=3) if q not in HELD]
SCHEMES = [(tt, tr) for tt in (True, False) for tr in (True, False)]
RUNS = {f'r1-comp-rnn-{c}-s{s}-a1': ('composition', 'MANIFEST.json') for c in ('inter', 'final') for s in (51200, 51201, 51202)}
RUNS.update({f'r1-rev-comp-rnn-untied-s{s}-a1': ('review', 'MANIFEST.sha256') for s in (51200, 51201, 51202)})


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def oracle_rows(progs):
    s_, q_, st_ = [], [], []
    for q in progs:
        for s in range(8):
            x, states = s, []
            for k in q:
                x = P[k][x]; states.append(x)
            s_.append(s); q_.append(list(q)); st_.append(states)
    return dict(s=np.array(s_), ops=np.array(q_), states=np.array(st_))


def get(p, name, k):
    return p[name] if name in p else p[f'{name}{k}']


def my_forward(p, s, ops, smap):
    h = np.maximum(np.eye(8)[s] @ p['Ws'] + p['bs'], 0); H, Z = [h], []
    for t, k in enumerate(smap):
        h = np.maximum(h @ get(p, 'A', k) + np.eye(4)[ops[:, t]] @ get(p, 'B', k) + get(p, 'bh', k), 0)
        H.append(h); Z.append(h @ get(p, 'V', k) + get(p, 'bv', k))
    return H, Z


def my_loss(p, s, ops, ys, w):
    total = 0.
    for t, z in enumerate(my_forward(p, s, ops, [1, 2, 3])[1]):
        lse = np.logaddexp.reduce(z, axis=1)
        total += w[t] * np.mean(lse - z[np.arange(len(s)), ys[t]])
    return total


def run_c2train(*argv):
    return subprocess.run([sys.executable, str(C2/'common'/'c2train.py'), *map(str, argv)], cwd=ROOT, capture_output=True, text=True)


def main(run_root):
    # 0. The pin's own tests, in their own process.
    t0 = time.time()
    r = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'research/composition-c2/common/test_c2.py'], cwd=ROOT, capture_output=True, text=True)
    check(f'test_c2.py in its own process: {r.stderr.strip().splitlines()[-3] if r.stderr else ""} {r.stderr.strip().splitlines()[-1]}',
          r.returncode == 0, f'{time.time() - t0:.1f} s')

    # 1. Task: load, rows of any length, N / notN.
    T = c2task.load()
    check('c2task.load(): permutations, Latin square, train/held programs and task seed equal the r1 tables',
          T['P'].tolist() == P and T['L'].tolist() == LSQ and [tuple(q) for q in T['train']] == TRAIN
          and [tuple(q) for q in T['held']] == HELD and T['task_seed_used'] == 51000)
    ok = True
    for L in range(1, 7):
        progs = list(product(range(4), repeat=L)); ok &= c2task.programs(L) == progs
        a, b = c2task.rows(T['P'], progs), oracle_rows(progs)
        ok &= all(np.array_equal(a[k], b[k]) for k in ('s', 'ops', 'states'))
    for progs in (TRAIN, HELD):
        a, b = c2task.rows(T['P'], progs), oracle_rows(progs)
        ok &= all(np.array_equal(a[k], b[k]) for k in ('s', 'ops', 'states'))
    check('c2task.rows and programs equal a plain-loop oracle for every row at lengths 1-6 and for the train/held programs', ok)
    tr, he = oracle_rows(TRAIN), oracle_rows(HELD)
    lab = {(int(s), *map(int, q)): int(y[-1]) for s, q, y in zip(tr['s'], tr['ops'], tr['states'])}
    mine = np.array([int(y[-1]) in {lab[k] for j in range(3) for v in range(4) if v != q[j]
                                    for k in [(int(s), *[int(x) for x in q[:j]], v, *[int(x) for x in q[j+1:]])] if k in lab}
                     for s, q, y in zip(he['s'], he['ops'], he['states'])])
    nbr, all_train = c2task.neighbour_answerable(T['P'], T['train'], T['held'])
    check(f'neighbour_answerable equals an independent recount (notN = {int((~mine).sum())} of 128; all neighbours are training rows)',
          np.array_equal(nbr, mine) and all_train)

    # 2. Init: r1's draw, one function across schemes, independent copies.
    rec = C2/'recovered'
    same = all(all(np.array_equal(c2rnn.init(128, s, True, tr_)[k], v) for k, v in nets.load(rec/run/'ckpt-0.npz')[0].items())
               and set(c2rnn.init(128, s, True, tr_)) == set(nets.load(rec/run/'ckpt-0.npz')[0])
               for run in RUNS for s, tr_ in [(int(run[-8:-3]), 'untied' not in run)])
    check('c2rnn.init equals every recovered r1 ckpt-0 (9 runs; tied/tied and tied/per-step), every array bitwise', same)
    d = c2task.rows(T['P'], T['train']); ref = c2rnn.forward(c2rnn.init(128, 51200), d['ops'], s=d['s'])
    ok = True
    for tt, tr_ in SCHEMES[1:]:
        p = c2rnn.init(128, 51200, tt, tr_)
        H, Z = c2rnn.forward(p, d['ops'], s=d['s'])
        ok &= all(np.array_equal(x, y) for x, y in zip(H + Z, ref[0] + ref[1]))
        for name in [n for n in ('A', 'B', 'bh', 'V', 'bv') if n not in p]:
            before = [p[f'{name}{k}'].copy() for k in (1, 2, 3)]
            p[f'{name}1'] += 1.
            ok &= np.array_equal(p[f'{name}2'], before[1]) and np.array_equal(p[f'{name}3'], before[2])
    check('all four schemes compute the same function at init; writing one copy in place leaves the other copies unchanged', ok)

    # 3. Loss value and gradients against a loss written here.
    r = np.random.default_rng(65300); sub = {k: v[::13] for k, v in d.items()}; ys = [sub['states'][:, t] for t in range(3)]
    worst_val, worst_fd, zero_ok, n_entries = 0., 0., True, 0
    for (tt, tr_) in SCHEMES:
        for w in ([1, 1, 1], [0, 0, 1], [.5, 0, 2], [2, 1, .25]):
            p = {k: v + r.normal(0, .3, v.shape) for k, v in c2rnn.init(5, 65301, tt, tr_).items()}
            total, _, g = c2rnn.loss_grad(p, sub['s'], sub['ops'], ys, w)
            worst_val = max(worst_val, abs(total - my_loss(p, sub['s'], sub['ops'], ys, w)))
            for k, v in p.items():
                for i in np.ndindex(v.shape):
                    old = v[i]; v[i] = old + 1e-6; up = my_loss(p, sub['s'], sub['ops'], ys, w)
                    v[i] = old - 1e-6; down = my_loss(p, sub['s'], sub['ops'], ys, w); v[i] = old
                    worst_fd = max(worst_fd, abs((up - down)/2e-6 - g[k][i])); n_entries += 1
            if not tr_:
                zero_ok &= all(not np.any(g[f'{n}{t+1}']) for t in range(3) if w[t] == 0 for n in ('V', 'bv'))
    check(f'loss value equals the one written here (4 schemes x 4 weight vectors incl. 0 and non-unit)', worst_val < 1e-12, f'max |diff| {worst_val:.1e}')
    check(f'every gradient entry equals central differences of the loss written here ({n_entries} entries)', worst_fd < 1e-7, f'worst abs error {worst_fd:.1e}')
    check('a weight-0 step gives its untied readout copy an exactly zero gradient', zero_ok)

    # 4. Any-length forward with a step map, and continuation from a hidden state.
    d5 = oracle_rows(list(product(range(4), repeat=5))[::29]); smap = [1, 2, 3, 3, 3]; worst = 0.
    for tt, tr_ in SCHEMES:
        p = {k: v + r.normal(0, .1, v.shape) for k, v in c2rnn.init(16, 65302, tt, tr_).items()}
        H, Z = c2rnn.forward(p, d5['ops'], s=d5['s'], step_map=smap); mH, mZ = my_forward(p, d5['s'], d5['ops'], smap)
        H2, Z2 = c2rnn.forward(p, d5['ops'][:, 2:], h0=mH[2], step_map=smap[2:])
        worst = max([worst] + [np.abs(a - b).max() for a, b in zip(H + Z + H2[1:] + Z2, mH + mZ + mH[3:] + mZ[2:])])
    check('length-5 forward with step map [1,2,3,3,3] and continuation from h2 equal the forward written here (4 schemes)', worst == 0, f'max |diff| {worst:.1e}')

    # 5. Recovered checkpoints: hashes, manifests, contents; no-change evaluation.
    recj = json.loads((rec/'RECOVERY.json').read_text())['files']
    pub = {}
    for run, (lane, man) in RUNS.items():
        mpath = ROOT/'research'/'regimes'/lane/'runs'/(run if man == 'MANIFEST.json' else '')/man
        if man == 'MANIFEST.json':
            pub.update({(run, m['file']): m['sha256'] for m in json.loads(mpath.read_text())})
        else:
            pub.update({tuple(l.split()[1].split('/', 1)): l.split()[0] for l in mpath.read_text().splitlines() if l.startswith(tuple('0123456789abcdef'))})
    ok, local = True, 0
    for e in recj:
        f = rec/e['run_id']/e['file']; h = hashlib.sha256(f.read_bytes()).hexdigest()
        p, st, _ = nets.load(f)
        ok &= h == e['sha256'] == pub[e['run_id'], e['file']] and st['t'] == int(e['file'][5:-4])
        ok &= c2rnn.scheme(p) == dict(transition=True, readout='untied' not in e['run_id'])
        orig = ROOT/'runs'/'regimes'/e['run_id']/e['file']
        if orig.exists():
            local += 1; ok &= orig.read_bytes() == f.read_bytes()
    listed = {(e['run_id'], e['file']) for e in recj}
    on_disk = {(q.parent.name, q.name) for q in rec.glob('*/*')}
    check(f'RECOVERY.json: {len(recj)} files; committed bytes = recorded SHA-256 = r1 published manifest; step and scheme in each file as named; '
          f'{local} also byte-equal to originals on this container; nothing unlisted', ok and len(recj) == 27 and listed == on_disk)
    data = dict(train=c2task.rows(T['P'], T['train']), held=c2task.rows(T['P'], T['held']))
    ok, notn = True, {}
    for run, (lane, _) in RUNS.items():
        p = nets.load(rec/run/'ckpt-10000.npz')[0]
        row = c2train.evaluate(p, data, nbr)[0]
        last = json.loads((ROOT/'research'/'regimes'/lane/'runs'/run/'metrics.jsonl').read_text().splitlines()[-1])
        ok &= last['step'] == 10000 and all(row[k] == last[k] for k in ('train', 'held', 'held_final_pred'))
        notn[run] = round(row['held_subset']['notN'], 3)
    check('no-change evaluation: c2train.evaluate on all 9 recovered ckpt-10000 equals the last published r1 row exactly (train, held, held_final_pred)', ok)
    print('INFO notN accuracy at 10,000 from c2train.evaluate:', notn)

    # 6. Resume guards (code checks in a temporary directory; one update each).
    src = rec/'r1-comp-rnn-final-s51200-a1'/'ckpt-10000.npz'; sha = hashlib.sha256(src.read_bytes()).hexdigest()
    usrc = rec/'r1-rev-comp-rnn-untied-s51200-a1'/'ckpt-10000.npz'; usha = hashlib.sha256(usrc.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); base = ['--init-seed', 51200, '--updates', 10001]
        bad_hash = run_c2train('--out', tmp/'a', *base, '--resume', src, '--resume-sha256', '0'*64, '--step-weights', 0, 0, 1)
        bad_scheme = run_c2train('--out', tmp/'b', *base, '--resume', usrc, '--resume-sha256', usha)
        check('resume refuses a wrong SHA-256 and a scheme that differs from the checkpoint', bad_hash.returncode != 0 and bad_scheme.returncode != 0)
        wrong_w = run_c2train('--out', tmp/'c', *base, '--resume', src, '--resume-sha256', sha)
        wrong_t = run_c2train('--out', tmp/'d', *base, '--resume', src, '--resume-sha256', sha, '--step-weights', 0, 0, 1,
                              '--task-id', 'ops63000-ls63100', '--lr', '3e-3', '--wd', '0', '--init-seed', 9)
        cfg = lambda x: json.loads((tmp/x/'config.json').read_text()) if (tmp/x/'config.json').exists() else {}
        print(f'INFO resume of r1 rnn-final ckpt-10000 with default step weights: exit {wrong_w.returncode}, recorded step_weights '
              f'{cfg("c").get("step_weights")} (source condition: {nets.load(src)[2].get("condition")})')
        print(f'INFO resume with another task id, lr, wd and init seed: exit {wrong_t.returncode}, recorded task '
              f'{cfg("d").get("task", {}).get("id")}, lr {cfg("d").get("lr")}, wd {cfg("d").get("wd")}, init seed {cfg("d").get("init_seed")}')
        check('EXPECTED GAP: resume compares the source run\'s recipe (step weights, task, lr, wd, init seed)',
              wrong_w.returncode != 0 and wrong_t.returncode != 0, 'accepted silently; see REVIEW.md C1')

    # 7. Cross-host resume (reviewer run).
    run = Path(run_root)/'c2-rev-verify-final-s51200-a1'
    keys = ('step', 'train', 'held', 'held_final_pred')
    ours = [{k: x[k] for k in keys} for x in map(json.loads, (run/'metrics.jsonl').read_text().splitlines())]
    theirs = [{k: x[k] for k in keys} for x in map(json.loads, (ROOT/'research/regimes/composition/runs/r1-comp-rnn-final-s51200-a1/metrics.jsonl').read_text().splitlines())
              if x['step'] >= 1000]
    (pa, sa, _), (pb, sb, _) = nets.load(run/'ckpt-10000.npz'), nets.load(rec/'r1-comp-rnn-final-s51200-a1'/'ckpt-10000.npz')
    arrays = sa['t'] == sb['t'] and pa.keys() == pb.keys() and all(np.array_equal(x[k], y[k]) for x, y in ((pa, pb), (sa['m'], sb['m']), (sa['v'], sb['v'])) for k in x)
    check(f'cross-host resume on this host: {len(ours)} logged rows 1,000-10,000 equal r1 published; ckpt-10000 weights, both moments and t equal the recovered original',
          ours == theirs and arrays)
    rt = json.loads((run/'runtime.json').read_text())
    print(f'INFO calibration on this host (1 job, tied/tied, step weights 0 0 1): {1000*rt["phases"]["update"]["wall_seconds"]/rt["phases"]["update"]["calls"]:.2f} ms per update, '
          f'peak RSS {rt["peak_rss_bytes"]/2**20:.0f} MiB, wall {rt["wall_seconds"]:.1f} s')


    # 8. c2probe (QUESTION.md Revisions, 04:06): labels, r1's probe recipe, table orientation.
    sys.path.insert(0, str(ROOT/'research'/'regimes'/'composition'))
    import analyze  # r1's, read-only
    p = nets.load(rec/'r1-comp-rnn-inter-s51200-a1'/'ckpt-10000.npz')[0]
    out = c2probe.decode_table(p, T, 64300)
    lab = lambda d: [d['s'], *d['states'].T]
    ok = all(np.array_equal(a, b) for d, q in ((tr, c2task.rows(T['P'], T['train'])), (he, c2task.rows(T['P'], T['held'])))
             for a, b in zip(lab(d), c2probe.symbols(q)))
    Htr, Hhe = my_forward(p, tr['s'], tr['ops'], [1, 2, 3])[0], my_forward(p, he['s'], he['ops'], [1, 2, 3])[0]
    for i in range(4):
        q = c2probe.fit_probe(Htr[i], lab(tr)[i], 64300)
        W, a_tr, a_he = analyze.fit_probe(Htr[i], lab(tr)[i], Hhe[i], lab(he)[i], 64300)
        mine = [float(np.mean((Hhe[j] @ q['W_y'] + q['b_y']).argmax(1) == lab(he)[j])) for j in range(4)]
        ok &= np.array_equal(W, q['W_y']) and out['table'][i] == mine and out['train_fit'][i] == a_tr and mine[i] == a_he
    check('c2probe.decode_table (inter s51200 ckpt-10000, probe seed 64300): labels, r1 probe weights bitwise, every entry re-scored, '
          'orientation [fitted][scored], diagonal = r1 held-out probe accuracy', ok)
    print('INFO decode table rows = fitted position 0..3, columns = scored position:', [[round(x, 3) for x in r] for r in out['table']],
          'train fit', [round(x, 3) for x in out['train_fit']])


if __name__ == '__main__':
    main(sys.argv[1])
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
