"""Recompute R1's M4 numbers from published artifacts, and score the reviewer's
two counterexamples (QUESTION.md, Revisions; REVIEW.md §1).

Labels, held-out rows and neighbours come from oracle_composition (written from
the brief, not from the lane's code). Checkpoints are read with plain np.load
and run through forward passes written here.

uv run --locked python research/regimes/review/recompute_composition.py LANE_COMMIT RUN_ROOT
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import numpy as np

from lanes import check, FAIL, ROOT
from oracle_composition import P, TRAIN, HELD, run

SEEDS = (51200, 51201, 51202)
MLP = [(c, s) for c in ('final', 'inter', 'shuf') for s in SEEDS]
RNN = [(c, s) for c in ('rnn-final', 'rnn-inter') for s in SEEDS]
rid = lambda c, s: f'r1-comp-{c}-s{s}-a1'


def published(commit):
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'archive', commit, 'research/regimes/composition'])
    t = tarfile.open(fileobj=io.BytesIO(raw))
    return {Path(m.name).relative_to('research/regimes/composition').as_posix(): t.extractfile(m).read()
            for m in t.getmembers() if m.isfile()}


def rows(progs):
    R = np.array([(s, *q, *(run(q[:k], s) for k in (1, 2, 3))) for q in progs for s in range(8)])
    return dict(zip(('s', 'a', 'b', 'c', 'step1', 'step2', 'final'), R.T))


def onehot(v, n):
    return np.eye(n)[v]


def mlp_logits(z, d):
    X = np.hstack([onehot(d['s'], 8)] + [onehot(d[k], 4) for k in 'abc'])
    h = np.maximum(np.maximum(X @ z['p.W1'] + z['p.b1'], 0) @ z['p.W2'] + z['p.b2'], 0)
    return {k[4:]: h @ z[k] + z['p.b_' + k[4:]] for k in z.files if k.startswith('p.W_')}


def rnn_states(z, d):
    H = [np.maximum(onehot(d['s'], 8) @ z['p.Ws'] + z['p.bs'], 0)]
    for k in 'abc':
        H.append(np.maximum(H[-1] @ z['p.A'] + onehot(d[k], 4) @ z['p.B'] + z['p.bh'], 0))
    return H


tr, he = rows(TRAIN), rows(sorted(HELD))   # the lane's order: programs lexicographic, then start symbol


def neighbours():
    """Per held-out row: {slot: final labels of the training rows that differ from it in that op slot only}."""
    lookup = {tuple(x): y for x, y in zip(np.stack([tr[k] for k in 'sabc'], 1).tolist(), tr['final'])}
    out, all_train = [], True
    for x in np.stack([he[k] for k in 'sabc'], 1).tolist():
        nb = {}
        for j, slot in zip((1, 2, 3), 'abc'):
            keys = [tuple(x[:j] + [v] + x[j+1:]) for v in range(4) if v != x[j]]
            all_train &= all(k in lookup for k in keys)
            nb[slot] = {lookup[k] for k in keys if k in lookup}
        out.append(nb)
    return out, all_train


NB, nb_all_train = neighbours()
N_ANY = np.array([y in set().union(*n.values()) for y, n in zip(he['final'], NB)])


def split_acc(pred):
    ok = pred == he['final']
    return ok.mean(), ok[N_ANY].mean(), ok[~N_ANY].mean()


def main(commit, root):
    F = published(commit)
    bad = [(r, m['file']) for r in sorted({p.split('/')[1] for p in F if p.startswith('runs/') and p.endswith('MANIFEST.json')})
           for m in json.loads(F[f'runs/{r}/MANIFEST.json']) if m['committed']
           and hashlib.sha256(F[f'runs/{r}/{m["file"]}']).hexdigest() != m['sha256']]
    check('every committed file in every published MANIFEST.json matches its SHA-256', not bad, str(bad[:3]))
    M = {(c, s): [json.loads(l) for l in F[f'runs/{rid(c, s)}/metrics.jsonl'].decode().splitlines()] for c, s in MLP + RNN}
    for c in ('full', 'interp'):
        M[c, 51200] = [json.loads(l) for l in F[f'runs/{rid(c, 51200)}/metrics.jsonl'].decode().splitlines()]

    # 1. Published ckpt-10000 rescored with the forward pass above.
    for c in ('final', 'inter', 'shuf'):
        z = np.load(io.BytesIO(F[f'runs/{rid(c, 51200)}/ckpt-10000.npz'])); last = M[c, 51200][-1]
        lt, lh = mlp_logits(z, tr), mlp_logits(z, he)
        same = all(float(np.mean(lg[h].argmax(1) == d[h])) == last[sp][h]['acc'] for sp, d, lg in (('train', tr, lt), ('held', he, lh))
                   for h in lg)
        check(f'{rid(c, 51200)}: ckpt-10000 rescored here equals the last logged accuracies ({sorted(lh)}) and held_final_pred',
              same and ''.join(map(str, lh['final'].argmax(1))) == last['held_final_pred'])

    # 2. Every run's logged held-out predictions, scored against the oracle's labels; NOTE's tables.
    worst = max(abs(np.mean(np.array(list(m[-1]['held_final_pred']), int) == he['final']) - m[-1]['held']['final']['acc'])
                for m in M.values())
    check(f'{len(M)} runs: logged held_final_pred scored against oracle labels equals the logged held-out final accuracy',
          worst == 0 and all(m[-1]['step'] == 10000 for m in M.values()))
    for c in ('final', 'inter', 'shuf', 'rnn-inter', 'rnn-final'):
        print(f'INFO held-out final at 10,000, {c:9s}', [round(M[c, s][-1]['held']['final']['acc'], 3) for s in SEEDS])
    print('INFO P1 full: train final', M['full', 51200][-1]['train']['final']['acc'], 'held final', M['full', 51200][-1]['held']['final']['acc'],
          '| P2 interp: held final', M['interp', 51200][-1]['held']['final']['acc'])
    first = lambda m, f: next((r['step'] for r in m if f(r)), None)
    print('INFO MLP: first step with train final acc 1.0', {f'{c}-{s}': first(M[c, s], lambda r: r['train']['final']['acc'] == 1.) for c, s in MLP})
    for s in SEEDS:
        early = [(r['step'], r['held']['final']['acc']) for r in M['final', s] if 100 <= r['step'] <= 500]
        print(f'INFO final-{s}: held-out final over updates 100-500: min {min(a for _, a in early):.3f}, max {max(a for _, a in early):.3f}; '
              f'at 5,000 {next(r["held"]["final"]["acc"] for r in M["final", s] if r["step"] == 5000):.3f}')
    for c in ('inter', 'shuf'):
        print(f'INFO {c}: held-out step1, step2 at 10,000', [(M[c, s][-1]['held']['step1']['acc'], round(M[c, s][-1]['held']['step2']['acc'], 3)) for s in SEEDS])
    for s in SEEDS:
        m = M['rnn-inter', s]; at = {r['step']: r['held']['final']['acc'] for r in m}
        print(f'INFO rnn-inter-{s}: train final 1.0 first at {first(m, lambda r: r["train"]["final"]["acc"] == 1.)}; '
              f'held-out final at 1k/4k/7k/10k {[round(at[k], 3) for k in (1000, 4000, 7000, 10000)]}')
        m = M['rnn-final', s]
        print(f'INFO rnn-final-{s}: shared readout on h1, h2 at 10,000: step1 {m[-1]["held"]["step1"]["acc"]:.3f}, step2 {m[-1]["held"]["step2"]["acc"]:.3f} '
              f'(equal to the training-row accuracies: {all(m[-1]["train"][h]["acc"] == m[-1]["held"][h]["acc"] for h in ("step1", "step2"))})')

    # 3. Neighbour labels and their base rates.
    check('all 9 one-op-slot neighbours of every held-out row are training rows (Latin-square property)', nb_all_train)
    errs = {(r['run_id'], r['measure']): r for r in csv.DictReader(io.StringIO(F['figures/heldout_error_types.csv'].decode()))}
    worst = 0.
    for (c, s), m in M.items():
        if (c, s) in (('full', 51200), ('interp', 51200)) or c.startswith('rnn'):
            continue
        pred = np.array(list(m[-1]['held_final_pred']), int); wrong = np.flatnonzero(pred != he['final'])
        for name, sets in [('neighbour in any op slot', [set().union(*n.values()) for n in NB])] + \
                [(f'neighbour differing in op slot {k}', [n[k] for n in NB]) for k in 'abc']:
            obs = np.mean([pred[i] in sets[i] for i in wrong]); exp = np.mean([len(sets[i] - {he['final'][i]})/7 for i in wrong])
            row = errs[rid(c, s), name]
            worst = max(worst, abs(obs - float(row['observed'])), abs(exp - float(row['expected_if_wrong_answers_uniform'])))
    check('heldout_error_types.csv (9 MLP runs x 4 measures): observed and uniform-expected rates equal the oracle recomputation',
          worst < 1e-12, f'max |diff| {worst:.1e}')
    allrows = np.mean([len(set().union(*n.values()) - {y})/7 for y, n in zip(he['final'], NB)])
    cslot = np.mean([len(n['c'] - {y})/7 for y, n in zip(he['final'], NB)])
    print(f'INFO over all 128 held-out rows: a uniform wrong answer is a neighbour label with probability {allrows:.3f} (any slot), {cslot:.3f} (slot c)')
    print('INFO distinct neighbour labels per held-out row (any slot):', np.bincount([len(set().union(*n.values())) for n in NB]).tolist())
    for c in ('final', 'inter'):
        for k in 'abc':
            print(f'INFO {c}: wrong answers that are slot-{k} neighbour labels', [round(float(errs[rid(c, s), f"neighbour differing in op slot {k}"]["observed"]), 3) for s in SEEDS])

    # 4. Counterexample 1: accuracy where the true answer is / is not some neighbour's label.
    print(f'INFO held-out rows whose true answer is a one-slot neighbour label (N): {N_ANY.sum()} of 128; not (notN): {(~N_ANY).sum()}')
    print('INFO accuracy at 10,000: all | N | notN   (a pure neighbour-copier scores 0 on notN)')
    for c in ('final', 'inter', 'shuf', 'rnn-final', 'rnn-inter'):
        for s in SEEDS:
            a, n, nn = split_acc(np.array(list(M[c, s][-1]['held_final_pred']), int))
            print(f'  {c:9s} {s}  {a:.3f} | {n:.3f} | {nn:.3f}')
    print('INFO rnn-inter during the rise, (step: all | N | notN):')
    for s in SEEDS:
        at = {r['step']: np.array(list(r['held_final_pred']), int) for r in M['rnn-inter', s]}
        print(f'  {s}', '; '.join(f'{k}: ' + ' | '.join(f'{x:.2f}' for x in split_acc(at[k])) for k in (1000, 2000, 4000, 7000, 10000)))
    for c in ('final', 'inter', 'shuf', 'rnn-inter'):
        for s in SEEDS:
            pred = np.array(list(M[c, s][-1]['held_final_pred']), int); w = np.flatnonzero(pred != he['final'])
            if c == 'rnn-inter' and len(w):
                print(f'INFO rnn-inter {s}: {len(w)} wrong rows, of which notN {int((~N_ANY[w]).sum())}, answered with a neighbour label '
                      f'{sum(int(pred[i]) in set().union(*NB[i].values()) for i in w)}')
    print('INFO wrong answers that are a neighbour label (any slot / slot c), 4 decimals:',
          {f'{c}-{s}': (round(float(errs[rid(c, s), 'neighbour in any op slot']['observed']), 4),
                        round(float(errs[rid(c, s), 'neighbour differing in op slot c']['observed']), 4)) for c, s in MLP})

    # Exposure across arms, and the paired MLP start.
    for (c, s) in MLP + RNN:
        e = json.loads(F[f'runs/{rid(c, s)}/config.json'])['exposure']
        if s == 51200:
            print(f'INFO exposure {c:9s}: fit rows {e["fit_rows"]}, updates {e["updates"]}, labels per row {e["labels_per_row"]}, parameters {e["total_parameters"]}')
    sys.path.insert(0, str(ROOT/'research/regimes/common')); import nets
    same = all(np.array_equal(nets.init(20, [128, 128], {'final': 8}, s)[k], nets.init(20, [128, 128], {h: 8 for h in ('step1', 'step2', 'final')}, s)[k])
               for s in SEEDS for k in ('W1', 'b1', 'W2', 'b2'))
    check('MLP trunk (W1, b1, W2, b2) at init is identical for final and inter/shuf heads at each seed', same)

    # 5. C1.1: what the held-out intermediate heads test.
    pre = {tuple(x) for x in np.stack([tr[k] for k in 'sab'], 1).tolist()}
    print('INFO held-out (s, a, b) prefixes that also occur in training rows:', sum(tuple(x) in pre for x in np.stack([he[k] for k in 'sab'], 1).tolist()), 'of 128')

    # 6. Probes and interchange.
    pr = list(csv.DictReader(io.StringIO(F['runs/r1-comp-probe-a1/probes.csv'].decode())))
    get = lambda c, step, lay, tgt: [round(float(r['probe_held_acc']), 3) for r in pr if r['condition'] == c and r['step'] == str(step)
                                     and r['layer'] == str(lay) and r['target'] == tgt]
    print('INFO probe held-out: input step1', [r['probe_held_acc'] for r in pr if r['layer'] == '0' and r['target'] == 'step1'],
          '| h1 step1 at init', sorted(set(get('final', 0, 1, 'step1') + get('inter', 0, 1, 'step1') + get('shuf', 0, 1, 'step1'))))
    for c in ('final', 'inter', 'shuf'):
        print(f'INFO probe held-out step2 {c}: h1 init {get(c, 0, 1, "step2")} h1 end {get(c, 10000, 1, "step2")} h2 end {get(c, 10000, 2, "step2")}')
    iv = list(csv.DictReader(io.StringIO(F['runs/r1-comp-probe-a1/interventions.csv'].decode())))
    trn = [r for r in iv if r['pairs_from'] == 'train']
    check(f'interchange on training pairs: the unpatched model is right on every base row ({len(trn)} result rows)',
          all(float(r['unpatched_base_acc']) == 1. for r in trn))
    print('INFO interchange on held-out pairs, unpatched base accuracy:', sorted({round(float(r['unpatched_base_acc']), 3) for r in iv if r['pairs_from'] == 'held'}))
    for c in ('final', 'inter'):
        for lay in ('1', '2'):
            sel = [r for r in trn if r['condition'] == c and r['layer'] == lay and r['variable'] == 'step2']
            f = lambda sub, k: [round(float(r[k]), 3) for r in sel if (r['subspace'] == 'probe') == sub and r['subspace'] != 'whole layer']
            print(f'INFO interchange step2 {c} h{lay}: follows cf probe {f(True, "follows_counterfactual")} random max {max(f(False, "follows_counterfactual"))}; '
                  f'keeps base probe {f(True, "keeps_base_answer")}; source answer probe {f(True, "gives_source_answer")} random range '
                  f'{min(f(False, "gives_source_answer"))}-{max(f(False, "gives_source_answer"))}')

    # 7. Follow-up order: pre-registration and plan commits before the RNN launches.
    when = lambda c: subprocess.check_output(['git', '-C', str(ROOT), 'show', '-s', '--format=%cI', c], text=True).strip()
    reg = [json.loads(l) for l in F['registry.jsonl'].decode().splitlines()]
    launch = min(r['time'] for r in reg if r['status'] == 'running' and '-rnn-' in r['run_id'] and 'calib' not in r['run_id'])
    calib = min(r['time'] for r in reg if r['status'] == 'running' and 'calib-rnn' in r['run_id'])
    check(f'follow-up order: predictions b842eb2 ({when("b842eb2")}) < RNN code 3a58b61 ({when("3a58b61")}) <= calibration launch ({calib}); '
          f'plan 4497ce1 ({when("4497ce1")}) < first follow-up launch ({launch})',
          when('b842eb2') < when('3a58b61') <= calib and when('4497ce1') < launch)

    # 8. Counterexample 2: the reviewer's untied-readout arm, paired with the lane's rnn-inter.
    mine = lambda r: [json.loads(l) for l in (root/r/'metrics.jsonl').read_text().splitlines()]
    check('replay: tied mode reproduces the lane r1-comp-rnn-inter-s51200-a1 metrics.jsonl line for line',
          (root/'r1-rev-comp-rnn-replay-inter-s51200-a1/metrics.jsonl').read_text() == F['runs/r1-comp-rnn-inter-s51200-a1/metrics.jsonl'].decode())
    print('INFO untied vs tied (lane) rnn-inter, same init: held-out final at 1k/4k/7k/10k; train fit; all | N | notN at 10k')
    for s in SEEDS:
        for name, m in ((f'untied {s}', mine(f'r1-rev-comp-rnn-untied-s{s}-a1')), (f'lane   {s}', M['rnn-inter', s])):
            at = {r['step']: r['held'] for r in m}
            a, n, nn = split_acc(np.array(list(m[-1]['held_final_pred']), int))
            print(f'  {name}: {[round(at[k]["final"]["acc"], 3) for k in (1000, 4000, 7000, 10000)]}; notN {[round(float(split_acc(np.array(list(r["held_final_pred"]), int))[2]), 2) for r in m if r["step"] in (1000, 4000, 7000, 10000)]}; '
                  f'fit at {first(m, lambda r: r["train"]["final"]["acc"] == 1.)}; held step1/step2 {at[10000]["step1"]["acc"]:.3f}/{at[10000]["step2"]["acc"]:.3f}; '
                  f'{a:.3f} | {n:.3f} | {nn:.3f}')
        z = np.load(root/f'r1-rev-comp-rnn-untied-s{s}-a1/ckpt-10000.npz')
        allr = {k: np.concatenate([tr[k], he[k]]) for k in tr}; H = rnn_states(z, allr)
        cross = {f'V{v} on h{t}': float(np.mean((H[t] @ z[f'p.V{v}'] + z[f'p.bv{v}']).argmax(1) == allr[('step1', 'step2', 'final')[t-1]]))
                 for t in (1, 2, 3) for v in (1, 2, 3)}
        print(f'  untied {s}: readout V_v applied to h_t, accuracy for step t on all 512 rows: '
              + ', '.join(f'{k} {v:.2f}' for k, v in cross.items()))


if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]))
    print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
