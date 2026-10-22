#!/usr/bin/env python3
"""Frozen evaluation of tied/tied c2 RNN checkpoints on programs of lengths 1..6 (QUESTION.md
Part A, and Part B's length curves). No fitting: every number is a forward pass of a saved model.

Rows are exhaustive at every length (all 4^L programs x 8 starts, weight 1 each, so row, program
and start-symbol weighting coincide). For each checkpoint it writes
  length.csv   accuracy by length and subset (QUESTION.md measures 1-2)
  steps.csv    per-step accuracy, first-error step, behaviour after a first error and
               cross-position drift on the length-6 rows (measures 3-4)
and, with --references, the reference systems (oracle loop, r1's P2 iterated, the network's own
one-step map re-encoded) in references.csv. checks.json records hashes, the length-3 comparison
with the run's own logged metrics, the causality identity and the oracle agreement.

uv run --locked python research/composition-c2/generalization/gen_eval.py --out DIR --recovered --references
uv run --locked python research/composition-c2/generalization/gen_eval.py --out DIR --run-dirs RUN_DIR ... --step 10000
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'common'), str(HERE.parents[1]/'regimes'/'common')]
import c2rnn  # noqa: E402
import c2task  # noqa: E402
import gen_task  # noqa: E402
import nets  # noqa: E402

C2 = HERE.parent
R1_RUNS = C2.parents[0]/'regimes'/'composition'/'runs'
MAX_LEN = 6
P2_RUN = 'r1-comp-interp-s51200-a1'
P2_SHA = 'bedf8ed3b8532f51bc9ed4354431bd5223031d501894416432dc0c0eb9e7defd'  # r1 MANIFEST.json, RECOVERY.json


def fwd(p, ops, s):
    """c2rnn.forward for a tied/tied model, any length. c2rnn's default step map is 1..L, which
    fails its own range check past L = 3 even when both groups are tied (pin note in README); tied
    groups ignore the map, so the explicit map [1, 2, 3, 3, ...] that test_c2 exercises is passed."""
    assert c2rnn.scheme(p) == dict(transition=True, readout=True), 'tied/tied only'
    L = np.asarray(ops).reshape(len(ops), -1).shape[1]
    return c2rnn.forward(p, ops, s=s, step_map=[min(t, c2task.TRAIN_LENGTH) for t in range(1, L+1)])


def recovered_entries(step=10000):
    """The six tied/tied r1 runs (inter, final) from recovered/, hashes from RECOVERY.json."""
    rec = json.loads((C2/'recovered'/'RECOVERY.json').read_text())['files']
    out = []
    for f in rec:
        cond = f['run_id'].split('-')[3] if f['run_id'].startswith('r1-comp-rnn-') else None
        if cond in ('inter', 'final') and f['file'] == f'ckpt-{step}.npz':
            r1cfg = json.loads((R1_RUNS/f['run_id']/'config.json').read_text())
            out.append(dict(run_id=f['run_id'], path=C2/'recovered'/f['run_id']/f['file'], sha256=f['sha256'],
                            condition=cond, init_seed=r1cfg['init_seed'], task_id=c2task.ORIGINAL,
                            metrics=R1_RUNS/f['run_id']/'metrics.jsonl'))
    return sorted(out, key=lambda e: (e['condition'], e['init_seed']))


def run_dir_entries(dirs, step):
    """c2train.py run directories (config.json, metrics.jsonl, ckpt-STEP.npz); hash computed here."""
    out = []
    for d in map(Path, dirs):
        cfg = json.loads((d/'config.json').read_text())
        w = cfg['step_weights']
        cond = 'inter' if w == [1., 1., 1.] else 'final' if w == [0., 0., 1.] else 'w' + '-'.join(f'{x:g}' for x in w)
        assert cfg['scheme'] == dict(transition=True, readout=True), f'{d}: only tied/tied is evaluated here'
        path = d/f'ckpt-{step}.npz'
        out.append(dict(run_id=cfg['run_id'] or d.name, path=path, sha256=nets.sha256_file(path), condition=cond,
                        init_seed=cfg['init_seed'], task_id=cfg['task']['id'], metrics=d/'metrics.jsonl'))
    return out


def load_checked(path, sha):
    got = nets.sha256_file(path)
    if got != sha:
        raise SystemExit(f'{path}: SHA-256 {got} does not equal {sha}')
    p, st, meta = nets.load(path)
    return p, st['t'], meta


def logged_row(metrics, step):
    for line in Path(metrics).read_text().splitlines():
        r = json.loads(line)
        if r['step'] == step:
            return r
    return None


def program_index(progs):
    return {tuple(q): i for i, q in enumerate(progs)}


def evaluate(p, T):
    """(length rows, step rows, checks) for one tied/tied model on task T."""
    P, train, held = T['P'], set(map(tuple, T['train'])), T['held']
    nbr = c2task.neighbour_answerable(P, T['train'], held)[0]
    out, checks, final_acc = [], {}, {}
    add = lambda L, subset, ok, n_prog: out.append(dict(length=L, subset=subset, n_rows=int(ok.size), n_programs=n_prog,
                                                        accuracy=float(ok.mean()) if ok.size else float('nan')))
    for L in range(1, MAX_LEN+1):
        progs = c2task.programs(L)
        d = c2task.rows(P, progs)
        ok = fwd(p, d['ops'], s=d['s'])[1][-1].argmax(1) == d['states'][:, -1]
        final_acc[L] = float(ok.mean())
        add(L, 'all', ok, len(progs))
        okp = ok.reshape(len(progs), c2task.N_SYM)  # rows are program-major, starts 0..7 within
        if L == 3:
            is_tr = np.array([q in train for q in progs])
            add(3, 'train programs', okp[is_tr].ravel(), int(is_tr.sum()))
            add(3, 'held-out programs', okp[~is_tr].ravel(), int((~is_tr).sum()))
            dh = c2task.rows(P, held)
            okh = fwd(p, dh['ops'], s=dh['s'])[1][-1].argmax(1)
            checks['held_final_pred'] = ''.join(map(str, okh))
            okh = okh == dh['states'][:, -1]
            add(3, 'held-out N', okh[nbr], len(held)); add(3, 'held-out notN', okh[~nbr], len(held))
        if L >= 4:
            pre = np.array([q[:3] in train for q in progs])
            add(L, 'length-3 prefix is a training program', okp[pre].ravel(), int(pre.sum()))
            add(L, 'length-3 prefix is a held-out program', okp[~pre].ravel(), int((~pre).sum()))
            cls = np.array(gen_task.classify(P, T['train'], held, L))
            for c in gen_task.CLASSES:
                add(L, f'function class {c}', okp[cls == c].ravel(), int((cls == c).sum()))
            # Per-function weighting: mean over programs sharing a product, then over products.
            tables = gen_task.products(P, L)[1]
            _, inv = np.unique(tables, axis=0, return_inverse=True)
            per_prog = okp.mean(1)
            per_fn = np.bincount(inv.ravel(), per_prog) / np.bincount(inv.ravel())
            out.append(dict(length=L, subset='all, per-function weighting', n_rows=int(ok.size), n_programs=len(progs),
                            n_functions=len(per_fn), accuracy=float(per_fn.mean())))
    steps = step_analysis(p, T, final_acc, checks)
    return out, steps, checks


def step_analysis(p, T, final_acc, checks):
    """Per-step measures on all length-6 rows (QUESTION.md measures 3-4)."""
    P = T['P']
    d = c2task.rows(P, c2task.programs(MAX_LEN))
    H, logits = fwd(p, d['ops'], s=d['s'])
    pred = np.stack([z.argmax(1) for z in logits], 1)  # (n, 6)
    ok = pred == d['states']
    acc = ok.mean(0)
    checks['causality_identity_max_abs_diff'] = float(max(abs(acc[t-1] - final_acc[t]) for t in range(1, MAX_LEN+1)))
    assert checks['causality_identity_max_abs_diff'] == 0, 'step-t accuracy on length-6 rows differs from length-t accuracy'
    first = np.where(ok.all(1), 0, (~ok).argmax(1) + 1)  # 0 = never wrong
    # Drift: centroids of training-row h3 by true final symbol.
    tr = c2task.rows(P, T['train'])
    h3 = fwd(p, tr['ops'], s=tr['s'])[0][3]
    cent = np.stack([h3[tr['states'][:, -1] == y].mean(0) for y in range(c2task.N_SYM)])
    ref_norm = np.median(np.linalg.norm(h3, axis=1))
    ref_dist = np.median(np.linalg.norm(h3 - cent[tr['states'][:, -1]], axis=1))
    sym = np.concatenate([d['s'][:, None], d['states']], 1)
    rows = []
    for t in range(0, MAX_LEN+1):
        r = dict(step=t, n_rows=len(d['s']),
                 norm_ratio=float(np.median(np.linalg.norm(H[t], axis=1)) / ref_norm),
                 centroid_distance_ratio=float(np.median(np.linalg.norm(H[t] - cent[sym[:, t]], axis=1)) / ref_dist))
        if t >= 1:
            r.update(accuracy=float(acc[t-1]), first_error_here=int((first == t).sum()))
            if t < MAX_LEN:  # behaviour at step t+1 of the rows whose first error is at step t
                sel = first == t
                follow = P[d['ops'][sel, t], pred[sel, t-1]]
                nxt = pred[sel, t]
                r.update(after_first_error_n=int(sel.sum()), after_recovered=int((nxt == d['states'][sel, t]).sum()),
                         after_follows_own_error=int((nxt == follow).sum()),
                         after_other=int(((nxt != d['states'][sel, t]) & (nxt != follow)).sum()))
        rows.append(r)
    rows.append(dict(step='never wrong in 6 steps', n_rows=len(d['s']), first_error_here=int((first == 0).sum())))
    return rows


def references(T, entries):
    """Oracle loop, r1's P2 iterated with argmax feedback, and each network's own one-step map."""
    P = T['P']
    out, checks = [], {}
    p2, _, _ = load_checked(C2/'recovered'/P2_RUN/'ckpt-10000.npz', P2_SHA)
    one_step = {}
    for e in entries:
        p, _, _ = load_checked(e['path'], e['sha256'])
        s, o = np.repeat(np.arange(c2task.N_SYM), c2task.N_OP), np.tile(np.arange(c2task.N_OP), c2task.N_SYM)
        F = np.zeros((c2task.N_OP, c2task.N_SYM), int)
        F[o, s] = fwd(p, o[:, None], s=s)[1][0].argmax(1)
        one_step[e['run_id']] = F
    for L in range(1, MAX_LEN+1):
        progs = c2task.programs(L)
        d = c2task.rows(P, progs)
        y = d['states'][:, -1]
        loop = np.array([c2task.r1.execute(P, s, q)[-1] for q in progs for s in range(c2task.N_SYM)])
        out.append(dict(system='oracle (per-row loop)', length=L, accuracy=float(np.mean(loop == y))))
        x = d['s'].copy()
        for t in range(L):
            X = np.zeros((len(x), c2task.N_SYM + c2task.N_OP))
            X[np.arange(len(x)), x] = 1; X[np.arange(len(x)), c2task.N_SYM + d['ops'][:, t]] = 1
            x = nets.forward(p2, X)[1]['next'].argmax(1)
        out.append(dict(system=f'P2 iterated ({P2_RUN})', length=L, accuracy=float(np.mean(x == y))))
        for rid, F in one_step.items():
            x = d['s'].copy()
            for t in range(L):
                x = F[d['ops'][:, t], x]
            out.append(dict(system=f'own one-step map re-encoded ({rid})', length=L, accuracy=float(np.mean(x == y))))
    checks['p2_sha256'] = P2_SHA
    checks['own_one_step_transitions_correct'] = {rid: int((F == P).sum()) for rid, F in one_step.items()}
    return out, checks


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--recovered', action='store_true', help="r1's tied/inter and tied/final ckpt-10000 (recovered/)")
    ap.add_argument('--run-dirs', nargs='*', default=[], help='c2train.py run directories (tied/tied)')
    ap.add_argument('--step', type=int, default=10000, help='checkpoint step for --run-dirs')
    ap.add_argument('--references', action='store_true', help='reference systems (original task only)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    entries = (recovered_entries() if args.recovered else []) + run_dir_entries(args.run_dirs, args.step)
    length_rows, step_rows, checks, tasks = [], [], {}, {}
    for e in entries:
        T = tasks.setdefault(e['task_id'], c2task.load(e['task_id']))
        p, t, meta = load_checked(e['path'], e['sha256'])
        assert c2rnn.scheme(p) == dict(transition=True, readout=True)
        lr, sr, ck = evaluate(p, T)
        tag = dict(run_id=e['run_id'], ckpt_step=t, condition=e['condition'], init_seed=e['init_seed'], task_id=e['task_id'])
        length_rows += [dict(tag, **r) for r in lr]; step_rows += [dict(tag, **r) for r in sr]
        logged = logged_row(e['metrics'], t)
        acc3 = {r['subset']: r['accuracy'] for r in lr if r['length'] == 3}
        acc12 = {r['length']: r['accuracy'] for r in lr if r['length'] in (1, 2) and r['subset'] == 'all'}
        ck.update(path=str(e['path']), sha256=e['sha256'], step=t,
                  logged_row_found=logged is not None,
                  held_final_pred_matches_log=logged is not None and logged['held_final_pred'] == ck['held_final_pred'],
                  train_final_matches_log=logged is not None and logged['train']['final']['acc'] == acc3['train programs'],
                  held_final_matches_log=logged is not None and logged['held']['final']['acc'] == acc3['held-out programs'],
                  length1_equals_logged_train_step1=logged is not None and logged['train']['step1']['acc'] == acc12[1],
                  length2_equals_logged_train_step2=logged is not None and logged['train']['step2']['acc'] == acc12[2])
        checks[e['run_id']] = ck
    if args.references:
        assert set(tasks) == {c2task.ORIGINAL}, 'reference systems exist for the original task only'
        ref_rows, ref_checks = references(tasks[c2task.ORIGINAL], entries)
        write_csv(args.out/'references.csv', ref_rows); checks['references'] = ref_checks
    write_csv(args.out/'length.csv', length_rows); write_csv(args.out/'steps.csv', step_rows)
    (args.out/'checks.json').write_text(json.dumps(checks, indent=1, default=int))


if __name__ == '__main__':
    main()
