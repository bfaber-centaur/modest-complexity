#!/usr/bin/env python3
"""Third e3 audit investigation (research/e3/audit/QUESTION-3.md): the nine mechanisms
continuations of packet 1, from their published files.

K1 provenance, K2 own forward pass on every published checkpoint, K3 an independent dense
replay of updates 1-300 (common-set accuracy at every update), K4 an independent replay of
updates 1,000-2,000 for the filtered runs, and the per-checkpoint comparisons with arm A
(argmax agreement, weight distance, pairwise preferences) for questions 3-4. Since investigation 4
(QUESTION-4.md, W1-W3) it also covers the six weight-decay-0 continuations of the same B and C
start files.

uv run --locked python research/e3/audit/mech_check.py OUT_DIR
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(root/'research/audit'))
import independent as ind  # noqa: E402
from pairwise import ladder_sets, preference  # noqa: E402

P = 31
ME = root/'research/e3/mechanisms'
SOURCES = (30000, 30001, 30002)
ARMS = ('keep', 'filt', 'sham', 'filt-wd0', 'sham-wd0')
WD_OF = lambda arm: 0. if arm.endswith('-wd0') else 1.
START = lambda arm, s: ME/'starts'/f"{RID(arm.removesuffix('-wd0'), s)}.npz"  # wd-0 runs reuse the gated start files
RID = lambda arm, s: f'e3-me-{arm}-w128-i{s}-a1'
plan = json.loads((root/'research/e3/plan.json').read_text())
ENTRY = {e['run_id']: e for e in plan['runs'] + plan.get('reserved', []) if e['run_id'].startswith('e3-me-')}
a, b, y, _, (tr, cal, aud) = ind.regenerate_split(P, 31000, 32000, .3)
tr50 = ind.regenerate_split(P, 31000, 32000, .5)[4][0]
COMMON = np.load(root/'research/e2/data/common-never-trained-30-50.npz')['index']
S, D, N, Y = ladder_sets(a, b, tr, tr50, COMMON, P)
C = np.eye(P, dtype=bool)[Y]
XA, XB = ind.one_hot_inputs(a, b, P)
LR = plan['common']['lr']


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def logits(w, ix=slice(None)):
    return ind.forward(w, XA[ix], XB[ix])[0]


def acc(z, ix):
    return float(np.mean(z.argmax(1) == y[ix]))


def checkpoints(run):
    return {int(f.stem.split('_')[1]): f for f in sorted((ME/'runs'/run).glob('checkpoint_*.npz'))}


def load(f, moments=False):
    with np.load(f) as z:
        w = [z[k] for k in 'UVW']
        return (w, [z[k] for k in ('mU', 'mV', 'mW')], [z[k] for k in ('vU', 'vV', 'vW')]) if moments else w


def k1():
    events = [json.loads(l) for l in (ME/'registry.jsonl').read_text().splitlines() if l.strip()]
    out = {}
    for arm in ARMS:
        for s in SOURCES:
            run = RID(arm, s)
            d = ME/'runs'/run
            ev = [e['status'] for e in events if e['run_id'] == run]
            cfg = json.loads((d/'config.json').read_text())
            man = json.loads((d/'manifest.json').read_text())
            start = START(arm, s)
            w0, m0, v0 = load(d/'checkpoint_000000.npz', True)
            files_ok = all(sha(d/n) == h['sha256'] for n, h in man['files'].items() if h.get('published') and (d/n).exists())
            missing = [n for n, h in man['files'].items() if h.get('published') and not (d/n).exists()]
            out[run] = dict(
                events=ev, completed=json.loads((d/'runtime.json').read_text())['completed_steps'],
                start_sha_recorded=cfg['init_weights_sha256'] == sha(start) == ENTRY[run]['init_weights_sha256'],
                recipe=(cfg['init_age'], cfg['steps'], cfg['lr'], cfg['wd'], cfg['log_every'], cfg['data_hashes']['split_sha256'][:12]),
                step0_is_start=all(np.array_equal(x, z) for x, z in zip(w0, load(start))),
                step0_moments_zero=not any(np.any(x) for x in m0 + v0),
                manifest_published_files_match=files_ok, manifest_missing=missing,
                producing_code_commit=man['producing_code_commit'], producing_tree_dirty=man['producing_tree_dirty'])
            out[run]['PASS'] = bool(ev == ['running', 'complete'] and out[run]['completed'] == 20000 and out[run]['start_sha_recorded']
                                    and out[run]['recipe'][:5] == (50000, 20000, .001, WD_OF(arm), 10) and out[run]['step0_is_start']
                                    and out[run]['step0_moments_zero'] and files_ok and not missing)
    return out


def k2_and_measures():
    """Own forward on every published checkpoint: logged-metric reproduction, and the
    per-checkpoint quantities used for questions 3-4."""
    k2, rows, W = {}, [], {}
    for arm in ARMS:
        for s in SOURCES:
            run = RID(arm, s)
            logged = {r['step']: r for r in map(json.loads, (ME/'runs'/run/'metrics.jsonl').read_text().splitlines())}
            worst_ce = 0.
            exact = True
            for step, f in checkpoints(run).items():
                w = W[(arm, s, step)] = load(f)
                z = logits(w)
                for name, ix in (('train', tr), ('cal', cal), ('audit', aud)):
                    exact &= acc(z[ix], ix) == logged[step][name]
                    ce = float(np.mean(np.logaddexp.reduce(z[ix], axis=1) - z[ix, y[ix]]))
                    lg = logged[step]['objective'][name]['cross_entropy']
                    worst_ce = max(worst_ce, abs(ce - lg)/(1e-10*abs(lg) + 1e-14))
                zc = z[COMMON]
                rows.append(dict(run_id=run, arm=arm, source=s, step=step, train_acc=acc(z[tr], tr), common_acc=acc(zc, COMMON),
                                 own_heldout_acc=acc(z[np.r_[cal, aud]], np.r_[cal, aud]),
                                 P_SN=float(np.mean(preference(zc, S, N)[0])), P_cN=float(np.mean(preference(zc, C, N)[0])),
                                 P_cS=float(np.mean(preference(zc, C, S)[0])), argmax=zc.argmax(1)))
            k2[run] = dict(checkpoints=len(checkpoints(run)), accuracies_exact=bool(exact),
                           worst_ce_diff_in_tolerance_units=worst_ce, PASS=bool(exact and worst_ce <= 1))
    keyed = {(r['arm'], r['source'], r['step']): r for r in rows}
    for r in rows:  # against A at the same update, where A has a checkpoint (B's first-fit checkpoints have no A partner)
        ref = keyed.get(('keep', r['source'], r['step']))
        wa, w = W.get(('keep', r['source'], r['step'])), W[(r['arm'], r['source'], r['step'])]
        r['agree_with_A'] = float(np.mean(r['argmax'] == ref['argmax'])) if ref else ''
        r['distance_from_A'] = float(np.sqrt(sum(np.sum((x - q)**2) for x, q in zip(w, wa))/sum(np.sum(q**2) for q in wa))) if ref else ''
    cross = []
    steps = sorted(set.intersection(*({r['step'] for r in rows if r['run_id'] == run} for run in {r['run_id'] for r in rows})))
    for step in steps:
        A = [keyed[('keep', s, step)]['argmax'] for s in SOURCES]
        cross.append(dict(step=step, A_A_agreement=[float(np.mean(A[i] == A[j])) for i, j in ((0, 1), (0, 2), (1, 2))],
                          B_C_agreement=[float(np.mean(keyed[('filt', s, step)]['argmax'] == keyed[('sham', s, step)]['argmax']))
                                         for s in SOURCES]))
    for r in rows:
        del r['argmax']
    return k2, rows, cross


def k3():
    """Independent replay of updates 1-300 from each start file with zero moments."""
    out, dense = {}, []
    for arm in ARMS:
        for s in SOURCES:
            run = RID(arm, s)
            ck = checkpoints(run)
            w = load(START(arm, s))
            scale = max(float(np.abs(x).max()) for x in w)
            m, v = [np.zeros_like(x) for x in w], [np.zeros_like(x) for x in w]
            diffs = {}
            for t in range(0, 301):
                if t:
                    ind.adamw_step(w, ind.loss_and_grad(w, XA[tr], XB[tr], y[tr])[1], m, v, t, LR, WD_OF(arm))
                z = logits(w)
                dense.append(dict(run_id=run, arm=arm, source=s, update=t, common_acc=acc(z[COMMON], COMMON),
                                  train_acc=acc(z[tr], tr)))
                if t in ck and t:
                    diffs[t] = max(float(np.abs(x - q).max()) for x, q in zip(w, load(ck[t])))
            out[run] = dict(max_abs_diff_at_checkpoints=diffs, tolerance=1e-9*scale, PASS=max(diffs.values()) <= 1e-9*scale)
    return out, dense


def k4():
    out = {}
    for s in SOURCES:
        run = RID('filt', s)
        ck = checkpoints(run)
        w, m, v = load(ck[1000], True)
        for t in range(1001, 2001):
            ind.adamw_step(w, ind.loss_and_grad(w, XA[tr], XB[tr], y[tr])[1], m, v, t, LR, 1.)
        w2 = load(ck[2000])
        a_rep, a_pub = acc(logits(w)[COMMON], COMMON), acc(logits(w2)[COMMON], COMMON)
        out[run] = dict(max_abs_weight_diff=max(float(np.abs(x - q).max()) for x, q in zip(w, w2)),
                        common_acc_replay=a_rep, common_acc_published=a_pub, PASS=abs(a_rep - a_pub) <= 1/483 + 1e-12)
    return out


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader()
        wr.writerows({k: (float(f'{v:.10g}') if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def main(out_dir):
    out = Path(out_dir)
    res = dict(K1=k1())
    res['K2'], rows, cross = k2_and_measures()
    res['K3'], dense = k3()
    res['K4'] = k4()
    write_csv(out/'mech-checkpoints.csv', rows)
    write_csv(out/'mech-dense.csv', dense)
    res['cross_source'] = cross
    (out/'mech-check.json').write_text(json.dumps(res, indent=1) + '\n')
    print({k: all(v['PASS'] for v in res[k].values()) for k in ('K1', 'K2', 'K3', 'K4')})


if __name__ == '__main__':
    main(sys.argv[1])
