#!/usr/bin/env python3
"""Dynamics checks (research/e3/audit/QUESTION-2.md, Y1-Y2) from the split rule alone.

Rebuilds every e3 ladder split with the e1 audit's independent rewrite of the split rule
(research/audit/independent.py, which imports nothing from experiment.py) and checks it
against the plan, the coordinator's ladders.json and the saved evaluation sets: hashes and
actual training counts, nesting, transpose closure, partition, the evaluation pairs as the
complement of the 50% training set, the label-set sizes, and the crossed design.

The `runs` mode checks the landed runs (Y2-Y5): step-0 weights against the independent init
rule, registry and manifest accounting, my own forward pass on every published checkpoint,
and independent replays of the first 2,000 updates of one run and of its last 1,000.

uv run --locked python research/e3/audit/dynamics_check.py splits OUT.json
uv run --locked python research/e3/audit/dynamics_check.py runs RUNS_DIR REGISTRY OUT.json [REPLAY_RUN_ID]
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(root/'research/audit'))
import independent as ind  # noqa: E402
sys.path.insert(0, str(root))
from experiment import init as exp_init  # noqa: E402  (compared with the independent rule only)
from pairwise import ladder_sets  # noqa: E402

P, CONTROL = 31, 32000
SHARES = (.3, .4, .45, .5)
SEEDS = (31000, 41001, 41002, 41003)
INITS = (30000, 30001)
plan = json.loads((root/'research/e3/plan.json').read_text())
ladders = {r['data_seed']: r for r in json.loads((root/'research/e3/coordinator/ladders.json').read_text())}
REFERENCE = {.3: ['research/coordinator/runs/e1-co-add-w128-i30000-a1', 'research/size-time/runs/e1-st-add-w128-i30001-a1'],
             .4: [f'research/e2/dynamics/runs/e2-dy-add-w128-i{i}-tf40-a1' for i in INITS],
             .45: [f'research/e2/dynamics/runs/e2-dy-add-w128-i{i}-tf45-a1' for i in INITS],
             .5: [f'research/coordinator/runs/e1f2-co-add-w128-i{i}-tf50-a1' for i in INITS]}


def transpose_closed(a, b, ix):
    return set((a[ix]*P + b[ix]).tolist()) == set((b[ix]*P + a[ix]).tolist())


def splits(out_json):
    entries = [e for e in plan['runs'] if e['run_id'].startswith('e3-dy-')]
    by_key = {(e['data_seed'], e['init_seed'], e['train_fraction']): e for e in entries}
    res = dict(design=dict(
        entries=len(entries), distinct=len(by_key),
        crossed=sorted(by_key) == sorted(itertools.product(SEEDS[1:], INITS, SHARES)),
        ids_match_fields=all(e['run_id'] == f"e3-dy-add-w128-i{e['init_seed']}-tf{round(100*e['train_fraction'])}-d{e['data_seed']}-a1"
                             for e in entries),
        recipe=sorted({json.dumps({k: {**plan['common'], **e}[k] for k in ('task', 'wd', 'width', 'steps', 'lr', 'log_every',
                                                                         'checkpoint_every', 'diagnostics', 'control_seed')})
                       for e in entries}),
        queue_positions_are_1_to_24=sorted(e['queue_position'] for e in entries) == list(range(1, 25)),
        init_weights_independent_of_data_seed='init(p, hidden, seed) takes no data seed (experiment.py); '
                                              'checked on landed step-0 checkpoints in Y2'), ladders={})
    for seed in SEEDS:
        lad, prev = dict(shares={}), None
        for share in SHARES:
            a, b, y, _, sp = ind.regenerate_split(P, seed, CONTROL, share)
            h = ind.hash_arrays(a, b, y, *sp)
            claims = [by_key[(seed, i, share)] for i in INITS] if seed != 31000 else []
            tr = set(sp[0].tolist())
            lad['shares'][str(share)] = dict(
                train=len(sp[0]), split_sha256=h, matches_ladders_json=h == ladders[seed]['shares'][str(share)]['split_sha256'],
                matches_plan=all(e['split_sha256'] == h and e['train_pairs'] == len(sp[0]) for e in claims) if claims else None,
                partition=sorted(np.concatenate(sp).tolist()) == list(range(P*P)),
                transpose_closed=all(transpose_closed(a, b, ix) for ix in sp),
                nested_in_this=prev is None or prev <= tr)
            prev = tr
        ev = np.array(sorted(set(range(P*P)) - prev))
        saved = np.load(root/f'research/e3/data/eval-d{seed}.npz')['index']
        S, D, N, yy = ladder_sets(a, b, ind.regenerate_split(P, seed, CONTROL, .3)[4][0], np.array(sorted(prev)), ev, P)
        lad['evaluation'] = dict(
            pairs=len(ev), equals_saved=np.array_equal(ev, saved), sha256=ind.hash_arrays(ev),
            matches_ladders_json=ind.hash_arrays(ev) == ladders[seed]['evaluation']['sha256'],
            transpose_closed=transpose_closed(a, b, ev),
            unordered_groups=len({(min(i, j), max(i, j)) for i, j in zip(a[ev], b[ev])}),
            per_class=[int(np.bincount(yy, minlength=P).min()), int(np.bincount(yy, minlength=P).max())],
            S=[int(S.sum(1).min()), round(float(S.sum(1).mean()), 2), int(S.sum(1).max())],
            D=[int(D.sum(1).min()), int(D.sum(1).max()), int((D.sum(1) == 0).sum())],
            N=[int(N.sum(1).min()), int(N.sum(1).max()), int((N.sum(1) == 0).sum()), int((N.sum(1) == 1).sum())])
        lad['evaluation']['N_matches_ladders_json'] = (
            lad['evaluation']['N'][0] == ladders[seed]['evaluation']['N_size']['min']
            and lad['evaluation']['N'][1] == ladders[seed]['evaluation']['N_size']['max']
            and lad['evaluation']['N'][3] == ladders[seed]['evaluation']['N_size']['singleton'])
        res['ladders'][str(seed)] = lad
    ref = {}
    for share, runs in REFERENCE.items():
        h = res['ladders']['31000']['shares'][str(share)]['split_sha256']
        for r in runs:
            c = json.loads((root/r/'config.json').read_text())
            ref[Path(r).name] = c['data_hashes']['split_sha256'] == h and c['seeds']['data_seed'] == 31000 \
                and c['seeds']['init_seed'] == int(Path(r).name.split('-i')[1][:5]) and c['hidden'] == 128 and c['wd'] == 1
    res['reference_ladder_31000_runs_match'] = ref
    res['init_rule_matches_independent'] = {i: all(np.array_equal(p, q) for p, q in zip(exp_init(P, 128, i), ind.initial_weights(P, 128, i)))
                                            for i in INITS}
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    lad_ok = all(all(s['matches_ladders_json'] and s['partition'] and s['transpose_closed'] and s['nested_in_this']
                     and s['matches_plan'] in (True, None) for s in l['shares'].values())
                 and l['evaluation']['equals_saved'] and l['evaluation']['matches_ladders_json'] and l['evaluation']['transpose_closed']
                 and l['evaluation']['N_matches_ladders_json'] for l in res['ladders'].values())
    d = res['design']
    print(dict(Y1=lad_ok, design=d['entries'] == d['distinct'] == 24 and d['crossed'] and d['ids_match_fields']
               and len(d['recipe']) == 1 and d['queue_positions_are_1_to_24'],
               reference=all(ref.values()), init=all(res['init_rule_matches_independent'].values())))


def runs(runs_dir, registry, out_json, replay_id=None):
    runs_dir = Path(runs_dir)
    entries = {e['run_id']: e for e in plan['runs'] if e['run_id'].startswith('e3-dy-')}
    events = [json.loads(l) for l in Path(registry).read_text().splitlines() if l.strip()]
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    init = {i: ind.initial_weights(P, 128, i) for i in INITS}
    res, splits_cache = dict(runs={}), {}
    for rid, e in sorted(entries.items(), key=lambda kv: kv[1]['queue_position']):
        ev = [x['status'] for x in events if x['run_id'] == rid]
        d = runs_dir/rid
        r = res['runs'][rid] = dict(events=ev, published=d.exists())
        if not d.exists():
            continue
        cfg = json.loads((d/'config.json').read_text())
        man = json.loads((d/'manifest.json').read_text())
        rt = json.loads((d/'runtime.json').read_text())
        key = (e['data_seed'], e['train_fraction'])
        if key not in splits_cache:
            splits_cache[key] = ind.regenerate_split(P, *key[:1], CONTROL, key[1])
        a, b, y, _, sp = splits_cache[key]
        XA, XB = ind.one_hot_inputs(a, b, P)
        logged = {x['step']: x for x in map(json.loads, (d/'metrics.jsonl').read_text().splitlines())}
        cks = {int(f.stem.split('_')[1]): f for f in sorted(d.glob('checkpoint_*.npz'))}
        exact, worst = True, 0.
        for step, f in cks.items():
            with np.load(f) as z:
                w = [z[k] for k in 'UVW']
            zl = ind.forward(w, XA, XB)[0]
            for name, ix in zip(('train', 'cal', 'audit'), sp):
                exact &= float(np.mean(zl[ix].argmax(1) == y[ix])) == logged[step][name]
                ce = float(np.mean(np.logaddexp.reduce(zl[ix], axis=1) - zl[ix, y[ix]]))
                lg = logged[step]['objective'][name]['cross_entropy']
                worst = max(worst, abs(ce - lg)/(1e-10*abs(lg) + 1e-14))
        step0 = None
        if 0 in cks:
            with np.load(cks[0]) as z:
                step0 = all(np.array_equal(z[k], x) for k, x in zip('UVW', init[e['init_seed']]))
        r.update(completed=rt['completed_steps'], status=rt['status'],
                 config_matches_plan=(cfg['data_hashes']['split_sha256'] == e['split_sha256'] and cfg['seeds']['data_seed'] == e['data_seed']
                                      and cfg['seeds']['init_seed'] == e['init_seed'] and cfg['train_fraction'] == e['train_fraction']
                                      and cfg['steps'] == 50000 and cfg['wd'] == 1 and cfg['hidden'] == 128 and cfg['lr'] == .001),
                 train_pairs=len(sp[0]) == e['train_pairs'],
                 manifest_ok=all(sha(d/n) == h['sha256'] for n, h in man['files'].items() if h.get('published') and (d/n).exists()),
                 manifest_missing=[n for n, h in man['files'].items() if h.get('published') and not (d/n).exists()],
                 checkpoints=sorted(cks), accuracies_exact=bool(exact), worst_ce_in_tolerance_units=worst, step0_equals_independent_init=step0)
    landed = [r for r in res['runs'].values() if r['published']]
    res['Y2_step0'] = [r['step0_equals_independent_init'] for r in landed]
    res['Y3'] = dict(launched=sum(bool(r['events']) for r in res['runs'].values()), landed=len(landed),
                     complete=sum(r.get('status') == 'complete' and r.get('completed') == 50000 for r in landed),
                     event_patterns=sorted({'/'.join(r['events']) for r in res['runs'].values()}),
                     all_landed_consistent=all(r['config_matches_plan'] and r['train_pairs'] and r['manifest_ok'] and not r['manifest_missing']
                                               for r in landed))
    res['Y5'] = all(r['accuracies_exact'] and r['worst_ce_in_tolerance_units'] <= 1 for r in landed)
    if replay_id:
        e, d = entries[replay_id], runs_dir/replay_id
        a, b, y, _, sp = splits_cache[(e['data_seed'], e['train_fraction'])]
        XT = ind.one_hot_inputs(a[sp[0]], b[sp[0]], P)
        cks = {int(f.stem.split('_')[1]): f for f in d.glob('checkpoint_*.npz')}
        w = [x.copy() for x in init[e['init_seed']]]
        m, v = [np.zeros_like(x) for x in w], [np.zeros_like(x) for x in w]
        early = {}
        for t in range(1, 2001):
            ind.adamw_step(w, ind.loss_and_grad(w, *XT, y[sp[0]])[1], m, v, t, .001, 1.)
            if t in cks:
                with np.load(cks[t]) as z:
                    early[t] = max(float(np.abs(x - z[k]).max()) for x, k in zip(w, 'UVW'))
        late = {}
        last = max(cks)
        start = max(s for s in cks if s <= last - 1000)
        with np.load(cks[start]) as z:
            w, m, v = [z[k] for k in 'UVW'], [z[k] for k in ('mU', 'mV', 'mW')], [z[k] for k in ('vU', 'vV', 'vW')]
        for t in range(start + 1, last + 1):
            ind.adamw_step(w, ind.loss_and_grad(w, *XT, y[sp[0]])[1], m, v, t, .001, 1.)
        with np.load(cks[last]) as z:
            late = dict(start=start, end=last, max_abs_diff=max(float(np.abs(x - z[k]).max()) for x, k in zip(w, 'UVW')),
                        max_abs_weight=max(float(np.abs(z[k]).max()) for k in 'UVW'))
        res['Y4'] = dict(run=replay_id, early_max_abs_diff=early, late=late)
    Path(out_json).write_text(json.dumps(res, indent=1) + '\n')
    print({k: res[k] for k in ('Y3', 'Y5')}, 'Y2 step-0:', set(res['Y2_step0']), res.get('Y4'))


if __name__ == '__main__':
    {'splits': splits, 'runs': runs}[sys.argv[1]](*sys.argv[2:])
