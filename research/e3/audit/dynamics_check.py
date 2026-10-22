#!/usr/bin/env python3
"""Dynamics checks (research/e3/audit/QUESTION-2.md, Y1-Y2) from the split rule alone.

Rebuilds every e3 ladder split with the e1 audit's independent rewrite of the split rule
(research/audit/independent.py, which imports nothing from experiment.py) and checks it
against the plan, the coordinator's ladders.json and the saved evaluation sets: hashes and
actual training counts, nesting, transpose closure, partition, the evaluation pairs as the
complement of the 50% training set, the label-set sizes, and the crossed design.

uv run --locked python research/e3/audit/dynamics_check.py splits OUT.json
"""
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


if __name__ == '__main__':
    {'splits': splits}[sys.argv[1]](*sys.argv[2:])
