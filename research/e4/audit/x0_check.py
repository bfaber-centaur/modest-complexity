#!/usr/bin/env python3
"""e4 audit control X0 (QUESTION.md addendum 4, D16): exempt-reference decay from the untrained init (I),
the C1 random-feature starts (RF), the unedited 50k starts (A) and B with k permuted across units (S).

Per run: provenance (registry, config against the plan, projection hash, shared code at the pin),
update 0 against the start, and my replay with my own projection to update 300 (dynamics_check.replay).
Then common-pair accuracy at 2,000 and 5,000 and the pre-stated labels, beside the comparators.
Writes x0-check.json and x0-series.csv (own held-out accuracy, train CE, norm, decay norm every logged update).

uv run --locked python research/e4/audit/x0_check.py OUT_DIR
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dynamics_check as dc  # noqa: E402

AU, ME = HERE/'runs', dc.ME/'runs'
ARMS = {'I': 'e4-au-init-xref-w128-i{}-a1', 'RF': 'e4-au-rf-xref-w128-i{}-a1', 'A': 'e4-au-keep-xref-w128-i{}-a1',
        'S': 'e4-au-filt-xrefsham-w128-i{}-a1'}
COMPARATORS = {'B xref': (ME, 'e4-me-filt-wd1-xref-w128-i{}-a1'), 'B wd 1': (ME, 'e4-me-filt-wd1-replay-w128-i{}-a1'),
               'B wd 0': (ME, 'e4-me-filt-wd0-replay-w128-i{}-a1'), 'A wd 1': (dc.E3, 'e3-me-keep-w128-i{}-a1'),
               'RF wd 1': (AU, 'e4-au-rf-w128-i{}-a1'), 'RF wd 0': (AU, 'e4-au-rf-wd0-w128-i{}-a1')}
E1 = {30000: dc.ROOT/'research/coordinator/runs/e1-co-add-w128-i30000-a1',
      30001: dc.ROOT/'research/size-time/runs/e1-st-add-w128-i30001-a1',
      30002: dc.ROOT/'research/size-time/runs/e1-st-add-w128-i30002-a1'}


def label(acc):
    return 'rescued' if acc >= .90 else 'not rescued' if acc <= .30 else 'intermediate'


def start_weights(cfg, seed):
    if cfg['init_weights'] is None:
        return [x.copy() for x in dc.ind.initial_weights(dc.P, dc.H, seed)]
    with np.load(dc.ROOT/cfg['init_weights']) as z:
        return [z[k].copy() for k in 'UVW']


def main(out_dir):
    reg = dc.registry(HERE/'registry.jsonl')
    run0 = dc.load(dc.E3/'e3-me-filt-w128-i30000-a1')
    sets = dc.ladder_sets(run0['d']['a'], run0['d']['b'], dc.tr30, dc.tr50, dc.COMMON, dc.P)
    acc_at = lambda run, t: dc.sigma_margin_and_acc([dc.state(run['ck'][t])[k] for k in 'UVW'], run['d'], sets)[0]
    res, series = dict(runs={}, comparators={}, labels={}), []
    for seed in dc.SEEDS:
        for arm, pat in ARMS.items():
            rid = pat.format(seed)
            run = dc.load(AU/rid)
            cfg, plan, ev = run['cfg'], dc.PLAN_RUNS[rid], reg.get(rid, [])
            proj = dc.ROOT/cfg['decay_projection']
            w0 = start_weights(cfg, seed)
            s0 = dc.state(run['ck'][0])
            with np.load(proj) as z:
                mine = dc.my_projection(z['k'], z['live'])
            zeros = [np.zeros_like(x) for x in w0]
            rp = dc.replay(run, w0, zeros, zeros, 0, 300, 1., dc.rest_of(mine))
            r = dict(arm=arm, seed=seed,
                     provenance=dict(registry=[e['status'] for e in ev],
                                     completed=[e.get('completed_steps') for e in ev if e['status'] == 'complete'],
                                     decay_on=cfg['decay_on'], projection=cfg['decay_projection'],
                                     projection_sha=dc.sha(proj) == cfg['decay_projection_sha256'] == plan['decay_projection_sha256'],
                                     init_weights=cfg['init_weights'] == plan.get('init_weights'),
                                     init_sha=(cfg['init_weights_sha256'] == plan.get('init_weights_sha256') == dc.sha(dc.ROOT/cfg['init_weights']))
                                     if cfg['init_weights'] else cfg['init_weights_sha256'] is None,
                                     wd=cfg['wd'], steps=cfg['steps'], shared=dc.shared_code(cfg['runtime']['git_commit']),
                                     git_status=cfg['runtime']['git_status']),
                     update0_equals_start=all(np.array_equal(s0[k], w0[i]) for i, k in enumerate('UVW')),
                     replay300=dc.summarise(rp))
            pv = r['provenance']
            r['provenance']['PASS'] = bool(pv['registry'] == ['running', 'complete'] and pv['completed'] == [5000] and pv['decay_on'] == 'rest'
                                           and pv['projection_sha'] and pv['init_weights'] and pv['init_sha'] and pv['wd'] == 1
                                           and pv['steps'] == 5000 and pv['shared']['equals_pin_8108d8b'])
            r['replay300']['PASS'] = bool(rp['rows'] == rp['rows_acc_equal'] == rp['rows_within'] == 30
                                          and max(rp['checkpoints'].values()) <= dc.TOL_W)
            r['acc'] = {t: acc_at(run, t) for t in (2000, 5000)}
            r['label'] = label(r['acc'][5000])
            res['runs'][rid] = r
            for t, row in sorted(run['rows'].items()):
                series.append(dict(group=arm, seed=seed, step=t, own=dc.own(row, run['d']), train_ce=row['objective']['train']['cross_entropy'],
                                   norm=row['norm'], decay_norm=(row['optimizer_update'] or {}).get('decay_norm')))
        for name, (root, pat) in COMPARATORS.items():
            run = dc.load(root/pat.format(seed))
            res['comparators'][f'{name}/{seed}'] = {t: acc_at(run, t) for t in (2000, 5000) if t in run['ck']}
            for t, row in sorted(run['rows'].items()):
                if t <= 5000:
                    series.append(dict(group=name, seed=seed, step=t, own=dc.own(row, run['d']),
                                       train_ce=row['objective']['train']['cross_entropy'], norm=row['norm'],
                                       decay_norm=(row['optimizer_update'] or {}).get('decay_norm')))
        run = dc.load(E1[seed])
        res['comparators'][f'I wd 1 (e1)/{seed}'] = dict({t: acc_at(run, t) for t in (2000, 5000) if t in run['ck']},
                                                         own_5000=dc.own(run['rows'][5000], run['d']) if 5000 in run['rows'] else None)
        for t, row in sorted(run['rows'].items()):
            if t <= 5000:
                series.append(dict(group='I wd 1 (e1)', seed=seed, step=t, own=dc.own(row, run['d']),
                                   train_ce=row['objective']['train']['cross_entropy'], norm=row['norm'], decay_norm=None))
    for arm in ARMS:
        labs = [res['runs'][ARMS[arm].format(s)]['label'] for s in dc.SEEDS]
        res['labels'][arm] = dict(per_seed=labs, arm=labs[0] if len(set(labs)) == 1 else 'mixed')
    lab = {a: v['arm'] for a, v in res['labels'].items()}
    res['reading'] = ('built in: SUPPORTED LOCALLY' if lab['I'] == 'rescued'
                      else 'built in: CONTRADICTED' if all(lab[a] == 'not rescued' for a in ARMS) else 'see per-arm labels')
    res['checks_PASS'] = dict(provenance=all(r['provenance']['PASS'] for r in res['runs'].values()),
                              update0=all(r['update0_equals_start'] for r in res['runs'].values()),
                              replay300=all(r['replay300']['PASS'] for r in res['runs'].values()))
    Path(out_dir, 'x0-check.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    with open(Path(out_dir, 'x0-series.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, list(series[0]))
        w.writeheader()
        w.writerows(series)
    print(json.dumps(dict(labels=res['labels'], reading=res['reading'], checks=res['checks_PASS'],
                          acc={k: v['acc'] for k, v in res['runs'].items()}, comparators=res['comparators']), indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
