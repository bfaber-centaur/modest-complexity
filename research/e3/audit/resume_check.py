#!/usr/bin/env python3
"""Sixth e3 audit investigation (research/e3/audit/QUESTION-6.md): dynamics packet 2, the ten
runs resumed from the published update-50,000 checkpoints. P1 resume provenance, P2 a short
independent replay after the resume (and a 10k replay reported as found), P3 the packet-1
settling correction from metrics.jsonl, P4 Table 4 on the ladder eval pairs, P5 the launch
deviation (D16).

uv run --locked python research/e3/audit/resume_check.py OUT_DIR
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(root/'research/audit'))
import independent as ind  # noqa: E402
import ladder_check as lc  # noqa: E402

P, CONTROL = 31, 32000
DY = root/'research/e3/dynamics'
plan = json.loads((root/'research/e3/plan.json').read_text())
ENTRY = {e['run_id']: e for e in plan['runs'] if e['run_id'].startswith('e3-dy-')}
CONT = sorted(r for r in ENTRY if '-c1' in r)
REPLAY = 'e3-dy-add-w128-i30001-tf40-d41002-c100k-a1'
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()


def source(run):
    return run.replace('-c100k', '').replace('-c150k', '')


def load(f, keys='UVW'):
    with np.load(f) as z:
        return [z[k] for k in keys]


def p1():
    events = [json.loads(l) for l in (DY/'registry.jsonl').read_text().splitlines() if l.strip()]
    out = {}
    for run in CONT:
        d, s = DY/'runs'/run, DY/'runs'/source(run)
        cfg = json.loads((d/'config.json').read_text())
        src_ck = s/'checkpoint_050000.npz'
        src_man = json.loads((s/'manifest.json').read_text())['files']['checkpoint_050000.npz']['sha256']
        same = all(np.array_equal(x, q) for x, q in zip(load(d/'checkpoint_050000.npz', ('U', 'V', 'W', 'mU', 'mV', 'mW', 'vU', 'vV', 'vW', 'step')),
                                                       load(src_ck, ('U', 'V', 'W', 'mU', 'mV', 'mW', 'vU', 'vV', 'vW', 'step'))))
        rt = json.loads((d/'runtime.json').read_text())
        out[run] = dict(resume_names_source=(root/cfg['resume']).resolve() == src_ck.resolve() == (root/ENTRY[run]['resume']).resolve(),
                        resume_sha_matches=cfg['resume_sha256'] == sha(src_ck) == src_man == ENTRY[run]['resume_sha256'],
                        published_50k_equals_source=bool(same),
                        events=[e['status'] for e in events if e['run_id'] == run],
                        completed=rt['completed_steps'], planned=ENTRY[run]['steps'],
                        steps_in_config=cfg['steps'])
        r = out[run]
        r['PASS'] = bool(r['resume_names_source'] and r['resume_sha_matches'] and same and r['events'] == ['running', 'complete']
                         and r['completed'] == r['planned'] == r['steps_in_config'])
    return out


def p2():
    e = ENTRY[REPLAY]
    a, b, y, _, sp = ind.regenerate_split(P, e['data_seed'], CONTROL, e['train_fraction'])
    XT, XA, XB = ind.one_hot_inputs(a[sp[0]], b[sp[0]], P), *ind.one_hot_inputs(a, b, P)
    logged = {r['step']: r for r in map(json.loads, (DY/'runs'/REPLAY/'metrics.jsonl').read_text().splitlines())}
    w, m, v = (load(DY/'runs'/source(REPLAY)/'checkpoint_050000.npz', k) for k in ('UVW', ('mU', 'mV', 'mW'), ('vU', 'vV', 'vW')))
    rows, worst, exact = [], 0., True
    for t in range(50001, 60001):
        ind.adamw_step(w, ind.loss_and_grad(w, *XT, y[sp[0]])[1], m, v, t, .001, 1.)
        if t <= 50100 and t in logged:
            z = ind.forward(w, XA, XB)[0]
            for name, ix in zip(('train', 'cal', 'audit'), sp):
                exact &= float(np.mean(z[ix].argmax(1) == y[ix])) == logged[t][name]
                ce = float(np.mean(np.logaddexp.reduce(z[ix], axis=1) - z[ix, y[ix]]))
                lg = logged[t]['objective'][name]['cross_entropy']
                worst = max(worst, abs(ce - lg)/(1e-8*abs(lg) + 1e-14))
                rows.append(dict(step=t, split=name, ce_replay=ce, ce_logged=lg))
    pub = load(DY/'runs'/REPLAY/'checkpoint_060000.npz')
    lad = lc.Ladder(e['data_seed'])
    return dict(run=REPLAY, logged_rows_compared=sorted({r['step'] for r in rows}), accuracies_exact=bool(exact),
                worst_ce_in_tolerance_units=worst, PASS=bool(exact and worst <= 1 and rows),
                replay_to_60k=dict(relative_weight_distance=float(np.sqrt(sum(np.sum((x - q)**2) for x, q in zip(w, pub))/sum(np.sum(q**2) for q in pub))),
                                   replay=lad.score(w, e['train_fraction']), published=lad.score(pub, e['train_fraction'])))


def p3():
    lane = {r['run_id']: r for r in csv.DictReader(open(DY/'figures/packet1_settle.csv'))}
    out, rows = {}, []
    for run, e in ENTRY.items():
        if '-c1' in run:
            continue
        sp = ind.regenerate_split(P, e['data_seed'], CONTROL, e['train_fraction'])[4]
        n_cal, n_aud = len(sp[1]), len(sp[2])
        logged = [r for r in map(json.loads, (DY/'runs'/run/'metrics.jsonl').read_text().splitlines()) if 20000 <= r['step'] <= 50000]
        acc = {r['step']: (r['cal']*n_cal + r['audit']*n_aud)/(n_cal + n_aud) for r in logged}
        final = acc[50000]
        settled = min(s for s in acc if all(abs(acc[q] - final) <= .01 for q in acc if q >= s))
        grid = {s: x for s, x in acc.items() if s % 1000 == 0}  # post hoc: the lane's 1,000-update grid (continuation.settle)
        settled_grid = min(s for s in grid if all(abs(grid[q] - final) <= .01 for q in grid if q >= s))
        ref = lane[run]
        match = settled == int(ref['settled_within_001']) and abs(final - float(ref['own_heldout_50k'])) < 1e-12
        out[run] = dict(share=e['train_fraction'], own_heldout_30k=acc[30000], own_heldout_40k=acc[40000], own_heldout_50k=final,
                        settled_within_001=settled, matches_lane=bool(match), settled_on_1000_grid=settled_grid,
                        matches_lane_on_1000_grid=bool(settled_grid == int(ref['settled_within_001'])
                                                       and all(abs(acc[k] - float(ref[f'own_heldout_{k//1000}k'])) < 1e-12 for k in (30000, 40000, 50000))),
                        share_of_30k_50k_change_by_40k=(acc[40000] - acc[30000])/(final - acc[30000]) if final != acc[30000] else None)
        rows.append(out[run] | dict(run_id=run))
    by = lambda share: sorted(v['settled_within_001'] for v in out.values() if v['share'] == share)
    return dict(runs=out, PASS=all(v['matches_lane'] for v in out.values()), settled_range={s: [by(s)[0], by(s)[-1]] for s in (.45, .5)},
                post_hoc_matches_lane_on_1000_grid=all(v['matches_lane_on_1000_grid'] for v in out.values()),
                all_45_settled_before_50k=all(x < 50000 for x in by(.45)))


def p4():
    table = {}
    lads = {}
    for run in CONT:
        e = ENTRY[run]
        lad = lads.setdefault(e['data_seed'], lc.Ladder(e['data_seed']))
        cks = sorted(int(f.stem.split('_')[1]) for f in (DY/'runs'/run).glob('checkpoint_*.npz'))
        table[run] = {s: lad.score(load(DY/'runs'/run/f'checkpoint_{s:06d}.npz'), e['train_fraction']) for s in cks}
    lane = {}
    for r in csv.DictReader(open(DY/'figures/continuation.csv')):
        lane[(r['run_id'], int(r['step']))] = r
    worst = max(abs(m[k] - float(lane[(run, s)][k])) for run, t in table.items() for s, m in t.items() if (run, s) in lane
                for k in ('acc', 'P_yS', 'P_SN', 'margin_N_sigma', 'rank_absent'))
    change45 = {run: t[150000]['acc'] - t[50000]['acc'] for run, t in table.items() if '-c150k' in run}
    change40 = {run: t[100000]['acc'] - t[50000]['acc'] for run, t in table.items() if '-c100k' in run}
    return dict(compared_rows=sum((run, s) in lane for run, t in table.items() for s in t), worst_abs_diff_from_lane=worst,
                PASS=worst <= 1e-9, acc_change_45_50k_to_150k=change45, acc_change_40_50k_to_100k=change40,
                max_abs_acc_change_45=max(map(abs, change45.values())),
                values={run: {s: {k: m[k] for k in ('acc', 'margin_N_sigma')} for s, m in t.items()} for run, t in table.items()})


def p5():
    diff = subprocess.run(['git', 'diff', '--name-only', '72cfb5d', '2f0aa7a'], cwd=root, capture_output=True, text=True).stdout.split()
    out = {}
    for run in CONT:
        cfg = json.loads((DY/'runs'/run/'config.json').read_text())
        man = json.loads((DY/'runs'/run/'manifest.json').read_text())
        out[run] = dict(producing_code_commit=man['producing_code_commit'][:7], git_commit=str(cfg['runtime'].get('git_commit'))[:7],
                        git_status=cfg['runtime']['git_status'])
    status_ok = all(str(v['git_status']).strip().split() == ['M', 'research/e3/dynamics/registry.jsonl'] for v in out.values())
    return dict(files_changed_between_pin_and_2f0aa7a=diff, runs=out,
                PASS=diff == ['research/e3/dynamics/QUESTION.md'] and all(v['producing_code_commit'] == '2f0aa7a' for v in out.values()) and status_ok)


def main(out_dir):
    res = dict(P1=p1(), P3=p3(), P4=p4(), P5=p5(), P2=p2())
    res['P1_all'] = all(v['PASS'] for v in res['P1'].values())
    res['rule3_reread'] = ('CONTRADICTED' if res['P3']['all_45_settled_before_50k'] and res['P4']['max_abs_acc_change_45'] <= .02
                           else 'unchanged')
    (Path(out_dir)/'resume-check.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps(dict(P1=res['P1_all'], P2={k: v for k, v in res['P2'].items()}, P3={k: v for k, v in res['P3'].items() if k != 'runs'},
                          P4={k: v for k, v in res['P4'].items() if k != 'values'}, P5=res['P5'], rule3=res['rule3_reread']), indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
