#!/usr/bin/env python3
"""e4 audit, QUESTION.md addendum 2: the mechanisms lane's channels.py (PR #35) against my
channels_check.py (a), on the six C1 runs (b), and whether "outvoted, not erased" is forced by the
basis (c: floor, basis sham, start content of the reference readout).

uv run --locked --group analysis python research/e4/audit/lane_check.py OUT_DIR
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import channels_check as mine  # noqa: E402  (puts the shared paths on sys.path)
from rf_starts import ROOT  # noqa: E402
sys.path.insert(0, str(ROOT/'research/e4/mechanisms'))
import channels as lane  # noqa: E402

e3 = lane.e3
SEEDS = mine.SEEDS
LR, DRAWS, SHAM_SEED = .001, 8, 41100
E4 = json.loads((ROOT/'research/e4/plan.json').read_text())
E4_PLAN = {r['run_id']: {**E4['common'], **r} for r in E4['runs']}
E3_PLAN = {r['run_id']: r for r in e3.entries()}


def entry(run_dir):
    return E4_PLAN.get(run_dir.name) or E3_PLAN[run_dir.name]


def centred(x):
    return x - x.mean(-1, keepdims=True)


def part_a():
    """Lane channels() against mine on every published state of B (wd 1, 0), A and the C1 runs."""
    worst, n, not_live = 0., 0, {}
    for seed in SEEDS:
        for (arm, wd), run_dir in mine.arms(seed).items():
            if arm == 'C':
                continue
            meta, d, ws = e3.check_run(run_dir, entry(run_dir))
            k, _, live = e3.dominant(ws)
            not_live[f'{arm}/{seed}/{wd}'] = int((~live).sum())
            sets = mine.Sets(d, e3.COMMON)
            for s in mine.load_dir(run_dir)[1]:
                w = mine.weights(run_dir, s)
                ch, z, _ = lane.channels(w, d, k, live)
                m = mine.channels(w, k, d, e3.COMMON, sets)[0]
                h = np.maximum(w[0][d['a']] + w[1][d['b']], 0)
                mu = h[d['train']].mean(0)
                Wdc, Wref, Woff, _ = mine.label_split(w[2], k)
                my = dict(offset=np.broadcast_to(mu @ (w[2] - Wdc), ch['ref'].shape), ref=(h - mu)[e3.COMMON] @ Wref,
                          off=(h - mu)[e3.COMMON] @ Woff)
                scale = np.abs(centred(z)).max()
                worst = max(worst, max(np.abs(centred(ch[q]) - centred(my[q])).max()/scale for q in my))
                n += 1
    return dict(states=n, max_rel_channel_diff=float(worst), units_not_live_at_update0=not_live, PASS=bool(worst <= 1e-10))


def part_b(out_dir):
    """The lane's analyse() on the six C1 runs; the rule of addendum 2 (b) against the lane's B rows."""
    src = [('c1', mine.AU/f"e4-au-rf{'-wd0' if wd == 0 else ''}-w128-i{s}-a1", None,
            E4_PLAN[f"e4-au-rf{'-wd0' if wd == 0 else ''}-w128-i{s}-a1"]) for s in SEEDS for wd in (1., 0.)]
    rows, _ = lane.analyse(src)
    with open(Path(out_dir, 'lane-channels-c1.csv'), 'w', newline='') as f:
        out = csv.DictWriter(f, list(rows[0]))
        out.writeheader()
        out.writerows(rows)
    lane_b = [r for r in csv.DictReader(open(ROOT/'research/e4/mechanisms/results/channels.csv'))
              if r['source'] == 'e3' and r['arm'] == 'filt']
    get = lambda rr, seed, wd, step: next(r for r in rr if int(r['init_seed']) == seed and float(r['wd']) == wd and int(r['step']) == step)
    res = {}
    for seed in SEEDS:
        per = {}
        for name, rr in (('rf', rows), ('B', lane_b)):
            for wd in (1., 0.):
                a, b = get(rr, seed, wd, 300), get(rr, seed, wd, 2000)
                dd = {q: float(b[f'margin_S_{q}']) - float(a[f'margin_S_{q}']) for q in ('total', 'ref', 'off', 'offset')}
                per[f'{name}/wd{wd:g}'] = dict(change=dd, share={q: dd[q]/dd['total'] for q in ('ref', 'off', 'offset')})
        r1, b1, r0 = per['rf/wd1'], per['B/wd1'], per['rf/wd0']
        both_fall = r1['change']['ref'] < 0 and r1['change']['off'] < 0
        share_ok = abs(r1['share']['ref'] - b1['share']['ref']) <= .15
        offset_ok = abs(r1['share']['offset']) < .05
        rises0 = r0['change']['ref'] > 0
        one_channel = max(r1['share']['ref'], r1['share']['off']) >= .9 or r1['change']['ref'] > 0
        per.update(both_fall=both_fall, ref_share_within_015_of_B=share_ok, offset_under_5pct=offset_ok,
                   ref_rises_wd0=rises0, one_channel_or_ref_rises=one_channel,
                   supported=bool(both_fall and share_ok and offset_ok and rises0))
        res[seed] = per
    verdict = ('SUPPORTED LOCALLY' if all(res[s]['supported'] for s in SEEDS)
               else 'CONTRADICTED' if all(res[s]['one_channel_or_ref_rises'] for s in SEEDS) else 'AMBIGUOUS')
    late = {f'{seed}/{step}': dict(rf=float(get(rows, seed, 1., step)['acc_ref_only']), B=float(get(lane_b, seed, 1., step)['acc_ref_only']))
            for seed in SEEDS for step in (5000, 20000)}
    return dict(per_seed=res, verdict=verdict, ref_only_acc_wd1=late)


def part_c():
    """Floor (A), basis sham (k permuted across live units), start content of the reference readout."""
    rng = np.random.default_rng(SHAM_SEED)
    res = {}
    for seed in SEEDS:
        runs = mine.arms(seed)
        _, dA, wsA = e3.check_run(runs[('A', 1.)], entry(runs[('A', 1.)]))
        kA, _, liveA = e3.dominant(wsA)
        acc = lambda ch, d: float(np.mean(ch['ref'].argmax(1) == d['yadd'][e3.COMMON]))
        floor = {s: acc(lane.channels(mine.weights(runs[('A', 1.)], s), dA, kA, liveA)[0], dA) for s in (5000, 20000)}
        for arm in ('B', 'rf'):
            run_dir = runs[(arm, 1.)]
            _, d, ws = e3.check_run(run_dir, entry(run_dir))
            k, _, live = e3.dominant(ws)
            W_ref0 = lane.split_readout(ws[2], k, live)[0]
            row = dict(floor_A=floor)
            for s in (5000, 20000):
                w = mine.weights(run_dir, s)
                true = acc(lane.channels(w, d, k, live)[0], d)
                sham = []
                for _ in range(DRAWS):
                    kp = k.copy()
                    kp[live] = rng.permutation(k[live])
                    sham.append(acc(lane.channels(w, d, kp, live)[0], d))
                row[s] = dict(true=true, sham_median=float(np.median(sham)), sham_max=float(np.max(sham)))
            for s in mine.load_dir(run_dir)[1]:  # pre-stated at 2,000, 5,000, 20,000; every published state for the figure
                W_ref = lane.split_readout(mine.weights(run_dir, s)[2], k, live)[0]
                rho = (1 - LR)**s
                row[f'start_{s}'] = dict(rho=rho, s_t=float(rho*np.linalg.norm(W_ref0)/np.linalg.norm(W_ref)),
                                         c_t=float(np.sum(W_ref*W_ref0)/np.sum(W_ref0**2)),
                                         cos=float(np.sum(W_ref*W_ref0)/np.linalg.norm(W_ref)/np.linalg.norm(W_ref0)))
            res[f'{arm}/{seed}'] = row
    above_floor = all(res[f'{a}/{s}'][t]['true'] - res[f'{a}/{s}']['floor_A'][t] >= .10 for a in ('B', 'rf') for s in SEEDS for t in (5000, 20000))
    forced = all(abs(res[f'{a}/{s}'][t]['true'] - res[f'{a}/{s}'][t]['sham_median']) <= .10 for a in ('B', 'rf') for s in SEEDS for t in (5000, 20000))
    carried = all(res[f'{a}/{s}'][t]['sham_median'] <= res[f'{a}/{s}']['floor_A'][t] + .05 for a in ('B', 'rf') for s in SEEDS for t in (5000, 20000))
    slot = all(res[f'{a}/{s}']['start_5000']['s_t'] < .05 for a in ('B', 'rf') for s in SEEDS)
    return dict(values=res, above_floor_by_010=above_floor, sham_within_010_of_true=forced, sham_at_most_floor_plus_005=carried,
                start_share_under_005_at_5000=slot)


def main(out_dir):
    res = dict(a=part_a(), b=part_b(out_dir), c=part_c())
    Path(out_dir, 'lane-check.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv[1])
