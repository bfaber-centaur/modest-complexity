#!/usr/bin/env python3
"""Audit of the size/time lane's figure tables and evidence packets, rederived with
independent code from the published run files (metrics.jsonl values were checked
against the checkpoints by audit_preview.py; operand-label results are recomputed
from checkpoints with independent.py).

uv run --locked python research/audit/audit_size_time.py research/size-time research/coordinator/runs OUT.json
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402

lane, co_runs, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
fig = lane/'figures'
dirs = sorted(lane.glob('runs/*')) + sorted(co_runs.glob('*'))
runs = {d.name: [json.loads(s) for s in (d/'metrics.jsonl').read_text().splitlines()] for d in dirs}
cond = {k: k.split('-')[2] for k in runs}
width = {k: int(k.split('-')[3][1:]) for k in runs}
ce = lambda z, s='train': z['objective'][s]['cross_entropy']
report = {}

# 1. summary.csv: per-run descriptors, rederived from metrics.jsonl.
mismatch = []
for s in csv.DictReader((fig/'trajectories/summary.csv').open()):
    rows = runs[s['run_id']]; steps = [z['step'] for z in rows]
    fit = [z['train'] >= .99 for z in rows]
    first = steps[fit.index(True)]
    sustained = next(steps[i] for i in range(len(rows)) if all(fit[i:]))
    audit = [z['audit'] for z in rows]
    mine = dict(first_fit=first, sustained_fit=sustained,
                fit_drops_after_first=sum(not f for f, t in zip(fit, steps) if t >= first),
                max_audit=max(audit), step_max_audit=steps[audit.index(max(audit))],
                final_train=rows[-1]['train'], final_audit=rows[-1]['audit'], final_train_ce=ce(rows[-1]),
                final_audit_ce=ce(rows[-1], 'audit'), final_train_margin_min=rows[-1]['objective']['train']['margin_min'],
                final_audit_margin_median=rows[-1]['objective']['audit']['margin_median'], final_norm=rows[-1]['norm'])
    mismatch += [(s['run_id'], k) for k, v in mine.items() if float(s[k]) != float(v)]
report['summary_csv'] = dict(rows=len(runs), mismatched_cells=mismatch)

# 2. Packet 1 statements.
add = [k for k in runs if cond[k] == 'add']
fit_steps = {}
for k in add:
    fit_steps.setdefault(width[k], set()).add(next(z['step'] for z in runs[k] if z['train'] >= .99))
w16 = [k for k in add if width[k] == 16]
report['packet1'] = dict(
    first_fit_by_width={w: sorted(v) for w, v in sorted(fit_steps.items())},
    all_add_fit=all(any(z['train'] >= .99 for z in runs[k]) for k in add),
    any_run_audit_ge_95=any(z['audit'] >= .95 for k in runs for z in runs[k]),
    add_runs_with_max_audit_only_at_step0=sum(max(z['audit'] for z in runs[k][1:]) < runs[k][0]['audit'] for k in add),
    add_runs=len(add),
    step0_audit_range=(min(runs[k][0]['audit'] for k in add), max(runs[k][0]['audit'] for k in add)),
    final_audit_range=(min(runs[k][-1]['audit'] for k in add), max(runs[k][-1]['audit'] for k in add)),
    w16_final_train_ce=sorted(ce(runs[k][-1]) for k in w16),
    w16_final_train_margin_min=sorted(runs[k][-1]['objective']['train']['margin_min'] for k in w16),
    w128_train_ce_add_vs_shuf={c: {s: [ce(z) for z in runs[k] if z['step'] == s][0]
                                   for s in (300, 5000, 10000)}
                               for k, c in (('e1-co-add-w128-i30000-a1', 'add i30000'),
                                            ('e1-st-shuf-w128-i30000-a1', 'shuf i30000'))})

# 3. Packet 2: matched-loss table and the growth of held-out loss.
table = {}
for k in runs:
    if cond[k] == 'shuf':
        continue
    for lvl in (1e-2, 1e-4):
        z = next((z for z in runs[k] if ce(z) <= lvl), None)
        table.setdefault((cond[k], width[k], lvl), []).append(None if z is None else (z['step'], ce(z, 'audit')))
    table.setdefault((cond[k], width[k], 'final'), []).append(ce(runs[k][-1], 'audit'))
packet2 = {}
for (c, w, lvl), v in sorted(table.items(), key=lambda t: (t[0][0], t[0][1], str(t[0][2]))):
    if lvl == 'final':
        packet2[f'{c} w{w} audit CE at 50k'] = (round(min(v), 2), round(max(v), 2))
    elif all(x is None for x in v):
        packet2[f'{c} w{w} train CE<={lvl}'] = 'not reached'
    else:
        got = [x for x in v if x is not None]
        packet2[f'{c} w{w} train CE<={lvl}'] = dict(reached=f'{len(got)}/{len(v)}',
                                                     steps=(min(x[0] for x in got), max(x[0] for x in got)),
                                                     audit_ce=(round(min(x[1] for x in got), 2), round(max(x[1] for x in got), 2)))
by = lambda k, s: next(z for z in runs[k] if z['step'] == s)
packet2['add: min over runs of audit CE(50k) - audit CE(1000)'] = round(min(ce(by(k, 50000), 'audit') - ce(by(k, 1000), 'audit') for k in add), 3)
add0 = [k for k in runs if cond[k] == 'add0']
packet2['add0 final train CE, norm'] = [(ce(runs[k][-1]), round(runs[k][-1]['norm'], 1)) for k in add0]
mm = []
for s in csv.DictReader((fig/'trajectories/matched.csv').open()):
    if s['reached'] != 'True':
        continue
    z = next(z for z in runs[s['run_id']] if (z['step'] == int(float(s['level'])) if s['match'] == 'step'
                                               else ce(z) <= float(s['level'])))
    mm += [s['run_id']] * (int(s['step']) != z['step'] or float(s['audit_ce']) != ce(z, 'audit')
                           or float(s['train_ce']) != ce(z) or float(s['norm']) != z['norm'])
packet2['matched_csv_mismatched_rows'] = mm
report['packet2'] = packet2

# 4. Interpretation 1: "stationary state by about 20,000-25,000 updates" for widths >= 128.
stat = {}
for k in sorted(runs):
    late = [z for z in runs[k] if z['step'] >= 25000]
    u = [z['optimizer_update'] for z in late]
    stat[k] = dict(norm_25k_50k=(round(late[0]['norm'], 2), round(late[-1]['norm'], 2)),
                   train_ce_25k_50k=(ce(late[0]), ce(late[-1])),
                   max_cosine=None if u[0]['cosine'] is None else round(max(x['cosine'] for x in u), 4),
                   max_rel_gap_adaptive_decay=None if not u[0]['decay_norm'] else
                   round(max(abs(x['adaptive_norm'] - x['decay_norm']) / x['decay_norm'] for x in u), 4))
report['stationarity_from_25k'] = stat

# 5. Packet 3 recomputed from checkpoints: position-specific operand label sets L(a, b).
p = 31
rows3, mism3 = [], []
lane_res = {(r['run'], r['step']): r for r in map(json.loads, (fig/'operand_labels/results.jsonl').read_text().splitlines())}
for d in dirs:
    with np.load(d/'data.npz') as z:
        a, b, y, yadd, tr, au = z['a'], z['b'], z['y'], z['yadd'], z['train'], z['audit']
    Xa, Xb = ind.one_hot_inputs(a, b, p)
    with_a = (Xa[tr].T @ np.eye(p)[y[tr]]) > 0   # with_a[x, c]: label c seen with x in first position
    with_b = (Xb[tr].T @ np.eye(p)[y[tr]]) > 0
    L = (Xa[au] @ with_a + Xb[au] @ with_b) > 0
    rix = np.arange(len(au))
    for step in (1000, 5000, 20000, 50000):
        f = d/f'checkpoint_{step:06d}.npz'
        if not f.exists():
            continue
        with np.load(f) as z:
            logits = ind.forward((z['U'], z['V'], z['W']), Xa, Xb)[0][au]
        logp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        prob = np.exp(logp); pred = logits.argmax(1)
        mine = dict(audit_accuracy=float(np.mean(pred == y[au])), premise=float(np.mean(L[rix, y[au]])),
                    pred_in_L=float(np.mean(L[rix, pred])), mean_L_size=float(L.sum(1).mean()),
                    mass_L=float(np.mean((prob*L).sum(1))), mass_correct=float(np.mean(prob[rix, y[au]])))
        if cond[d.name] == 'shuf':  # rank of a+b among position-specific absent labels, where a+b is absent
            ok = ~L[rix, yadd[au]]; lg, ab, t = logits[ok], ~L[ok], yadd[au][ok]; r2 = np.arange(ok.sum())
            ab_other = ab.copy(); ab_other[r2, t] = False
            above = ((lg > lg[r2, t][:, None]) & ab_other).sum(1)
            mine.update(addition_label_absent_pairs=int(ok.sum()),
                        addition_rank_among_absent_mean=float((above/ab_other.sum(1)).mean()),
                        addition_top_among_absent=float((above == 0).mean()),
                        chance_top=float(np.mean(1/ab.sum(1))))
        ref = lane_res[(d.name, step)]
        dev = max(abs(mine[k] - ref[k]) for k in ('audit_accuracy', 'premise', 'pred_in_L', 'mean_L_size', 'mass_L', 'mass_correct'))
        rows3.append(dict(run=d.name, step=step, max_abs_diff_vs_lane=dev, **{k: round(v, 4) for k, v in mine.items()}))
report['packet3'] = dict(rows_compared=len(rows3), lane_rows=len(lane_res),
                         max_abs_diff_vs_lane=max(r['max_abs_diff_vs_lane'] for r in rows3), rows=rows3)
out.write_text(json.dumps(report, indent=1, default=float) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('stationarity_from_25k', 'packet3')}, indent=1, default=float))
