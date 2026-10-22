#!/usr/bin/env python3
"""Early-rise probe (research/e3/audit/QUESTION-4.md; DECISIONS D13): does B's held-out rise
after the filter edit need fresh Adam's +-lr first steps?

From each published B start file, on the 299 training pairs, with my own code
(research/audit/independent.py): the reference (Adam lr 1e-3, 300 updates), Adam at lr 1e-4
(3,000 updates; same lr x updates and total decay) and gradient descent with the reference's
decoupled decay and a step size eta matched to the reference's first Adam step in norm
(3,000 updates). Diagnostic replays only: no run IDs, nothing published as a run.

uv run --locked python research/e3/audit/probe.py OUT_DIR
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mech_check as mc  # noqa: E402  (split, common pairs, loaders; same definitions as investigation 3)
ind = mc.ind
tr, y, COMMON = mc.tr, mc.y, mc.COMMON
XT = (mc.XA[tr], mc.XB[tr])


def first_adam_step_norm(w, lr):
    g = ind.loss_and_grad(w, *XT, y[tr])[1]
    return float(np.sqrt(sum(np.sum((lr*x/(np.abs(x) + ind.EPS))**2) for x in g))), float(np.sqrt(sum(np.sum(x**2) for x in g)))


def run(w, condition, steps, eta=None):
    w = [x.copy() for x in w]
    m, v = [np.zeros_like(x) for x in w], [np.zeros_like(x) for x in w]
    rows = []
    for t in range(steps + 1):
        if t:
            g = ind.loss_and_grad(w, *XT, y[tr])[1]
            if condition == 'gradient descent':
                w = [x*(1 - 1e-3) - eta*gx for x, gx in zip(w, g)]
            else:
                ind.adamw_step(w, g, m, v, t, 1e-3 if condition == 'reference' else 1e-4, 1.)
        z = mc.logits(w)
        rows.append(dict(update=t, train_acc=mc.acc(z[tr], tr), common_acc=mc.acc(z[COMMON], COMMON),
                         train_ce=float(np.mean(np.logaddexp.reduce(z[tr], axis=1) - z[tr, y[tr]]))))
    return rows


def summary(rows):
    r0 = rows[0]
    fit = next((r for r in rows if r['train_acc'] == 1.0), None)
    at = fit or max(rows, key=lambda r: (r['train_acc'], -r['update']))
    peak = max(rows, key=lambda r: r['common_acc'])
    return dict(update0=r0['common_acc'], fit_reached=fit is not None, fit_update=at['update'], at_fit=at['common_acc'],
                gain_at_fit=at['common_acc'] - r0['common_acc'], peak=peak['common_acc'], peak_update=peak['update'],
                final=rows[-1]['common_acc'], final_train_acc=rows[-1]['train_acc'])


def main(out_dir):
    out = Path(out_dir)
    res, dense = {}, []
    for s in mc.SOURCES:
        w = mc.load(mc.START('filt', s))
        step_norm, g_norm = first_adam_step_norm(w, 1e-3)
        eta = step_norm/g_norm
        res[s] = dict(eta=eta, first_adam_step_norm=step_norm, initial_gradient_norm=g_norm)
        for condition, steps in (('reference', 300), ('small Adam', 3000), ('gradient descent', 3000)):
            rows = run(w, condition, steps, eta)
            res[s][condition] = summary(rows)
            dense += [dict(source=s, condition=condition, **r) for r in rows]
    for condition in ('small Adam', 'gradient descent'):
        gains = [res[s][condition]['gain_at_fit'] for s in mc.SOURCES]
        res[condition] = dict(gains=gains, at_least_010_all=all(g >= .10 for g in gains), below_005_all=all(g < .05 for g in gains))
    both = [res[c] for c in ('small Adam', 'gradient descent')]
    res['verdict'] = ('a property of fitting from the edited point' if all(c['at_least_010_all'] for c in both) else
                      'tied to the restart\'s step rule' if all(c['below_005_all'] for c in both) else 'mixed')
    res['reference_matches_investigation_3'] = [res[s]['reference']['peak'] for s in mc.SOURCES]
    with open(out/'probe.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(dense[0])); wr.writeheader()
        wr.writerows({k: (float(f'{v:.10g}') if isinstance(v, float) else v) for k, v in r.items()} for r in dense)
    (out/'probe.json').write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1) + '\n')
    for s in mc.SOURCES:
        print(s, round(res[s]['eta'], 4), {c: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[s][c].items()}
                                           for c in ('reference', 'small Adam', 'gradient descent')})
    print(res['verdict'])


if __name__ == '__main__':
    main(sys.argv[1])
