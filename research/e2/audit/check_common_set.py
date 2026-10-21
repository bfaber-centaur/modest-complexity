#!/usr/bin/env python3
"""Independent check of research/e2/coordinator/common_set.py (DISPATCH item 3), which
uses experiment.forward. Recomputes from the frozen split files and each run's
data.npz: nesting of the two training sets, the common never-trained set and its
identity with 50% cal + audit, its hash and class counts; then every checkpoint's
accuracy, cross-entropy and median margin on that set with research/audit/independent.py,
compared with common_set_scores.csv.

uv run --locked python research/e2/audit/check_common_set.py > OUT.json
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root/'research/audit'))
import independent as ind  # noqa: E402

co = root/'research/e2/coordinator'
s30, s50 = (np.load(root/f'research/data/{n}') for n in ('e1-split.npz', 'e1f2-split-tf50.npz'))
a, b, y = s30['a'], s30['b'], s30['yadd']
p = 31
assert all(np.array_equal(s30[k], s50[k]) for k in ('a', 'b', 'yadd')) and np.array_equal(y, (a + b) % p)
sets = {f'{k}{n}': set(s[k].tolist()) for n, s in (('30', s30), ('50', s50)) for k in ('train', 'cal', 'audit')}
never = np.ones(p*p, bool); never[s30['train']] = False; never[s50['train']] = False
common = np.flatnonzero(never)
claimed = json.loads((co/'common_set.json').read_text())
saved = np.load(root/'research/e2/data/common-never-trained-30-50.npz')['index']
mine = dict(
    train_nested=sets['train30'] <= sets['train50'], train30=len(sets['train30']), train50=len(sets['train50']),
    added_by_50=len(sets['train50'] - sets['train30']), common_never_trained=len(common),
    common_unordered_groups=len({(min(i, j), max(i, j)) for i, j in zip(a[common], b[common])}),
    common_transpose_closed=set(zip(a[common], b[common])) == set(zip(b[common], a[common])),
    common_class_counts=np.bincount(y[common], minlength=p).tolist(),
    common_equals_50pct_cal_plus_audit=set(common.tolist()) == sets['cal50'] | sets['audit50'],
    common_inside_30pct_audit=len(set(common.tolist()) & sets['audit30']),
    common_inside_30pct_cal=len(set(common.tolist()) & sets['cal30']),
    common_sha256=ind.hash_arrays(common))
added = sets['train50'] - sets['train30']
out = dict(fields_equal={k: mine[k] == claimed[k] for k in mine}, saved_index_equal=bool(np.array_equal(saved, common)),
           added_by_50_from_30cal=len(added & sets['cal30']), added_by_50_from_30audit=len(added & sets['audit30']),
           rows=[])
worst = dict(acc=0., ce=0., margin_median=0.)
for r in csv.DictReader(open(co/'common_set_scores.csv')):
    run = next(root.glob(f"research/*/runs/{r['run_id']}"))
    with np.load(run/'data.npz') as d:
        assert np.array_equal(d['yadd'], y) and set(d['train'].tolist()) == sets['train30' if r['share'] == '30%' else 'train50']
    with np.load(run/f"checkpoint_{int(r['step']):06d}.npz") as z:
        ev = ind.evaluate(ind.forward((z['U'], z['V'], z['W']), *ind.one_hot_inputs(a, b, p))[0][common], y[common])
    diff = dict(acc=abs(ev['acc'] - float(r['common_acc'])), ce=abs(ev['cross_entropy'] - float(r['common_ce'])),
                margin_median=abs(ev['margin_median'] - float(r['common_margin_median'])))
    worst = {k: max(worst[k], diff[k]) for k in worst}
    out['rows'].append(dict(run=r['run_id'], step=int(r['step']), acc=round(ev['acc'], 4), ties=ev['ties'],
                            cross_entropy=round(ev['cross_entropy'], 4)))
final = [x for x in out['rows'] if x['step'] == 50000]
out.update(rows_checked=len(out['rows']), worst_abs_diff=worst,
           acc_at_50k={sh: [x['acc'] for x in final if ('tf50' in x['run']) == (sh == '50%')] for sh in ('30%', '50%')})
print(json.dumps(out))
