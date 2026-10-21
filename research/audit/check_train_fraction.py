#!/usr/bin/env python3
"""Review check for the --train-fraction change (fe284a5): the default must leave the
e1 split and the historical single-stream split byte-identical, and the follow-up 2
split must match an independent rebuild, the frozen file and the plan's hash.

uv run --locked python research/audit/check_train_fraction.py OLD_EXPERIMENT_PY OUT.json
"""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
import experiment as new  # noqa: E402
import independent as ind  # noqa: E402

spec = importlib.util.spec_from_file_location('experiment_old', sys.argv[1])
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)


def same(x, y):
    return all(np.array_equal(u, v) for u, v in zip(x[:4], y[:4])) and all(np.array_equal(u, v) for u, v in zip(x[4], y[4]))


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


out = {}
out['default_equals_previous_code'] = same(new.data(31, 31000, 32000), old.data(31, 31000, 32000))
out['single_stream_equals_previous_code'] = same(new.data(31, 31000), old.data(31, 31000))
e1 = load(ROOT/'research/data/e1-split.npz')
a, b, y, ysh, s = new.data(31, 31000, 32000)
out['default_equals_frozen_e1'] = all(np.array_equal(u, e1[k]) for u, k in
                                      zip((a, b, y, ysh, *s), ('a', 'b', 'yadd', 'yshuffled', 'train', 'cal', 'audit')))
plan = json.loads((ROOT/'research/plan-followups.json').read_text())
tf = {r['train_fraction'] for r in plan['runs'] if 'train_fraction' in r}; assert tf == {.5}
want = {r['split_sha256'] for r in plan['runs'] if 'train_fraction' in r}
a, b, y, ysh, s = ind.regenerate_split(31, 31000, 32000, .5)
f2 = load(ROOT/'research/data/e1f2-split-tf50.npz')
out['tf50_independent_equals_frozen'] = all(np.array_equal(u, f2[k]) for u, k in
                                            zip((a, b, y, *s), ('a', 'b', 'yadd', 'train', 'cal', 'audit')))
out['tf50_independent_equals_shared'] = same((a, b, y, ysh, s), new.data(31, 31000, 32000, .5))
out['tf50_hash'] = ind.hash_arrays(a, b, y, *s)
out['tf50_hash_matches_plan'] = want == {out['tf50_hash']}
out['tf50_sizes'] = [len(x) for x in s]
# Same data seed and permutation, so the two splits are nested.
S1, S2 = [set(x.tolist()) for x in (e1['train'], e1['cal'], e1['audit'])], [set(x.tolist()) for x in s]
out['nesting'] = dict(tf50_train_contains_e1_train_and_cal=S1[0] | S1[1] <= S2[0],
                      tf50_train_extra_from_e1_audit=len(S2[0] & S1[2]),
                      tf50_cal_within_e1_audit=S2[1] <= S1[2], tf50_audit_within_e1_audit=S2[2] <= S1[2])
present = np.zeros((31, 31), bool)
tr = s[0]; present[a[tr], y[tr]] = True; present[b[tr], y[tr]] = True
held = np.concatenate(s[1:]); pres = present[a[held]] | present[b[held]]
n_abs = 31 - pres.sum(1)
out['tf50_label_absence'] = dict(heldout_pairs=len(held), true_label_present=int(pres[np.arange(len(held)), y[held]].sum()),
                                 absent_labels_per_pair=(int(n_abs.min()), float(np.median(n_abs)), int(n_abs.max())))
# The CLI bound .8 still leaves every split non-empty at p = 31 (16 groups per residue).
out['tf80_sizes'] = [len(x) for x in new.data(31, 31000, 32000, .8)[4]]
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
