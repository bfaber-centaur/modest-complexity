#!/usr/bin/env python3
"""Follow-up 1 (width 512, seed 30000, 200,000 updates) against the map run it
extends: bitwise identity up to 50,000 at every checkpoint both publish and in every
logged metric, provenance, independent recomputation at every published checkpoint,
independent replays of 2,000 updates from 50k, 100k and 150k against the logged rows,
and the coordinator's INDEX observations rederived from metrics.jsonl.

uv run --locked python research/audit/check_e1f1.py FOLLOWUP_DIR BASE_DIR CODE_COMMIT OUT.json
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import audit_preview as ap  # noqa: E402

fd, bd, commit, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
rows = lambda d: [json.loads(s) for s in (d/'metrics.jsonl').read_text().splitlines()]
cfg = lambda d: json.loads((d/'config.json').read_text())
fr_rows, base_rows = rows(fd), rows(bd)
fc, bc = cfg(fd), cfg(bd)
plan = json.loads(Path('research/plan-followups.json').read_text())
pl = {**plan['common'], **next(r for r in plan['runs'] if r['run_id'] == fd.name)}
report = {}

man = json.loads((fd/'manifest.json').read_text())
local = {f.name for f in fd.iterdir() if f.is_file() and f.name != 'manifest.json'}
report['manifest'] = dict(hashes_match=all(man['files'][n]['sha256'] == ap.sha(fd/n) for n in local),
                          flags_match=local == {n for n, f in man['files'].items() if f['published']},
                          n_listed=len(man['files']), n_published=len(local))
src = {n: hashlib.sha256(subprocess.run(['git', 'show', f'{commit}:{n}'], capture_output=True, check=True).stdout).hexdigest()
       for n in fc['runtime']['source_sha256']}
report['config'] = dict(
    keys_differing_from_base=sorted(k for k in set(fc) | set(bc) if k not in ('runtime',) and fc.get(k) != bc.get(k)),
    steps_width_wd_task=(fc['steps'], fc['hidden'], fc['wd'], fc['task']) == (pl['steps'], pl['width'], float(pl['wd']), pl['task']),
    seeds_match_plan=fc['seeds'] == dict(data_seed=pl['data_seed'], init_seed=pl['init_seed'],
                                         control_seed=pl['control_seed'], diagnostic_seed=pl['diagnostic_seed']),
    split_and_target_hash_match_plan=(fc['data_hashes']['split_sha256'], fc['data_hashes']['target_sha256']) ==
                                     (pl['split_sha256'], plan['targets']['add']),
    git_commit=fc['runtime']['git_commit'], source_hashes_match_commit=fc['runtime']['source_sha256'] == src,
    experiment_py_same_as_base=fc['runtime']['source_sha256']['experiment.py'] == bc['runtime']['source_sha256']['experiment.py'],
    environment_same_as_base=fc['runtime']['environment_sha256'] == bc['runtime']['environment_sha256'],
    host_python_numpy=[(c['runtime'].get('host'), c['runtime']['python'], c['runtime']['numpy']) for c in (fc, bc)])

# Bitwise identity up to 50,000.
shared = sorted({p.name for p in fd.glob('checkpoint_*.npz')} & {p.name for p in bd.glob('checkpoint_*.npz')})
same_ck = {}
for n in shared:
    x, y = ap.load_ck(fd/n), ap.load_ck(bd/n)
    same_ck[n] = x.keys() == y.keys() and all(np.array_equal(x[k], y[k]) and x[k].dtype == y[k].dtype for k in x)
strip = lambda z: {k: v for k, v in z.items() if k not in ('seconds', 'elapsed_after_diagnostics')}  # wall-clock fields
early = [z for z in fr_rows if z['step'] <= base_rows[-1]['step']]
report['bitwise_to_50k'] = dict(shared_published_checkpoints=same_ck, logged_rows_compared=len(early),
                                logged_rows_identical_except_wall_clock=len(early) == len(base_rows) and
                                all(strip(u) == strip(v) for u, v in zip(early, base_rows)),
                                data_npz_identical=ap.sha(fd/'data.npz') == ap.sha(bd/'data.npz'))

# Independent recomputation and replays.
c = plan['common']; p = c['p']
e1 = json.loads(Path('research/plan.json').read_text())
_, fr = ap.check_split({**e1['common'], 'targets': e1['targets']}, Path('research/data/e1-split.npz'))
by = {z['step']: z for z in fr_rows}
rc = ap.recompute_checkpoints(fd, fr, by, p, fr['yadd'])
report['recompute'] = dict(steps=sorted(rc), max_acc_diff=max(rc[s][sp]['diff']['acc'] for s in rc for sp in ('train', 'cal', 'audit')),
                           max_rel_ce_or_margin_diff=max(v for s in rc for sp in ('train', 'cal', 'audit')
                                                         for k, v in rc[s][sp]['diff'].items() if k != 'acc'),
                           max_norm_rel_diff=max(rc[s]['norm_rel_diff'] for s in rc),
                           checkpoints_ok=all(rc[s]['finite'] and rc[s]['v_nonnegative'] and rc[s]['file_step_matches'] for s in rc))
report['replays'] = []
for s0 in (50000, 100000, 150000):
    _, worst, _ = ap.replay(ap.load_ck(fd/f'checkpoint_{s0:06d}.npz'), s0, s0 + 2000, fr, by, p, c['lr'], float(pl['wd']), fr['yadd'])
    report['replays'].append(dict(start=s0, stop=s0 + 2000, logged_rows=worst))

# INDEX observations, rederived.
late = [z for z in fr_rows if z['step'] >= 1000]
ce = lambda z, s: z['objective'][s]['cross_entropy']
peak = max(fr_rows, key=lambda z: ce(z, 'audit'))
from50 = [z for z in fr_rows if z['step'] >= 50000]
report['index_observations'] = dict(
    max_heldout_acc_from_1000=max(max(z['cal'], z['audit']) for z in late),
    audit_ce_peak=(peak['step'], ce(peak, 'audit')), audit_ce_final=ce(fr_rows[-1], 'audit'),
    audit_ce_at=[(s, round(ce(by[s], 'audit'), 3)) for s in (50000, 100000, 150000, 200000)],
    train_ce_range_from_50k=(min(ce(z, 'train') for z in from50), max(ce(z, 'train') for z in from50)),
    norm_range_from_50k=(min(z['norm'] for z in from50), max(z['norm'] for z in from50)),
    train_margin_min_at=[(s, round(by[s]['objective']['train']['margin_min'], 3)) for s in (50000, 100000, 150000, 200000)],
    max_cosine_from_50k=max(z['optimizer_update']['cosine'] for z in from50))
out.write_text(json.dumps(report, indent=1, default=float) + '\n')
print(json.dumps(report, indent=1, default=float))
