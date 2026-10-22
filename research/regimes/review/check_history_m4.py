"""R3 M4 checks from published artifacts (REVIEW.md §3): follow-up plan and
prediction precede its runs; every branch started from its seed's ckA; and
relearning times re-derived from metrics with one criterion and one grid.

uv run --locked python research/regimes/review/check_history_m4.py LANE_COMMIT PREDICTION_COMMIT
"""
import gzip
from pathlib import Path
import json
import subprocess
import sys

from lanes import check, FAIL, ROOT

commit, pred = sys.argv[1:3]
git = lambda *a: subprocess.check_output(['git', '-C', str(ROOT), *a])
ls = git('ls-tree', '-r', '--name-only', commit, 'research/regimes/history/runs/').decode().split()
runs = sorted({p.split('/')[4] for p in ls if p.count('/') >= 5})
reg = [json.loads(l) for l in git('show', f'{commit}:research/regimes/history/registry.jsonl').splitlines() if l.strip()]
t_pred = git('show', '-s', '--format=%cI', pred).decode().strip()
fu_start = min(e['time'] for e in reg if e['run_id'].startswith('r1-hist-fu-') and e['status'] == 'running')
plan_pred = json.loads(git('show', f'{pred}:research/regimes/history/plan.json'))
fu_ids = {e['run_id'] for e in reg if e['run_id'].startswith('r1-hist-fu-')}
check(f'follow-up prediction commit {pred} ({t_pred}) precedes the first follow-up launch ({fu_start}); '
      f'all {len(fu_ids)} follow-up IDs already in its plan', t_pred < fu_start and fu_ids <= {r['run_id'] for r in plan_pred['runs']})
status = {}
for e in reg:
    status[e['run_id']] = e['status']
print('INFO registry terminal states:', {s: sum(1 for v in status.values() if v == s) for s in set(status.values())})
cfg = {r: json.loads(git('show', f'{commit}:research/regimes/history/runs/{r}/config.json')) for r in runs}
man = {}
for line in git('show', f'{commit}:research/regimes/history/runs/MANIFEST.jsonl').splitlines():
    m = json.loads(line); man[m['run_id'], m['file']] = m['sha256']
started = [r for r in runs if cfg[r]['source']]
ok = []
for r in started:
    src_run, src_file = Path(cfg[r]['source']).parts[-2:]
    ok.append(cfg[r]['source_file_sha256'] == man[src_run, src_file] and src_run[-8:-3] == r[-8:-3])
ck_branch = [r for r in started if Path(cfg[r]['source']).parts[-2].startswith('r1-hist-A-')]
check(f'all {len(started)} checkpoint-started runs record the MANIFEST SHA-256 of their source file, from their own seed '
      f'({sum(ok)} ok; {len(ck_branch)} start from ckA, the rest are follow-ups from end-of-B or scratch-B checkpoints)', all(ok))


def first_hit(run, phase):
    rows = [json.loads(l) for l in gzip.decompress(git('show', f'{commit}:research/regimes/history/runs/{run}/metrics.jsonl.gz')).splitlines()]
    hit = [r['step'] for r in rows if r['phase'] == phase and r['A_test_acc'] >= .95]
    return min(hit) if hit else None


fresh = {s: first_hit(f'r1-hist-A-s{s}-a1', 1) for s in ('53200', '53201', '53202')}
print('INFO fresh-A updates to held-out A >= 0.95 (phase A):', fresh)
for arm in ('rel', 'unrel'):
    for d in (1000, 10000):
        kept = [first_hit(f'r1-hist-{arm}-d{d}-s{s}-a1', 2) for s in fresh]
        print(f'INFO {arm} d={d}: return-phase updates to criterion, head kept {kept}')
print('ALL PASS' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
# Follow-up (M3) table in NOTE obs. 3, re-derived with the same criterion and grid.
for name, fmt in (('head reset, end-of-B trunk', 'r1-hist-fu-reset-{a}-d{d}-s{s}-a1'),
                  ('head at init, B-only trunk', 'r1-hist-fu-reset-scratch-B{a}-d{d}-s{s}-a1')):
    for a in ('rel', 'unrel'):
        vals = {d: [first_hit(fmt.format(a=a, d=d, s=s), 1) for s in fresh] for d in (1000, 10000)}
        print(f'INFO {name}, {a}: {vals}')
print('INFO head reset, end-of-A trunk (ckA):', [first_hit(f'r1-hist-fu-reset-ckA-s{s}-a1', 1) for s in fresh])
