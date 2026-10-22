"""QUESTION.md section 6 (synthesis check), reads only: SY4 run counts and wall time from the c3 registries;
post-hoc, after reading the synthesis: the keep control on logged rows and the correction call site at the head."""
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

C3 = Path(__file__).resolve().parents[1]
LANES = ['correction', 'scaffolding', 'review', 'coordinator']


def t(s):
    return datetime.fromisoformat(s).timestamp()


def lane(name):
    rows = [json.loads(line) for line in open(C3 / name / 'registry.jsonl')]
    start, end = {}, {}
    for r in rows:
        if r['status'] == 'running':
            start.setdefault(r['run_id'], t(r['time']))
        else:
            end[r['run_id']] = r
    ids = sorted(set(start) | set(end))
    status = {i: end[i]['status'] if i in end else 'running' for i in ids}
    if any(i not in start for i in ids):
        sys.exit(f'{name}: a run has no running event')
    # union of [start, start + wall] intervals: elapsed time with at least one job running
    spans = sorted((start[i], start[i] + end[i]['wall_seconds_including_startup']) for i in ids if i in end)
    union, cur = 0.0, None
    for a, b in spans:
        if cur is None or a > cur[1]:
            if cur:
                union += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        union += cur[1] - cur[0]
    total = sum(end[i]['wall_seconds_including_startup'] for i in ids if i in end)
    return ids, status, union, total


def keep_control():
    """Post-hoc, after reading the synthesis: A'-A and B'-B ranges on the logged rows."""
    rows = list(csv.DictReader(open(C3 / 'scaffolding' / 'figures' / 'scaf_followup.csv')))
    val = {(r['arm'], r['seed'], int(r['step'])): r for r in rows}
    for lo, hi in ((1050, 3000), (3050, 7950), (8000, 50000), (10000, 50000)):
        for m in ('held_final_acc', 'held_notN_acc'):
            out = []
            for arm, ref in (('keep-reset', 'keep'), ('drop-reset', 'drop')):
                d = [float(r[m]) - float(val[(ref, s, k)][m]) for (a, s, k), r in val.items()
                     if a == arm and lo <= k <= hi and (ref, s, k) in val]
                out.append(f'{arm} - {ref} [{min(d):+.3f}, {max(d):+.3f}] (n {len(d)})')
            print(f'  logged {lo}-{hi} {m}: ' + '; '.join(out))


def call_site():
    """The synthesis's 'Not independently checked' item: corr_core.load_base at the integrated head (5b92e1b)."""
    sys.path.insert(0, str(C3 / 'correction'))
    import corr_core
    import nets
    b = corr_core.bases('original')[0]
    p, st, meta = nets.load(b['path'])
    try:  # the pre-fix call (be41d16 signature) against the re-pinned check_source
        corr_core.c3train.check_source(b['path'], b['sha256'], b['config'], meta, st['t'], *corr_core.expected_recipe(b))
        print('  old call accepted (unexpected)')
    except TypeError as e:
        print(f'  old call at head raises TypeError: {e}')
    recorded = {}
    for run in ('c3-corr-main-a1', 'c3-corr-f2-main-a1'):
        for r in json.load(open(C3 / 'correction' / 'runs' / run / 'checks.json'))['bases']:
            recorded[r['run_id']] = r
    bad = []
    for b in corr_core.bases('original') + corr_core.bases('f2'):
        _, rec = corr_core.load_base(b)
        bad += [(b['run_id'], k) for k, v in rec.items() if recorded[b['run_id']].get(k) != v]
    print(f'  load_base at head on {len(recorded)} bases: fields differing from the recorded checks.json: {bad}')
    r = subprocess.run([sys.executable, str(C3 / 'correction' / 'test_corr.py')], capture_output=True, text=True)
    print(f'  test_corr.py exit {r.returncode}: {r.stderr.strip().splitlines()[-1]}')


def main():
    print('Post-hoc keep control (logged rows, every 50 updates):')
    keep_control()
    print('Correction call site at the integrated head:')
    call_site()
    for name in LANES:
        ids, status, union, total = lane(name)
        bad = {i: s for i, s in status.items() if s != 'complete'}
        print(f'{name}: {len(ids)} runs, not complete {bad}; '
              f'union wall {union / 60:.2f} min, summed wall {total / 60:.2f} min')
        for i in ids:
            print(f'    {status[i]:9s} {i}')


if __name__ == '__main__':
    main()
