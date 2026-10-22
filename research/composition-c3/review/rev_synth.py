"""SY4 of QUESTION.md section 6: run counts and wall time from the c3 registries (reads only)."""
import json
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


def main():
    for name in LANES:
        ids, status, union, total = lane(name)
        bad = {i: s for i, s in status.items() if s != 'complete'}
        print(f'{name}: {len(ids)} runs, not complete {bad}; '
              f'union wall {union / 60:.2f} min, summed wall {total / 60:.2f} min')
        for i in ids:
            print(f'    {status[i]:9s} {i}')


if __name__ == '__main__':
    main()
