#!/usr/bin/env python3
"""One table of every planned run: plan, last registry event, published location.

uv run --locked python research/coordinator/registry_table.py > research/coordinator/RUNS.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    planned = []
    for name in ('plan.json', 'plan-followups.json'):
        plan = json.loads((ROOT/'research'/name).read_text())
        planned += [(name, {**plan['common'], **r}) for r in plan['runs']]
    events = {}
    for path in sorted((ROOT/'research').glob('*/registry.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                events.setdefault(e['run_id'], []).append((path.parent.name, e))
    published = {p.parent.name: p.parent.relative_to(ROOT) for p in (ROOT/'research').glob('*/runs/*/manifest.json')}
    print('| run ID | plan | task | wd | width | init | train share | steps | status | completed | events | session | published |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for plan_name, r in planned:
        evs = events.get(r['run_id'], [])
        last = evs[-1][1] if evs else {}
        print(f"| {r['run_id']} | {plan_name} | {r['task']} | {r['wd']} | {r['width']} | {r['init_seed']} | "
              f"{r.get('train_fraction', .3)} | {r['steps']} | {last.get('status', 'planned')} | "
              f"{last.get('completed_steps', '')} | {len(evs)} ({evs[0][0] if evs else ''}) | "
              f"{last.get('session_label', '')} | {published.get(r['run_id'], 'not published')} |")
    extra = sorted(set(events) - {r['run_id'] for _, r in planned})
    if extra:
        print(f'\nRegistry events without a plan entry: {extra}')


if __name__ == '__main__':
    main()
