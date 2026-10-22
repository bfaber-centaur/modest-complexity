#!/usr/bin/env python3
"""Run a lane's planned jobs as complete subprocesses and log their status.

The regime counterpart of launch_runs.py, which is tied to experiment.py. A plan
is {"runs": [{"run_id": ..., "argv": [script, args...], "timeout": seconds}]};
"{out}" in argv becomes ROOT/RUN_ID. The registry is append-only JSON lines
with a 'running' event and one terminal event (complete, failed or capped) per
run. A run ID is never reused; a job killed at its timeout is 'capped', not a
scientific failure, and is kept. There is no automatic resume.

uv run --locked python research/regimes/common/run_jobs.py --plan research/regimes/LANE/plan.json \
    --root runs/regimes --registry research/regimes/LANE/registry.jsonl --session-label LABEL --ids ...
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from instrumentation import THREAD_VARS  # noqa: E402
from launch_runs import read_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True, help='Local output root outside git; one directory per run ID')
    ap.add_argument('--registry', type=Path, required=True, help="The lane's append-only registry")
    ap.add_argument('--session-label', required=True, help='Host/session label, e.g. r1-comp-cloud-1')
    ap.add_argument('--ids', nargs='+', required=True, help='Run IDs from the plan, in execution order')
    ap.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = ap.parse_args()
    runs = {r['run_id']: r for r in json.loads(args.plan.read_text())['runs']}
    unknown = [i for i in args.ids if i not in runs]
    if unknown:
        ap.error(f'not in plan: {unknown}')
    if len(set(args.ids)) != len(args.ids):
        ap.error('duplicate run IDs requested')
    used = {e['run_id'] for e in read_events(args.registry)}
    for i in args.ids:
        if i in used or (args.root/i).exists():
            ap.error(f'{i} already has a registry entry or output directory; never reuse a run ID')
    args.root.mkdir(parents=True, exist_ok=True)
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({k: '1' for k in THREAD_VARS})
    # Keep freed arrays in the glibc heap instead of mmap/munmap per temporary;
    # on the calibration host this sped small full-batch updates up about 1.6x.
    # Allocation only: the arithmetic and results are unchanged.
    env.update(MALLOC_MMAP_THRESHOLD_='4294967296', MALLOC_TRIM_THRESHOLD_='4294967296')
    lock = threading.Lock()

    def log(event):
        event = dict(time=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     host=platform.node(), session_label=args.session_label, **event)
        with lock, args.registry.open('a') as stream:
            stream.write(json.dumps(event)+'\n')

    def job(run_id):
        out = args.root/run_id
        cmd = [sys.executable] + [a.replace('{out}', str(out)) for a in runs[run_id]['argv']]
        timeout = runs[run_id]['timeout']
        log(dict(run_id=run_id, status='running', command=cmd, timeout=timeout))
        start = time.perf_counter()
        with (args.root/f'{run_id}.log').open('w') as stream:
            try:
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=timeout).returncode
                status, reason = ('complete', None) if code == 0 else ('failed', f'exit code {code}')
            except subprocess.TimeoutExpired:
                code, status, reason = None, 'capped', f'resource timeout {timeout:g} s'
        log(dict(run_id=run_id, status=status, reason=reason, returncode=code,
                 wall_seconds_including_startup=round(time.perf_counter()-start, 3)))
        return run_id, status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for run_id, status in pool.map(job, args.ids):
            print(run_id, status, flush=True)


if __name__ == '__main__':
    main()
