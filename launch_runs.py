#!/usr/bin/env python3
"""Run planned exploratory jobs as complete subprocesses and log their status.

The registry is append-only JSON lines: one 'running' event when a job starts and
one terminal event (complete, failed or capped) when it stops. A run ID is never
reused; a job killed at the timeout is 'capped', not a scientific failure.
Automatic resume is not implemented.
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
from instrumentation import THREAD_VARS

TERMINAL = ('complete', 'failed', 'capped')


def read_events(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def command(plan, run, out, session_label):
    c = {**plan['common'], **run}
    return [sys.executable, str(Path(__file__).with_name('experiment.py')),
            '--out', str(out), '--run-id', run['run_id'], '--purpose', c['purpose'],
            '--session-label', session_label, '--task', c['task'], '--wd', str(c['wd']),
            '--hidden', str(c['width']), '--p', str(c['p']), '--steps', str(c['steps']),
            '--lr', str(c['lr']), '--data-seed', str(c['data_seed']), '--init-seed', str(c['init_seed']),
            '--control-seed', str(c['control_seed']), '--diagnostic-seed', str(c['diagnostic_seed']),
            '--log-every', str(c['log_every']), '--checkpoint-every', str(c['checkpoint_every']),
            '--diagnostics', c['diagnostics'], '--expect-split-sha256', c['split_sha256']] + (
            ['--train-fraction', str(c['train_fraction'])] if 'train_fraction' in c else []) + (
            ['--init-weights', c['init_weights'], '--expect-init-sha256', c['init_weights_sha256'],
             '--init-age', str(c['init_age'])] if 'init_weights' in c else []) + (
            ['--checkpoint-steps', c['checkpoint_steps']] if 'checkpoint_steps' in c else []) + (
            ['--decay-projection', c['decay_projection'], '--expect-decay-projection-sha256', c['decay_projection_sha256'],
             '--decay-on', c['decay_on']] if 'decay_projection' in c else []) + (
            ['--resume', c['resume'], '--expect-init-sha256', c['resume_sha256']] if 'resume' in c else [])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, default=Path('research/plan.json'))
    ap.add_argument('--root', type=Path, required=True, help='Local output root; one directory per run ID')
    ap.add_argument('--registry', type=Path, required=True, help="This lane's append-only registry")
    ap.add_argument('--session-label', required=True, help='Host/session label, e.g. co-cloud-1')
    ap.add_argument('--ids', nargs='+', required=True, help='Run IDs from the plan, in execution order')
    ap.add_argument('--workers', type=int, choices=(1, 2), default=2)
    ap.add_argument('--timeout', type=float, required=True, help='Seconds per job before it is capped')
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    runs = {r['run_id']: r for r in plan['runs']}
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
    lock = threading.Lock()

    def log(event):
        event = dict(time=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     host=platform.node(), session_label=args.session_label, **event)
        with lock, args.registry.open('a') as stream:
            stream.write(json.dumps(event)+'\n')

    def job(run_id):
        out = args.root/run_id
        cmd = command(plan, runs[run_id], out, args.session_label)
        log(dict(run_id=run_id, status='running', command=cmd))
        start = time.perf_counter()
        with (args.root/f'{run_id}.log').open('w') as stream:
            try:
                code = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      timeout=args.timeout).returncode
                status, reason = ('complete', None) if code == 0 else ('failed', f'exit code {code}')
            except subprocess.TimeoutExpired:
                code, status, reason = None, 'capped', f'resource timeout {args.timeout:g} s'
        runtime_path = out/'runtime.json'
        runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
        log(dict(run_id=run_id, status=status, reason=reason, returncode=code,
                 completed_steps=runtime.get('completed_steps'), planned_steps={**plan['common'], **runs[run_id]}['steps'],
                 wall_seconds_including_startup=round(time.perf_counter()-start, 3)))
        return run_id, status

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for run_id, status in pool.map(job, args.ids):
            print(run_id, status, flush=True)


if __name__ == '__main__':
    main()
