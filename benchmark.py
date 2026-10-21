#!/usr/bin/env python3
"""Bounded calibration with isolated seeds, full diagnostics, and explicit thread limits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from instrumentation import THREAD_VARS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--workers',type=int,choices=(1,2,4),default=1)
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--blas-threads',type=int,default=1)
    ap.add_argument('--timeout',type=float,default=120,
                    help='Hard timeout per child in seconds; retains partial artifacts')
    args=ap.parse_args()
    if min(args.steps,args.jobs,args.blas_threads,args.timeout)<=0:
        ap.error('steps, jobs, BLAS threads and timeout must be positive')
    args.out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update({key:str(args.blas_threads) for key in THREAD_VARS})
    script=Path(__file__).with_name('experiment.py').resolve()
    manifest=dict(steps=args.steps,workers=args.workers,jobs=args.jobs,
                  blas_threads=args.blas_threads,timeout_per_job=args.timeout,
                  purpose='calibration only; never pool with pilot or confirmatory seeds')
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def job(i):
        run=args.out/f'job_{i:03d}'
        cmd=[sys.executable,str(script),'--seed',str(900000+i),'--steps',str(args.steps),
             '--out',str(run)]
        start=time.perf_counter()
        with (args.out/f'job_{i:03d}.log').open('w') as log:
            try:
                result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
                status='complete' if result.returncode==0 else 'failed'
                code=result.returncode
            except subprocess.TimeoutExpired:
                status='timeout'; code=None
        elapsed=time.perf_counter()-start
        runtime=json.loads((run/'runtime.json').read_text()) if (run/'runtime.json').exists() else None
        record=dict(job=i,command=cmd,status=status,returncode=code,
                    wall_seconds_including_startup=elapsed,runtime=runtime)
        # Parent status is authoritative if a timeout killed the child's finally block.
        (args.out/f'job_{i:03d}.json').write_text(json.dumps(record,indent=2))
        return record
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(job,range(args.jobs)))
    wall=time.perf_counter()-start
    complete=all(r['status']=='complete' for r in records)
    summary=dict(**manifest,status='complete' if complete else 'incomplete',wall_seconds=wall,
                 aggregate_updates_per_second=args.steps*args.jobs/wall if complete else None,
                 jobs_detail=records,
                 caveat='Short calibration, not a full-run forecast. Includes step-zero/final diagnostics; repeat on target host.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='jobs_detail'},indent=2))
    return 0 if complete else 1

if __name__=='__main__':
    sys.exit(main())
