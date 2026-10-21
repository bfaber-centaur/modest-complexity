"""Sequential launcher. Main stage requires the preregistered pilot gate."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from analysis import read_run,times

ap=argparse.ArgumentParser()
ap.add_argument('stage',choices=['pilot','main'])
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--pilot-root',type=Path)
ap.add_argument('--execute',action='store_true',help='Otherwise only write the fixed run schedule')
args=ap.parse_args()
if args.stage=='main':
    if args.pilot_root is None: ap.error('main requires --pilot-root')
    pts=[times(read_run(args.pilot_root/f'{s}_add_wd1')[1]) for s in [1000,1001]]
    early=[]
    for s,t in zip([1000,1001],pts):
        rows=read_run(args.pilot_root/f'{s}_add_wd1')[1]
        early.append(any(r['step']==t['interpolation'] and r['audit']<.20 for r in rows))
    gate=all(t['interpolation'] is not None for t in pts) and any(t['grokking'] is not None and e for t,e in zip(pts,early))
    if not gate: raise SystemExit('Pilot gate failed: stop and revise the protocol before further experiments.')
jobs=([(s,'add',1) for s in [1000,1001]] if args.stage=='pilot' else
      [(s,t,w) for s in range(2000,2008) for t in ['add','random'] for w in [0,1]])
np.random.default_rng(20261020).shuffle(jobs)
args.out.mkdir(parents=True,exist_ok=True)
commands=[[sys.executable,str(Path(__file__).with_name('experiment.py')),'--seed',str(s),'--task',t,'--wd',str(w),
           '--out',str(args.out/f'{s}_{t}_wd{w}')] for s,t,w in jobs]
schedule=args.out/'schedule.json'
if schedule.exists() and json.loads(schedule.read_text())!=commands: raise SystemExit('Existing schedule differs')
schedule.write_text(json.dumps(commands,indent=2))
print(f'Saved {len(jobs)} ordered jobs to {schedule}',flush=True)
if args.execute:
    for cmd in commands: subprocess.run(cmd,check=True)
else: print('Schedule only. Add --execute to run. No experiments launched.')
