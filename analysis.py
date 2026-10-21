"""HISTORICAL protocol-v1 analysis (composite H1/H2/H3 verdict). Not part of the
exploratory reporting path; see research/SWARM.md. Run only after all 32 main jobs complete."""
import argparse,json
from pathlib import Path
import numpy as np


def readout(row):
    # Renamed from 'transfer' (readout accessibility, not compositional reuse).
    return row['readout_accessibility'] if 'readout_accessibility' in row else row['transfer']


def read_run(path):
    cfg=json.loads((path/'config.json').read_text())
    rows=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()]
    if rows[-1]['step']!=50000: raise ValueError(f'Incomplete run: {path}')
    return cfg,rows


def times(rows):
    tm=next((r['step'] for r in rows if r['train']>=.99),None)
    tg=None
    for i in range(len(rows)-2):
        rr=rows[i:i+3]
        if all(r['audit']>=.95 for r in rr) and all(rr[j+1]['step']-rr[j]['step']==100 for j in [0,1]):
            tg=rr[0]['step']; break
    scheduled=[r for r in rows if r['step']%1000==0 and 'intervention' in r]
    tr=None
    for x,y in zip(scheduled,scheduled[1:]):
        if y['step']-x['step']==1000 and x['intervention']['retained']>=.90 and y['intervention']['retained']>=.90 and x['audit']<.50:
            tr=x['step']; break
    return dict(interpolation=tm,generalization=tg,
                grokking=tg if tg is not None and tm is not None and tg-tm>=1000 else None,
                pre_grok_retention=tr)


def valid(c):
    return c['audit']['agreement']>=.99 and c['audit']['relative_mse']<=.01


def wilson(k,n):
    z=1.959963984540054; ph=k/n; den=1+z*z/n
    mid=(ph+z*z/(2*n))/den
    half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
    return [float(mid-half),float(mid+half)]


def summarize(root):
    records=[]; runs={}
    for seed in range(2000,2008):
        for task in ['add','random']:
            for wd in [0,1]:
                path=root/f'{seed}_{task}_wd{wd}'
                cfg,rows=read_run(path)
                assert cfg['seed']==seed and cfg['task']==task and cfg['wd']==wd
                runs[seed,task,wd]=rows
        ad=runs[seed,'add',1][-1]; ra=runs[seed,'random',1][-1]
        eligible=ad['train']>=.99 and ra['train']>=.99
        codec_valid=valid(ad['compression']) and valid(ra['compression'])
        ratio=ad['compression']['bits']/ra['compression']['bits']
        d=ad['intervention']['selective_damage']; r=readout(ad).get('gain')
        h1=eligible and codec_valid and ratio<=.8
        h2=d>=.1 and ad['intervention']['retained']>=.95
        h3=r is not None and r>=.2 and readout(ad)['trained']>=.8
        row=dict(seed=seed,interpolated_pair=eligible,codec_audit_valid=codec_valid,
                 complexity_ratio=ratio,log2_coding_advantage=float(-np.log2(ratio)) if eligible and codec_valid else None,
                 selective_damage=d,reuse_gain=r,final_accuracy=ad['audit'],H1=bool(h1),H2=bool(h2),H3=bool(h3),
                 joint_success=bool(h1 and h2 and h3 and ad['audit']>=.95),times=times(runs[seed,'add',1]))
        row['decay_accuracy_effect']=ad['audit']-runs[seed,'add',0][-1]['audit']
        row['target_by_decay_accuracy_interaction']=(ad['audit']-ra['audit'])-(runs[seed,'add',0][-1]['audit']-runs[seed,'random',0][-1]['audit'])
        row['arms']={}
        for task in ['add','random']:
            for wd in [0,1]:
                rr=runs[seed,task,wd]; final=rr[-1]
                interp=next((z for z in rr if z['train']>=.99 and 'compression' in z),None)
                row['arms'][f'{task}_wd{wd}']=dict(final_accuracy=final['audit'],train=final['train'],
                    complexity=final['compression'],reuse=readout(final),intervention=final['intervention'],
                    complexity_change=final['compression']['bits']/interp['compression']['bits'] if interp else None,
                    times=times(rr),seconds=final['seconds'])
        records.append(row)
    rng=np.random.default_rng(20261022); draws=rng.integers(0,8,(10000,8))
    endpoints={}
    for key in ['log2_coding_advantage','selective_damage','reuse_gain','final_accuracy','decay_accuracy_effect','target_by_decay_accuracy_interaction']:
        vals=np.array([r[key] if r[key] is not None else np.nan for r in records])
        finite=np.isfinite(vals)
        if not finite.any(): endpoints[key]={'n':0}; continue
        samples=vals[draws]; count=np.isfinite(samples).sum(1)
        means=np.nansum(samples,axis=1)[count>0]/count[count>0]
        endpoints[key]=dict(n=int(finite.sum()),mean=float(np.nanmean(vals)),
                            percentile_interval=np.quantile(means,[.025,.975]).tolist())
    k=sum(r['joint_success'] for r in records); eligible=sum(r['interpolated_pair'] for r in records)
    decision='supporting pattern' if k>=6 else ('contrary pattern' if k<=2 and eligible>=6 else 'mixed or unresolved')
    return dict(blocks=records,endpoints=endpoints,successes=k,n=8,wilson95=wilson(k,8),decision=decision)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); args=ap.parse_args()
    result=summarize(args.root)
    path=args.root/'summary.json'; path.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:result[k] for k in ['successes','n','wilson95','decision']}))
