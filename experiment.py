#!/usr/bin/env python3
"""NumPy research prototype. No training-data downloads; see PROTOCOL.md."""
import argparse, hashlib, itertools, json, os, platform, time
from pathlib import Path
import numpy as np
from instrumentation import Timings, inventory


def data(p, seed, control_seed=None, train_fraction=.3):
    """Grouped split from `seed`. The shuffled-label table uses `control_seed`
    when given; None keeps the historical single-stream draw. The calibration
    share stays .1; the audit split takes the rest."""
    rng = np.random.default_rng(seed)
    a, b = np.indices((p, p)).reshape(2, -1)
    split_masks = [np.zeros((p,p),bool) for _ in range(3)]
    for c in range(p):
        groups = np.array([(i,j) for i in range(p) for j in range(i,p) if (i+j)%p==c])
        rng.shuffle(groups); n=len(groups)
        nt=max(1,round(train_fraction*n)); nc=max(1,round(.1*n))
        for mask,part in zip(split_masks,np.split(groups,[nt,nt+nc])):
            for i,j in part: mask[i,j]=mask[j,i]=True
    masks = [np.flatnonzero(mask.ravel()) for mask in split_masks]
    if min(map(len, masks)) == 0: raise ValueError('a split is empty; lower --train-fraction or raise p')
    y = (a+b) % p
    # Preserve exact class counts separately in every split. Random control is
    # deliberately noncommutative; no transpose pair ever crosses a split.
    yr = y.copy()
    if control_seed is not None: rng = np.random.default_rng(control_seed)
    for ix in masks: yr[ix] = rng.permutation(y[ix])
    return a, b, y, yr, masks


def array_sha256(*arrays):
    """Order-sensitive hash of dtype, shape and bytes."""
    digest = hashlib.sha256()
    for x in arrays:
        x = np.ascontiguousarray(x)
        digest.update(f'{x.dtype.str}{x.shape}'.encode()); digest.update(x.tobytes())
    return digest.hexdigest()


def init(p, h, seed):
    r = np.random.default_rng(seed)
    return [r.normal(0,.2,(p,h)), r.normal(0,.2,(p,h)),
            r.normal(0,1/np.sqrt(h),(h,p))]


def forward(w,a,b):
    z = w[0][a] + w[1][b]
    h = np.maximum(z,0)
    return h @ w[2], h, z


def lossgrad(w,a,b,y):
    logits,h,z = forward(w,a,b)
    ls = logits-logits.max(1,keepdims=True)
    ex = np.exp(ls); prob = ex/ex.sum(1,keepdims=True)
    loss = np.mean(np.log(ex.sum(1))-ls[np.arange(len(y)),y])
    prob[np.arange(len(y)),y] -= 1
    prob /= len(y)
    gw = h.T@prob
    dz = (prob@w[2].T)*(z>0)
    gu,gv = np.zeros_like(w[0]),np.zeros_like(w[1])
    np.add.at(gu,a,dz); np.add.at(gv,b,dz)
    return float(loss),[gu,gv,gw]


def adam(w,g,m,v,t,lr,wd,measure=False):
    a2=d2=dot=0.
    for i in range(3):
        m[i] *= .9; m[i] += .1*g[i]
        v[i] *= .999; v[i] += .001*g[i]**2
        update = -lr*(m[i]/(1-.9**t))/(np.sqrt(v[i]/(1-.999**t))+1e-8)
        if measure:
            decay = -lr*wd*w[i]
            a2 += float(np.sum(update**2)); d2 += float(np.sum(decay**2))
            dot += float(np.sum(update*decay))
        w[i] *= 1-lr*wd
        w[i] += update
    if measure:
        return dict(adaptive_norm=float(np.sqrt(a2)), decay_norm=float(np.sqrt(d2)),
                    cosine=dot/np.sqrt(a2*d2) if a2*d2 > 0 else None)



def accuracy(logits,y,ix):
    return float(np.mean(logits[ix].argmax(1)==y[ix]))


def fidelity(x,ref,ix):
    x=x[ix]; ref=ref[ix]
    xc=x-x.mean(1,keepdims=True); rc=ref-ref.mean(1,keepdims=True)
    return {"agreement":float(np.mean(x.argmax(1)==ref.argmax(1))),
            "relative_mse":float(np.sum((xc-rc)**2)/max(np.sum(rc**2),1e-30))}


def compress(w,a,b,masks):
    """Fixed no-retraining hidden-unit pruning + scalar quantization code."""
    p,h=w[0].shape
    score=(np.linalg.norm(w[0],axis=0)+np.linalg.norm(w[1],axis=0))*np.linalg.norm(w[2],axis=1)
    order=np.argsort(-score,kind='stable')
    ref=forward(w,a,b)[0]
    candidates=[]
    for k in sorted(set([min(x,h) for x in [8,16,32,64,h]])):
        ix=order[:k]
        base=[w[0][:,ix],w[1][:,ix],w[2][ix]]
        for q in [4,8,16,64]:
            cw=[]
            for x in base:
                scale=max(float(np.abs(x).max()),1e-30)
                levels=2**(q-1)-1 if q!=64 else 1
                cw.append(x.copy() if q==64 else np.rint(x/scale*levels)/levels*scale)
            out=forward(cw,a,b)[0]
            f=[fidelity(out,ref,j) for j in masks[:2]]
            if all(z['agreement']>=.99 and z['relative_mse']<=.01 for z in f):
                # 32-bit p,k; 8-bit q; three float64 scales; fixed decoder.
                bits=32+32+8+3*64+3*p*k*q
                candidates.append((bits,k,q,out,f))
    bits,k,q,out,f=min(candidates,key=lambda z:(z[0],z[1],z[2]))
    return dict(bits=int(bits),k=int(k),q=int(q),train=f[0],calibration=f[1],
                audit=fidelity(out,ref,masks[2]),raw_bits=264+3*p*h*64)


def interventions(w,a,b,y,ix,seed,nsham=32):
    p=w[0].shape[0]; K=min(4,(p-1)//2)
    fs=[np.fft.rfft(x,axis=0) for x in w[:2]]
    energy=sum(np.sum(np.abs(f)**2,axis=1) for f in fs)
    chosen=np.argsort(-energy[1:],kind='stable')[:K]+1
    delta=[]; retained=[]
    for x,f in zip(w[:2],fs):
        g=np.zeros_like(f); g[chosen]=f[chosen]
        d=np.fft.irfft(g,n=p,axis=0)
        delta.append(d); retained.append(d+x.mean(0,keepdims=True))
    base=accuracy(forward(w,a,b)[0],y,ix)
    removed=accuracy(forward([w[0]-delta[0],w[1]-delta[1],w[2]],a,b)[0],y,ix)
    kept=accuracy(forward(retained+[w[2]],a,b)[0],y,ix)
    rng=np.random.default_rng(seed)
    sham=[]
    for _ in range(nsham):
        # Permuting rows preserves delta rank, singular values, mean and norm.
        ds=[d[rng.permutation(p)] for d in delta]
        sham.append(accuracy(forward([w[0]-ds[0],w[1]-ds[1],w[2]],a,b)[0],y,ix))
    return dict(frequencies=chosen.tolist(),baseline=base,retained=kept,removed=removed,
                sham_accuracy=sham,selective_damage=float(np.median(sham)-removed),
                delta_norm=[float(np.linalg.norm(d)) for d in delta])


def transfer(w,initial,a,b,yadd,masks,seed):
    """Frozen representation, fixed-budget ridge readout; no target test tuning."""
    rng=np.random.default_rng(seed); p=w[0].shape[0]
    labels=rng.permutation(p)[yadd]
    chosen=[]
    for c in range(p):
        possible=masks[0][yadd[masks[0]]==c]
        if len(possible)<2: return {"missing_class":int(c)}
        chosen.extend(rng.choice(possible,2,replace=False).tolist())
    ix=np.array(chosen); audit=masks[2]
    h=forward(w,a,b)[1]; hi=forward(initial,a,b)[1]
    # A row shuffle preserves the entire activation matrix's singular spectrum.
    shuffled=h[rng.permutation(len(h))]
    answers={}
    for name,features in [('trained',h),('initial',hi),('row_shuffled',shuffled)]:
        mu=features[ix].mean(0); sd=features[ix].std(0)
        x=(features-mu)/np.maximum(sd,1e-6)
        x=np.column_stack([x,np.ones(len(x))])
        targets=np.eye(p)[labels[ix]]
        # Dual ridge, fixed lambda=1; same samples and budget in all arms.
        head=x[ix].T@np.linalg.solve(x[ix]@x[ix].T+np.eye(len(ix)),targets)
        pred=x@head
        answers[name]=accuracy(pred,labels,audit)
    answers['gain']=answers['trained']-max(answers['initial'],answers['row_shuffled'])
    answers['target_samples']=len(ix)
    return answers


def objective_metrics(logits, y, ix):
    x=logits[ix]; labels=y[ix]; rows=np.arange(len(ix))
    centered=x-x.max(1,keepdims=True)
    ce=np.mean(np.log(np.exp(centered).sum(1))-centered[rows,labels])
    other=x.copy(); other[rows,labels]=-np.inf
    margins=x[rows,labels]-other.max(1)
    return dict(cross_entropy=float(ce), margin_min=float(margins.min()),
                margin_median=float(np.median(margins)),
                margin_p05=float(np.quantile(margins,.05)))


def resolve_seeds(args):
    """Explicit seeds override the historical derivation from --seed, where one
    seed set both the split and (offset) the initialization."""
    s=args.seed
    return dict(data_seed=s if args.data_seed is None else args.data_seed,
                init_seed=s+10000 if args.init_seed is None else args.init_seed,
                control_seed=args.control_seed,
                diagnostic_seed=s+30000 if args.diagnostic_seed is None else args.diagnostic_seed)


def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    timing=Timings(); status='failed'; completed=0; first_interp=None
    try:
        with timing.measure('setup'):
            seeds=resolve_seeds(args)
            a,b,y,yr,masks=data(args.p,seeds['data_seed'],seeds['control_seed'],args.train_fraction)
            target=y if args.task=='add' else yr
            # The split hash covers inputs, addition labels and split membership,
            # so it is identical for both tasks; the target hash identifies labels.
            hashes=dict(split_sha256=array_sha256(a,b,y,*masks),target_sha256=array_sha256(target))
            start_sha=hashlib.sha256(Path(args.init_weights).read_bytes()).hexdigest() if args.init_weights else None
            cfg=vars(args).copy()
            cfg.update(seeds=seeds,data_hashes=hashes,host=platform.node(),runtime=inventory(out),init_weights_sha256=start_sha,
                       init_rule=('U, V, W from --init-weights (init_seed unused); Adam moments zero, update count restarts at 1'
                                  if args.init_weights else 'init(p, hidden, init_seed)'),
                       purpose=args.purpose or ('calibration' if args.seed>=900000 else 'research-prototype'),
                       diagnostic_schedule=dict(metrics_every=args.log_every,checkpoint_every=args.checkpoint_every,
                           checkpoint_extra='last step, first train accuracy >= .99 and --checkpoint-steps',checkpoint_steps=args.checkpoint_steps,
                           inline=args.diagnostics, inline_every=args.audit_every if args.diagnostics=='inline' else None))
            (out/'config.json').write_text(json.dumps(cfg,indent=2))
            if args.expect_split_sha256 and hashes['split_sha256']!=args.expect_split_sha256:
                raise ValueError(f"split hash {hashes['split_sha256']} != expected {args.expect_split_sha256}")
            if args.init_weights:
                # A continuation from given (e.g. edited) weights restarts the optimizer: its
                # checkpoint age lives in the plan, and every step here is a new update.
                if args.expect_init_sha256 and start_sha!=args.expect_init_sha256:
                    raise ValueError(f'init weights hash {start_sha} != expected {args.expect_init_sha256}')
                with np.load(args.init_weights) as z: w=[z[k] for k in 'UVW']
                shapes=[(args.p,args.hidden),(args.p,args.hidden),(args.hidden,args.p)]
                if [x.shape for x in w]!=shapes or any(x.dtype!=np.float64 for x in w):
                    raise ValueError(f'init weights must be float64 with shapes {shapes}')
            else:
                w=init(args.p,args.hidden,seeds['init_seed'])
            initial=[x.copy() for x in w]
            m=[np.zeros_like(x) for x in w]; v=[x.copy() for x in m]
        with timing.measure('data_write'):
            np.savez_compressed(out/'data.npz',a=a,b=b,y=target,yadd=y,train=masks[0],cal=masks[1],audit=masks[2])
        extra={int(s) for s in args.checkpoint_steps.split(',') if s}
        with (out/'metrics.jsonl').open('w') as stream:
            for step in range(args.steps+1):
                logging=step%args.log_every==0 or step==args.steps
                update=None
                if step:
                    with timing.measure('training'):
                        tr=masks[0]
                        loss,g=lossgrad(w,a[tr],b[tr],target[tr])
                        if not np.isfinite(loss): raise FloatingPointError('nonfinite loss; retain failed run')
                        update=adam(w,g,m,v,step,args.lr,args.wd,measure=logging)
                    completed=step
                if logging:
                    with timing.measure('evaluation'):
                        logits=forward(w,a,b)[0]
                        acc=[accuracy(logits,target,j) for j in masks]
                        row=dict(step=step,seconds=time.perf_counter()-timing.start,
                                 train=acc[0],cal=acc[1],audit=acc[2],
                                 norm=float(np.sqrt(sum(np.sum(x*x) for x in w))),
                                 objective={name:objective_metrics(logits,target,j)
                                            for name,j in zip(('train','cal','audit'),masks)},
                                 optimizer_update=update)
                    first=first_interp is None and acc[0]>=.99
                    if first: first_interp=step
                    if args.diagnostics=='inline' and (step%args.audit_every==0 or first or step==args.steps):
                        with timing.measure('compression'):
                            row['compression']=compress(w,a,b,masks)
                        with timing.measure('intervention'):
                            row['intervention']=interventions(w,a,b,target,masks[2],seeds['diagnostic_seed'],args.shams)
                        with timing.measure('readout'):
                            row['readout_accessibility']=transfer(w,initial,a,b,y,masks,seeds['diagnostic_seed']+10000)
                    if step%args.checkpoint_every==0 or step in extra or first or step==args.steps:
                        with timing.measure('checkpoint_write'):
                            np.savez_compressed(out/f'checkpoint_{step:06d}.npz',U=w[0],V=w[1],W=w[2],
                                                mU=m[0],mV=m[1],mW=m[2],vU=v[0],vV=v[1],vW=v[2],step=step)
                    row['elapsed_after_diagnostics']=time.perf_counter()-timing.start
                    with timing.measure('metrics_write'):
                        stream.write(json.dumps(row)+'\n'); stream.flush()
                    (out/'runtime.json').write_text(json.dumps(dict(status='running',completed_steps=completed,
                                                                  **timing.snapshot()),indent=2))
        status='complete'
    finally:
        report=dict(status=status,completed_steps=completed,first_interpolation=first_interp,
                    **timing.snapshot())
        report['output_bytes_before_runtime_report']=sum(p.stat().st_size for p in out.iterdir()
                                                        if p.is_file() and p.name!='runtime.json')
        report['timing_scope']='Excludes Python startup/imports; phases are nonoverlapping. Parent benchmark includes startup.'
        (out/'runtime.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(out=str(out),seconds=report['wall_seconds'],first_interpolation=first_interp,
                          final_accuracy=acc)))


def selftest():
    p=7; a,b,y,yr,masks=data(p,8); w=init(p,9,12)
    assert sorted(np.concatenate(masks).tolist())==list(range(p*p))
    for ix in masks:
        assert np.array_equal(np.bincount(y[ix],minlength=p),np.bincount(yr[ix],minlength=p))
        assert set((a[ix]*p+b[ix]).tolist())==set((b[ix]*p+a[ix]).tolist())
    loss,g=lossgrad(w,a,b,y)
    for j,pos in [(0,(2,3)),(1,(4,1)),(2,(3,2))]:
        old=w[j][pos]; eps=1e-6
        w[j][pos]=old+eps; lp=lossgrad(w,a,b,y)[0]
        w[j][pos]=old-eps; lm=lossgrad(w,a,b,y)[0]
        w[j][pos]=old
        assert np.isclose((lp-lm)/(2*eps),g[j][pos],rtol=1e-4,atol=1e-7)
    # Input/output relabeling with conjugated weights is exactly the same task.
    r=np.random.default_rng(21); pi=r.permutation(p); rho=r.permutation(p)
    ww=[np.empty_like(w[0]),np.empty_like(w[1]),np.empty_like(w[2])]
    ww[0][pi]=w[0]; ww[1][pi]=w[1]; ww[2][:,rho]=w[2]
    assert np.allclose(forward(ww,pi[a],pi[b])[0][:,rho],forward(w,a,b)[0])
    cc=compress(w,a,b,masks)
    assert cc['train']['agreement']>=.99 and cc['calibration']['relative_mse']<=.01
    # Actual optimization decreases loss, not just a finite-difference match.
    m=[np.zeros_like(x) for x in w]; v=[x.copy() for x in m]
    for t in range(1,51):
        _,gg=lossgrad(w,a,b,y); adam(w,gg,m,v,t,.01,0)
    assert lossgrad(w,a,b,y)[0]<loss
    ii=interventions(w,a,b,y,masks[2],44,3)
    assert len(ii['sham_accuracy'])==3
    print('PASS: disjoint/grouped splits, label counts, finite-difference gradients, relabel invariance, codec fidelity, learning, intervention execution')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--out',default='run'); ap.add_argument('--task',choices=['add','random'],default='add')
    ap.add_argument('--seed',type=int,default=1000,help='Historical: derives any seed not given explicitly')
    ap.add_argument('--data-seed',type=int); ap.add_argument('--init-seed',type=int)
    ap.add_argument('--control-seed',type=int,help='Shuffled-label table; omitted = historical shared stream')
    ap.add_argument('--diagnostic-seed',type=int)
    ap.add_argument('--expect-split-sha256',help='Fail before training unless the split hash matches')
    ap.add_argument('--init-weights',help='Start from U, V, W in this .npz instead of init(); Adam starts fresh')
    ap.add_argument('--expect-init-sha256',help='Fail before training unless the --init-weights file hash matches')
    ap.add_argument('--init-age',type=int,help='Updates the --init-weights source had taken (recorded only; steps here start at 0)')
    ap.add_argument('--run-id'); ap.add_argument('--purpose'); ap.add_argument('--session-label')
    ap.add_argument('--p',type=int,default=31)
    ap.add_argument('--train-fraction',type=float,default=.3,help='Share of grouped pairs per class in train')
    ap.add_argument('--hidden',type=int,default=128); ap.add_argument('--steps',type=int,default=50000)
    ap.add_argument('--lr',type=float,default=.001); ap.add_argument('--wd',type=float,default=1.)
    ap.add_argument('--log-every',type=int,default=100); ap.add_argument('--audit-every',type=int,default=1000)
    ap.add_argument('--checkpoint-every',type=int,default=1000)
    ap.add_argument('--checkpoint-steps',default='',help='Extra checkpoint steps, comma separated, multiples of --log-every')
    ap.add_argument('--diagnostics',choices=['inline','none'],default='inline',
                    help='inline: historical compression/top-4 intervention/readout at --audit-every')
    ap.add_argument('--shams',type=int,default=32)
    args=ap.parse_args()
    if args.selftest: selftest()
    else:
        if args.p<7 or args.p%2==0 or args.hidden<1 or args.steps<0 or min(args.log_every,args.audit_every,args.checkpoint_every,args.shams)<1:
            ap.error('Require odd p >= 7, positive hidden/cadences/shams, and nonnegative steps')
        if not 0<args.train_fraction<=.8:
            ap.error('Require 0 < --train-fraction <= .8 (calibration keeps .1)')
        if args.checkpoint_every%args.log_every or args.audit_every%args.log_every or any(
                int(s)%args.log_every for s in args.checkpoint_steps.split(',') if s):
            ap.error('Checkpoint and audit cadences must be multiples of --log-every')
        run(args)
