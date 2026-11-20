
import numpy as np, time, json, os
N=16
def sweeps_run(T,R,n_eq,n_meas,seed,nwin=4):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    pacc=np.array([1.0 if x<=0 else np.exp(-2*x/T) for x in range(-4,5)])
    def sweep(s):
        for m in masks:
            nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
            sn=(s*nb+4).astype(np.int8)
            acc=rng.random((R,N,N))<pacc[sn]
            s=np.where(acc&m,-s,s).astype(np.int8)
        return s
    for _ in range(n_eq): s=sweep(s)
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    win=n_meas//nwin; wins=[]
    w2=np.zeros(R); w4=np.zeros(R); wc=0
    for k in range(n_meas):
        s=sweep(s)
        m=s.sum((1,2))/(N*N); m2=m*m
        S2+=m2; S4+=m2*m2; cnt+=1
        w2+=m2; w4+=m2*m2; wc+=1
        if wc==win:
            wins.append(1-(w4/wc).mean()/(3*(w2/wc).mean()**2)); w2[:]=0; w4[:]=0; wc=0
    return S2/cnt,S4/cnt,wins
def binder_jk(S2,S4):
    R=len(S2)
    U=1-S4.mean()/(3*S2.mean()**2)
    tot2=S2.sum(); tot4=S4.sum()
    loo2=(tot2-S2)/(R-1); loo4=(tot4-S4)/(R-1)
    Uj=1-loo4/(3*loo2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return float(U),float(err)
res={}
for T,seed in [(2.3,105),(2.4,106)]:
    t=time.time()
    S2,S4,wins=sweeps_run(T,2000,4000,4000,seed)
    U,err=binder_jk(S2,S4)
    res[T]=dict(U=U,err=err,wins=[float(w) for w in wins],seed=seed,R=2000,n_eq=4000,n_meas=4000)
    print(T,U,err,[round(w,4) for w in wins],'time',round(time.time()-t,1),flush=True)
    json.dump(res,open('/work/res_c.json','w'))
