
import numpy as np, json, time, os
from multiprocessing import Pool
N=16; R=1000; NEQ=1500; NMEAS=4000; NGRP=20
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==p) for p in (0,1)]
def run(args):
    T,seed=args
    rng=np.random.default_rng(seed)
    S=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
    # acceptance probs for s*h, h in {-4..4}
    # dE = 2*s*h ; accept prob min(1,exp(-dE/T))
    def sweep():
        for m in masks:
            nb=(np.roll(S,1,1).astype(np.int8)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
            dE=2*S*nb
            acc=np.exp(-np.clip(dE,0,None)/T)
            r=rng.random(S.shape)
            flip=(r<acc)&m
            S[flip]*=-1
    for _ in range(NEQ): sweep()
    m2=np.zeros((NMEAS,R)); m4=np.zeros((NMEAS,R))
    for t in range(NMEAS):
        sweep()
        mm=S.reshape(R,-1).mean(axis=1,dtype=np.float64)
        m2[t]=mm**2; m4[t]=mm**4
    # per-replica time averages
    a2=m2.mean(0); a4=m4.mean(0)
    # jackknife over replica groups
    groups=np.array_split(np.arange(R),NGRP)
    U_full=1-a4.mean()/(3*a2.mean()**2)
    Uj=[]
    for g in groups:
        keep=np.ones(R,bool); keep[g]=False
        Uj.append(1-a4[keep].mean()/(3*a2[keep].mean()**2))
    Uj=np.array(Uj)
    err=np.sqrt((NGRP-1)/NGRP*np.sum((Uj-Uj.mean())**2))
    return dict(T=T,U=float(U_full),U_err=float(err),m2=float(a2.mean()),m4=float(a4.mean()))
if __name__=="__main__":
    t0=time.time()
    with Pool(8) as p:
        res=p.map(run,[(T,1000+k) for k,T in enumerate(Ts)])
    for r in res: print(r)
    json.dump(res,open('/work/res1.json','w'))
    print("time",time.time()-t0)
