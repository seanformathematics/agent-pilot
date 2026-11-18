
import numpy as np, time, json
N=16
def run(T,R=1024,neq=1000,nmeas=4000,seed=0):
    rng=np.random.default_rng(seed)
    s=np.ones((R,N,N),dtype=np.int8)
    ii,jj=np.indices((N,N)); masks=[((ii+jj)%2==c) for c in (0,1)]
    pacc={dE:np.exp(-dE/T) for dE in (4,8)}
    M2=np.zeros(R);M4=np.zeros(R)
    for t in range(neq+nmeas):
        for mk in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            p=np.where(dE<=0,1.0,np.where(dE==4,pacc[4],pacc[8]))
            flip=(rng.random(s.shape)<p)&mk
            s=np.where(flip,-s,s).astype(np.int8)
        if t>=neq:
            m=s.sum((1,2))/N**2; m2=m*m; M2+=m2; M4+=m2*m2
    M2/=nmeas;M4/=nmeas
    U=1-M4.mean()/(3*M2.mean()**2)
    S2=M2.sum();S4=M4.sum()
    Uj=1-((S4-M4)/(R-1))/(3*((S2-M2)/(R-1))**2)
    err=np.sqrt((R-1)/R*((Uj-Uj.mean())**2).sum())
    return float(U),float(err),R*nmeas
Ts=[1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res={}
t0=time.time()
for k,T in enumerate(Ts):
    res[T]=run(T,seed=1000+k); print(T,res[T],time.time()-t0,flush=True)
    json.dump(res,open('/work/res.json','w'))
