
import numpy as np, time
N=16
def sweeps_run(T,R,n_eq,n_meas,seed,blocks=None):
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
    out=[]
    S2=np.zeros(R); S4=np.zeros(R); cnt=0
    for k in range(n_meas):
        s=sweep(s)
        m=s.sum((1,2))/(N*N); m2=m*m
        S2+=m2; S4+=m2*m2; cnt+=1
        if blocks and (k+1) in blocks:
            out.append((S2/cnt,S4/cnt)); 
    return S2/cnt,S4/cnt,out
def binder_jk(S2,S4):
    R=len(S2)
    U=1-S4.mean()/(3*S2.mean()**2)
    # delete-one jackknife over replicas (vectorized via sums)
    tot2=S2.sum(); tot4=S4.sum()
    loo2=(tot2-S2)/(R-1); loo4=(tot4-S4)/(R-1)
    Uj=1-loo4/(3*loo2**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
t=time.time()
S2,S4,out=sweeps_run(2.3,2000,3000,3000,11,blocks={1000,2000,3000})
print('time',time.time()-t)
print('full U,err',binder_jk(S2,S4))
# note out holds cumulative averages at blocks; cumulative only
print('per-sample std of m^2 across replicas',S2.std(),'U-mean',S4.mean()/(3*S2.mean()**2))
