import sys, time; sys.path.insert(0,'/work')
from ising import *
import numpy as np
N=4
cfg=np.arange(2**(N*N))
bits=((cfg[:,None]>>np.arange(N*N))&1)
S=(2*bits-1).reshape(-1,N,N).astype(np.float64)
E=-(np.sum(S*np.roll(S,-1,1),axis=(1,2))+np.sum(S*np.roll(S,-1,2),axis=(1,2)))
m=S.sum(axis=(1,2))/N**2
masks=make_masks(N)
def exact_U(T):
    w=np.exp(-(E-E.min())/T); w/=w.sum()
    m2=np.sum(w*m**2); m4=np.sum(w*m**4)
    return 1-m4/(3*m2**2)
t0=time.time()
for T,seed in [(2.0,4242),(2.8,4343)]:
    rng=np.random.default_rng(seed)
    pac=pacc_table(T)
    R=4000
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int32)
    for _ in range(500): s=sweep(s,pac,masks,rng)
    acc2=np.zeros(R); acc4=np.zeros(R); nm=0
    B=10; blocks2=[];blocks4=[]
    for k in range(20000):
        s=sweep(s,pac,masks,rng)
        mm=s.sum(axis=(1,2))/N**2
        acc2+=mm**2; acc4+=mm**4
        if (k+1)%B==0:
            blocks2.append(acc2/B); blocks4.append(acc4/B); acc2[:]=0; acc4[:]=0
    bm2=np.array(blocks2); bm4=np.array(blocks4)
    p2=bm2.mean(axis=0); p4=bm4.mean(axis=0)
    U=1-p4.mean()/(3*p2.mean()**2)
    S2=p2.sum(); S4=p4.sum()
    jk=np.array([1-((S4-p4[r])/(R-1))/(3*((S2-p2[r])/(R-1))**2) for r in range(R)])
    err=np.sqrt((R-1)/R*np.sum((jk-jk.mean())**2))
    print(f"N=4 T={T}: MC U={U:.5f} ± {err:.5f} (jackknife); exact={exact_U(T):.5f}; diff/err={(U-exact_U(T))/err:.2f}; time={time.time()-t0:.1f}")
