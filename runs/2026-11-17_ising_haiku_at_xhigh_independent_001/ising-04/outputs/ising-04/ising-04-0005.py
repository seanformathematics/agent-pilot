import sys, time; sys.path.insert(0,'/work')
from ising import *
import numpy as np
N=16; R=1000
Ts=[2.2,2.3,2.4,2.5,2.6,2.7,2.8]
masks=make_masks(N)
B=10
def bonds_energy(s):
    return -(np.sum(s*np.roll(s,-1,1),axis=(1,2))+np.sum(s*np.roll(s,-1,2),axis=(1,2)))
t0=time.time()
for idx,T in enumerate(Ts):
    seed=20240601+7+idx
    rng=np.random.default_rng(seed)
    pac=pacc_table(T)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int32)
    for _ in range(2000):
        s=sweep(s,pac,masks,rng)
    nmeas=3000
    nb=nmeas//B
    bm2=np.zeros((nb,R)); bm4=np.zeros((nb,R)); be=np.zeros((nb,R))
    acc2=np.zeros(R); acc4=np.zeros(R); acce=np.zeros(R)
    for k in range(nmeas):
        s=sweep(s,pac,masks,rng)
        m=s.sum(axis=(1,2))/N**2
        e=bonds_energy(s)/N**2
        acc2+=m**2; acc4+=m**4; acce+=e
        if (k+1)%B==0:
            j=(k+1)//B-1
            bm2[j]=acc2/B; bm4[j]=acc4/B; be[j]=acce/B
            acc2[:]=0; acc4[:]=0; acce[:]=0
    np.savez(f'/work/prod_T{T:.2f}.npz', bm2=bm2, bm4=bm4, be=be, seed=seed, T=T)
    m2=bm2.mean(); m4=bm4.mean()
    print(f"T={T} seed={seed} U_quick={1-m4/(3*m2**2):.5f} <m2>={m2:.5f} <e>={be.mean():.5f} time={time.time()-t0:.1f}", flush=True)
