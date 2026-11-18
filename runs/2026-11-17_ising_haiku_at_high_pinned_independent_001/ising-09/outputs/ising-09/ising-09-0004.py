
import numpy as np, time, os
N=16; R=1000
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for mk in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*nn
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-dE/T))
        s[acc&mk]*=-1
    return s
def run(T,seed,nequil=3000,nmeas=12000,every=10):
    rng=np.random.default_rng(seed)
    s=rng.choice(np.array([-1,1],dtype=np.int64),size=(R,N,N))
    for _ in range(nequil): sweep(s,T,rng)
    m2s=np.zeros(R); m4s=np.zeros(R); cnt=0
    for k in range(nmeas):
        sweep(s,T,rng)
        if k%every==0:
            m=s.mean(axis=(1,2))
            m2s+=m*m; m4s+=m**4; cnt+=1
    return m2s/cnt, m4s/cnt
def U_jack(m2r,m4r):
    R=len(m2r)
    U=1-m4r.mean()/(3*m2r.mean()**2)
    S2=m2r.sum(); S4=m4r.sum()
    a2=(S2-m2r)/(R-1); a4=(S4-m4r)/(R-1)
    Uj=1-a4/(3*a2*a2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
t0=time.time()
for i,T in enumerate([2.4,2.5,2.6]):
    m2r,m4r=run(T,seed=2004+i)
    np.savez(f'/work/res/T{T}.npz',m2=m2r,m4=m4r)
    U,err=U_jack(m2r,m4r)
    print(f"T={T} U={U:.5f} err={err:.5f} time={time.time()-t0:.0f}",flush=True)
