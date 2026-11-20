
import numpy as np, time, json
N=16
def run(T, R, nsweep, nburn, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    p4=np.exp(-4/T); p8=np.exp(-8/T)
    def sweep(s):
        for m in masks:
            nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nb
            r=rng.random(s.shape)
            flip=m & ((dE<=0) | ((dE==4)&(r<p4)) | ((dE==8)&(r<p8)))
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nburn): s=sweep(s)
    M2=np.zeros(R); M4=np.zeros(R); cnt=0
    for _ in range(nsweep):
        s=sweep(s)
        M=s.reshape(R,-1).sum(1).astype(float)
        M2+=M**2; M4+=M**4; cnt+=1
    return M2/cnt, M4/cnt
def U_and_err(m2,m4):
    R=len(m2)
    U=1-m4.mean()/(3*m2.mean()**2)
    # delete-one jackknife over replicas
    S2=m2.sum(); S4=m4.sum()
    Uj=1-((S4-m4)/(R-1))/(3*((S2-m2)/(R-1))**2)
    err=np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return U,err
out={}
for T in [1.9,2.0,2.1,2.2,2.3,2.4]:
    t=time.time()
    seed=int(round(T*1000))+7
    m2,m4=run(T,2000,4000,2000,seed)
    U,e=U_and_err(m2,m4)
    np.savez(f'/work/T{T}.npz',m2=m2,m4=m4)
    out[T]=(U,e)
    print(T,U,e,time.time()-t,flush=True)
