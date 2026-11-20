
import numpy as np, time, os
N=16
os.makedirs('/work',exist_ok=True)
def run(T, R, nsweep, nequil, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    acc={4:np.exp(-4/T),8:np.exp(-8/T)}
    def sweep(s):
        for p in (0,1):
            nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE=2*s*nn
            prob=np.where(dE<=0,1.0,np.where(dE==4,acc[4],np.where(dE==8,acc[8],0.0)))
            u=rng.random((R,N,N))
            flip=(u<prob)&masks[p]
            s=np.where(flip,-s,s).astype(np.int8)
        return s
    for _ in range(nequil): s=sweep(s)
    m2=np.zeros(R); m4=np.zeros(R); cnt=0
    for _ in range(nsweep):
        s=sweep(s)
        m=s.reshape(R,-1).mean(1)
        m2+=m*m; m4+=m**4; cnt+=1
    return m2/cnt, m4/cnt
def binder(m2,m4):
    return 1-m4.mean()/(3*m2.mean()**2)
def jack(m2,m4):
    R=len(m2); Ujk=np.empty(R)
    S2=m2.sum(); S4=m4.sum()
    for i in range(R):
        a=(S2-m2[i])/(R-1); b=(S4-m4[i])/(R-1)
        Ujk[i]=1-b/(3*a*a)
    U=binder(m2,m4)
    err=np.sqrt((R-1)/R*np.sum((Ujk-Ujk.mean())**2))
    return U,err
import sys
for T in [1.9,2.0,2.1]:
    t=time.time()
    m2,m4=run(T,1000,8000,1000,seed=int(T*1000))
    np.savez(f'/work/T{T}.npz',m2=m2,m4=m4)
    U,e=jack(m2,m4)
    print(T,U,e,time.time()-t,flush=True)
