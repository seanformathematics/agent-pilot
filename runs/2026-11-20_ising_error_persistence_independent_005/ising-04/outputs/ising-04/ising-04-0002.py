
import time, numpy as np, sys
T=1.9; N=16; R=1000; seed=101; nequil=3000; nmeas=30000
rng=np.random.default_rng(seed)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
beta=1.0/T
def sweep(s):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float32)
        dE=2*s.astype(np.float32)*nb
        acc=(rng.random(s.shape,dtype=np.float32)<np.exp(-beta*dE))
        s[acc&m]*=-1
for _ in range(nequil): sweep(s)
m2=np.zeros(R); m4=np.zeros(R); m1=np.zeros(R)
t=time.time()
for k in range(nmeas):
    sweep(s)
    mm=s.sum(axis=(1,2),dtype=np.int64)/(N*N)
    mm=mm.astype(np.float64)
    m2+=mm**2; m4+=mm**4; m1+=np.abs(mm)
m2/=nmeas; m4/=nmeas; m1/=nmeas
np.savez(f"/work/res_{T}.npz",m2=m2,m4=m4,m1=m1,seed=seed,nequil=nequil,nmeas=nmeas)
U_r=1-m4/(3*m2**2)
print("time",time.time()-t)
print("U mean",U_r.mean(),"sem",U_r.std(ddof=1)/np.sqrt(R))
