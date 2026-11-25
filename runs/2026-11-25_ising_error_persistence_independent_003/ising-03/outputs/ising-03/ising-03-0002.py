import numpy as np, time
N=16
def make(R, T, seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==p) for p in (0,1)]
    # acceptance prob for prod=s*nb in -4..4 ; dE=2*prod
    P=np.array([min(1.0,np.exp(-2*p/T)) for p in range(-4,5)])
    return s,rng,masks,P
def sweep(s,rng,masks,P):
    for mk in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        prod=(s*nb+4)
        acc=(rng.random(s.shape,dtype=np.float32)<P[prod]) & mk
        s=np.where(acc,-s,s).astype(np.int8)
    return s
R=1000
s,rng,masks,P=make(R,2.27,1)
t=time.time()
for k in range(200): s=sweep(s,rng,masks,P)
dt=(time.time()-t)/200
print("time per sweep (R=1000):",dt)
# quick autocorr test of |m|
ms=[]
for k in range(400):
    s=sweep(s,rng,masks,P)
    ms.append(np.abs(s.sum(axis=(1,2))/N**2).mean())
ms=np.array(ms)
print("mean |m|",ms.mean())
