
import numpy as np, time
N=16
def setup(R,seed):
    rng=np.random.default_rng(seed)
    s=np.where(rng.random((R,N,N))<0.5,-1,1).astype(np.int8)
    ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
    masks=[((ii+jj)%2==c) for c in (0,1)]
    return rng,s,masks
def sweep(s,T,rng,masks):
    for mk in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*h  # energy change for flipping
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-np.maximum(dE,0)/T))
        s[acc&mk]*=-1
    return s
R=2000
rng,s,masks=setup(R,1)
t=time.time()
for k in range(50): sweep(s,2.27,rng,masks)
print("time per sweep (R=2000):",(time.time()-t)/50)
