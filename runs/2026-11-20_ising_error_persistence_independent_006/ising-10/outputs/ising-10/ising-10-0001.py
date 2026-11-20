
import numpy as np, time
N=16; R=2000; T=2.269
rng=np.random.default_rng(1)
S=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
t0=time.time()
for sweep in range(20):
    for m in masks:
        nb=(np.roll(S,1,1)+np.roll(S,-1,1)+np.roll(S,1,2)+np.roll(S,-1,2))
        dE=2*S*nb
        acc=(dE<=0)|(rng.random(S.shape,dtype=np.float32)<np.exp(-np.maximum(dE,0)/T))
        S=np.where(m[None]&acc,-S,S).astype(np.int8)
print((time.time()-t0)/20,"s per sweep for R=",R)
