
import numpy as np, time
N=16; R=1000; T=2.27
rng=np.random.default_rng(1)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T):
    for m in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-2+2,2)*0)  # placeholder
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*h
        acc=rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T)
        flip=acc&m
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(50): s=sweep(s,T)
print((time.time()-t)/50, "s per sweep for R=",R)
