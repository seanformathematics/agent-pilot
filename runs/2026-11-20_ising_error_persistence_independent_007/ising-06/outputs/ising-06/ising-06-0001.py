
import numpy as np, time
N=16; R=1000
rng=np.random.default_rng(1)
s=rng.choice([-1,1],size=(R,N,N)).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
T=2.2
t0=time.time()
for sweep in range(50):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        upd=acc&m
        s=np.where(upd,-s,s).astype(np.int8)
print((time.time()-t0)/50,"s per sweep for R=",R)
