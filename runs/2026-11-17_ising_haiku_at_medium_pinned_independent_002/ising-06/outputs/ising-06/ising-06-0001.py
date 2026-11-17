
import numpy as np, time
N=16; R=400
rng=np.random.default_rng(1)
s=rng.choice([-1,1],size=(R,N,N)).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        flip=acc&m
        s[flip]*=-1
    return s
t=time.time()
for _ in range(20): sweep(s,2.27,rng)
print((time.time()-t)/20, "s per sweep for R=",R)
