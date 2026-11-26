
import numpy as np, time
N=16; R=1000; T=2.27
rng=np.random.default_rng(1)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T):
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int16)
        dE=2*s.astype(np.int16)*nn
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        upd=acc&m
        s[upd]*=-1
    return s
t=time.time()
for k in range(50): sweep(s,T)
print("per sweep",(time.time()-t)/50)
