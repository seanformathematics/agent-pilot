
import numpy as np, time
N=16; R=700
rng=np.random.default_rng(1)
T=np.linspace(1.5,2.8,14)
Tr=np.repeat(T,R//14+1)[:R]
beta=(1/Tr)[:,None,None]
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s):
    for m in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h
        acc=np.logical_or(np.less_equal(dE,0), rng.random(s.shape)<np.exp(-beta*np.maximum(dE,0)))
        flip=np.logical_and(acc,m)
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(100): s=sweep(s)
print('time per sweep',(time.time()-t)/100)
