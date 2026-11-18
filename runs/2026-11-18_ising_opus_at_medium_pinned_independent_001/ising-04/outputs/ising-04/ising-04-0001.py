
import numpy as np, time
Ts=np.array([1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8])
N=16;R=128;nT=len(Ts)
rng=np.random.default_rng(12345)
s=np.ones((nT,R,N,N),dtype=np.int8)
ii,jj=np.indices((N,N));masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep():
    global s
    for m in masks:
        nb=(np.roll(s,1,2)+np.roll(s,-1,2)+np.roll(s,1,3)+np.roll(s,-1,3)).astype(np.int16)
        dE=2*s*nb
        p=np.exp(-dE/Ts[:,None,None,None])
        flip=(rng.random(s.shape)<p)&m
        s=np.where(flip,-s,s)
t=time.time()
for _ in range(20): sweep()
print((time.time()-t)/20)
