
import time, numpy as np
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
R=1000; N=16
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T):
    for m in masks:
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE=2*s*h
        p=np.exp(-dE/T)
        acc=(rng.random(s.shape)<p)&m
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(200): s=sweep(s,2.27)
print("time per sweep (R=1000):",(time.time()-t)/200)
