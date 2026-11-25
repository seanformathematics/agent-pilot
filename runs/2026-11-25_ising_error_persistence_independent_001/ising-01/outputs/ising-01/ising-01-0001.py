
import time, numpy as np
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
R,N=1000,16
rng=np.random.default_rng(1)
s=rng.choice([-1,1],size=(R,N,N)).astype(np.int8)
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb
        p=np.exp(-dE/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        upd=acc&m
        s[upd]*=-1
    return s
t=time.time()
for _ in range(200): sweep(s,2.2)
print("time per sweep",(time.time()-t)/200)
