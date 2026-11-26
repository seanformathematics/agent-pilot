
import numpy as np, time
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
N=16; R=512
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.maximum(dE,0)/T))
        flip=acc&m
        s[flip]*=-1
    return s
t=time.time()
for k in range(100): sweep(s,2.27,rng)
print("time per sweep",(time.time()-t)/100)
