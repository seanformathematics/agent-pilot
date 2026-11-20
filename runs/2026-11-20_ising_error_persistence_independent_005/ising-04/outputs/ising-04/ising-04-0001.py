
import time, numpy as np
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
N=16; R=1000
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,beta):
    for m in masks:
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb  # energy change for flipping s (J=1)
        acc=(rng.random(s.shape,dtype=np.float32)<np.exp(-beta*dE.astype(np.float32)))
        flip=acc&m
        s[flip]*=-1
    return s
t=time.time()
for _ in range(20): sweep(s,1/2.2)
print("per sweep", (time.time()-t)/20)
