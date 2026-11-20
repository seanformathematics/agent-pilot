
import time, numpy as np
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
import os; print(os.cpu_count())
R=512; N=16
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
mask0=((ii+jj)%2==0); mask1=~mask0
def sweep(s,beta,rng):
    for m in (mask0,mask1):
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.float32)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-beta*np.clip(dE,0,None)))
        flip=acc&m
        s=np.where(flip,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(20): s=sweep(s,1/2.27,rng)
print("time per sweep (512 replicas):",(time.time()-t)/20)
