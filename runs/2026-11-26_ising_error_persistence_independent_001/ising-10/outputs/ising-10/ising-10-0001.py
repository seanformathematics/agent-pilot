import time, numpy as np
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
R,N=1000,16
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==k) for k in (0,1)]
def sweep(s,beta,rng):
    for m in masks:
        nb=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nb  # energy change for flipping
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-beta*np.clip(dE,0,None)).astype(np.float32))
        flip=acc&m
        s[flip]*=-1
    return s
t=time.time()
for _ in range(20): sweep(s,1/2.2,rng)
print("time per sweep R=1000:", (time.time()-t)/20)