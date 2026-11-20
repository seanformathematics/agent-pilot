
import numpy as np, time
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
R,N=2000,16
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==c) for c in (0,1)]
def sweep(s,T,rng):
    for c in (0,1):
        nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        dE=2*s*nb
        acc=(dE<=0)|(rng.random(s.shape)<np.exp(-np.clip(dE,0,None)/T))
        s=np.where(acc&masks[c],-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(100): s=sweep(s,2.2,rng)
print("time per sweep (R=2000)", (time.time()-t)/100)
