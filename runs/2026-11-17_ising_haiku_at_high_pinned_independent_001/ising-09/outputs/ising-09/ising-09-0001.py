
import numpy as np, time, os
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba")
os.makedirs('/work', exist_ok=True)
N=16; R=1000
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for mk in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nn
        p=np.exp(-dE/T)
        acc=(dE<=0)|(rng.random(s.shape)<p)
        upd=acc&mk
        s[upd]*=-1
    return s
t=time.time()
for _ in range(200): sweep(s,2.0,rng)
print("time per sweep R=1000:",(time.time()-t)/200)
