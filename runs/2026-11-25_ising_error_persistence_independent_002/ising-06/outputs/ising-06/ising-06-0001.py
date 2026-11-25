
import numpy as np, time, os
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
os.makedirs('/work', exist_ok=True)
N=16; R=2000
rng=np.random.default_rng(12345)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
ii,jj=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((ii+jj)%2==0),((ii+jj)%2==1)]
def sweep(s,T,rng):
    for m in masks:
        nn=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*nn  # energy change for flipping s
        acc=(dE<=0)|(rng.random(s.shape,dtype=np.float32)<np.exp(-np.maximum(dE,0)/T).astype(np.float32))
        s[acc & m[None]] *= -1
    return s
t=time.time()
for _ in range(20): s=sweep(s,2.27,rng)
print("time per sweep (R=2000):",(time.time()-t)/20)
