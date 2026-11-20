
import numpy as np, time
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba")
N=16; R=1000; T=2.3
rng=np.random.default_rng(1)
s=rng.choice(np.array([-1,1],dtype=np.int8),size=(R,N,N))
I,J=np.meshgrid(np.arange(N),np.arange(N),indexing='ij')
masks=[((I+J)%2==c) for c in (0,1)]
def sweep(s,T,rng):
    for c in (0,1):
        h=(np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE=2*s*h
        p=np.exp(-np.clip(dE,0,None)/T)
        acc=(rng.random(s.shape)<p)&masks[c]
        s=np.where(acc,-s,s).astype(np.int8)
    return s
t=time.time()
for _ in range(100): s=sweep(s,T,rng)
print("time per sweep", (time.time()-t)/100)
