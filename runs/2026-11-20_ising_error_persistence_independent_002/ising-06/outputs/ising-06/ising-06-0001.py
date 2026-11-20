
import time, numpy as np
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
rng = np.random.default_rng(1)
R, N = 1000, 16
s = rng.choice([-1,1], size=(R,N,N)).astype(np.int8)
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
masks = [((ii+jj)%2==p) for p in (0,1)]
def sweep(s, beta):
    for m in masks:
        nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        dE = 2*s*nb
        acc = (rng.random(s.shape) < np.exp(-beta*np.clip(dE,0,None)))
        flip = m[None] & ((dE<=0) | acc)
        s[flip] *= -1
    return s
t=time.time()
for _ in range(100): sweep(s, 1/2.2)
print("time per sweep (R=1000):", (time.time()-t)/100)
