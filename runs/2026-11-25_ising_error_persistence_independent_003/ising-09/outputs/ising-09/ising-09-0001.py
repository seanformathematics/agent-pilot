import time, numpy as np
try:
    import numba; print("numba", numba.__version__)
except Exception as e: print("no numba", e)
rng=np.random.default_rng(1)
R=1000;N=16
s=rng.choice([-1,1],size=(R,N,N)).astype(np.int8)
t=time.time()
for _ in range(10):
    nb=np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
print("10 nbr sums", time.time()-t)
