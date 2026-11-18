
import time, numpy as np, sys
print(sys.version)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba", e)

N=16; R=1000; T=2.269
rng = np.random.default_rng(12345)
s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
even = ((ii+jj)%2==0)
odd = ~even
def sweep(s, T, rng):
    for mask in (even, odd):
        nb = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        dE = 2*s*nb
        acc = (dE<=0) | (rng.random(s.shape) < np.exp(-np.maximum(dE,0)/T))
        flip = acc & mask
        s[flip] *= -1
    return s
t0=time.time()
for _ in range(100):
    s = sweep(s, T, rng)
dt=time.time()-t0
print("time per sweep (R=1000):", dt/100)
