
import numpy as np, time, os
os.makedirs('/work',exist_ok=True)
code = r'''
import numpy as np, sys, time, json
def run(T, R, nburn, nmeas, seed, N=16, state=None):
    rng = np.random.default_rng(seed)
    beta = 1.0/T
    # acceptance table indexed by s*h+4 where h = neighbour sum, dE = 2*s*h
    acc = np.zeros(9)
    for x in range(-4,5):
        acc[x+4] = min(1.0, np.exp(-2*beta*x))
    if state is None:
        s = np.where(rng.random((R,N,N))<0.5, 1, -1).astype(np.int8)
    else:
        s = state
    # checkerboard masks
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
    def half(s, mask):
        h = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)).astype(np.int8)
        x = (s*h).astype(np.int16)+4
        p = acc[x]
        u = rng.random(s.shape)
        flip = (u < p) & mask
        s[flip] *= -1
        return s
    def sweep(s):
        s = half(s, masks[0]); s = half(s, masks[1]); return s
    for _ in range(nburn):
        s = sweep(s)
    S2 = np.zeros(R); S4 = np.zeros(R); S1=np.zeros(R)
    V = N*N
    for _ in range(nmeas):
        s = sweep(s)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        S2 += m2; S4 += m2*m2; S1 += np.abs(m)
    return s, S2/nmeas, S4/nmeas, S1/nmeas
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import run
t=time.time()
s,m2,m4,am = run(2.269, 400, 0, 200, 1)
print("time per sweep R=400:", (time.time()-t)/200)
print(m2.mean(), m4.mean())
