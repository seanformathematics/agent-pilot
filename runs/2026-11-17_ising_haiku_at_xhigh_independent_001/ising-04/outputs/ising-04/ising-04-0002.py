import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np
def make_masks(N):
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    m0 = ((ii+jj)%2==0)
    return m0, ~m0
def pacc_table(T):
    # index k=(s*nn+4)>>1, s*nn in {-4,-2,0,2,4}; dE = 2*s*nn
    sn = np.array([-4,-2,0,2,4])
    dE = 2*sn
    p = np.where(dE<=0, 1.0, np.exp(-np.maximum(dE,0)/T))
    return p.astype(np.float32)
def sweep(s, pacc, masks, rng):
    for mask in masks:
        nn = np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2)
        k = (s*nn+4)>>1
        p = pacc[k]
        r = rng.random(s.shape, dtype=np.float32)
        flip = mask & (r < p)
        s = np.where(flip, -s, s)
    return s
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import *
N=16; R=1000
rng=np.random.default_rng(12345)
masks=make_masks(N)
s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int32)
pac=pacc_table(2.269)
t=time.time()
for _ in range(50):
    s=sweep(s,pac,masks,rng)
dt=(time.time()-t)/50
print("time per sweep R=1000:",dt, " replica-sweeps/s:", R/dt)
