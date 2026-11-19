
code = r'''
import numpy as np, os, time
N=16
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
MASKS = [((ii+jj)%2==0), ((ii+jj)%2==1)]
def sweep(s, acc, rng):
    for mask in MASKS:
        h = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
        x = (s*h).astype(np.int16)+4
        p = acc[x]
        flip = (rng.random(s.shape) < p) & mask
        s[flip] *= -1
    return s
def step(T, R, nsweeps_burn, nblocks, per_block, seed, path):
    """Resumable: if path exists, load state; otherwise initialise random and burn in."""
    beta=1.0/T
    acc=np.array([min(1.0,np.exp(-2*beta*x)) for x in range(-4,5)])
    if os.path.exists(path):
        d=np.load(path, allow_pickle=True)
        s=d['s']; rng=np.random.default_rng(d['seedstate'].item()) if False else None
        rng=np.random.Generator(np.random.PCG64()); rng.bit_generator.state=d['bitgen'].item()
        burned=int(d['burned']); blocks_m2=list(d['m2']); blocks_m4=list(d['m4'])
    else:
        rng=np.random.default_rng(seed)
        s=np.where(rng.random((R,N,N))<0.5,1,-1).astype(np.int8)
        burned=0; blocks_m2=[]; blocks_m4=[]
    if burned==0:
        for _ in range(nsweeps_burn): s=sweep(s,acc,rng)
        burned=1
    t=time.time()
    while len(blocks_m2)<nblocks:
        S2=np.zeros(R); S4=np.zeros(R)
        for _ in range(per_block):
            s=sweep(s,acc,rng)
            m=s.reshape(R,-1).mean(axis=1)
            m2=m*m; S2+=m2; S4+=m2*m2
        blocks_m2.append(S2/per_block); blocks_m4.append(S4/per_block)
        np.savez(path, s=s, bitgen=np.array(rng.bit_generator.state,dtype=object), burned=burned,
                 m2=np.array(blocks_m2), m4=np.array(blocks_m4))
    return np.array(blocks_m2), np.array(blocks_m4)
'''
open('/work/ising2.py','w').write(code)
import sys, time, numpy as np
sys.path.insert(0,'/work')
from ising2 import step
t=time.time()
m2,m4 = step(1.9, 400, 3000, 2, 500, 7, '/work/test19.npz')
print("elapsed", time.time()-t)
print(m2.mean(axis=1), m4.mean(axis=1))
