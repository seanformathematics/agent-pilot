
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np, sys, time, json
def run(T, R, nsweep, ntherm, seed, N=16):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj)%2==0), ((ii+jj)%2==1)]
    # acceptance table indexed by dE in {0,...,8} (dE = 2*s*nn)
    acc = np.exp(-np.arange(0,9)/T)
    def sweep(s):
        for m in masks:
            nn = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            dE = 2*s.astype(np.int16)*nn
            prob = np.where(dE<=0, 1.0, acc[np.clip(dE,0,8)])
            flip = (rng.random(s.shape) < prob) & m
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(ntherm):
        s = sweep(s)
    S2 = np.zeros(R); S4 = np.zeros(R); cnt = 0
    M2 = []; M4=[]
    for k in range(nsweep):
        s = sweep(s)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        S2 += m2; S4 += m2*m2; cnt += 1
    return S2/cnt, S4/cnt
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import run
t=time.time()
a,b = run(2.3, 1000, 50, 0, 1)
print("time per sweep R=1000:", (time.time()-t)/50)
