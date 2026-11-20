import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np, sys, time
def run(T, R, N, nwarm, nmeas, every, seed, init=None):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N)) if init is None else init.copy()
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii+jj)%2==p) for p in (0,1)]
    # acceptance table for x = s*nsum in {-4..4}; dE = 2*x
    tab = np.array([np.exp(-2.0*x/T) for x in range(-4,5)])
    tab_pos = {x: np.exp(-2.0*x/T) for x in (1,2,3,4)}
    accp = np.zeros(9)
    for x in range(-4,5):
        accp[x+4] = 1.0 if x<=0 else np.exp(-2.0*x/T)
    def sweep(s):
        for p in (0,1):
            nsum = (np.roll(s,1,1)+np.roll(s,-1,1)+np.roll(s,1,2)+np.roll(s,-1,2))
            x = (s*nsum).astype(np.int16)
            pa = accp[x+4]
            u = rng.random(s.shape)
            flip = (u < pa) & masks[p]
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(nwarm):
        s = sweep(s)
    M2 = []; M4 = []; Mabs=[]
    for k in range(nmeas):
        for _ in range(every):
            s = sweep(s)
        m = s.reshape(R,-1).mean(axis=1, dtype=np.float64)
        m2 = m*m
        M2.append(m2); M4.append(m2*m2); Mabs.append(np.abs(m))
    return s, np.array(M2), np.array(M4), np.array(Mabs)
'''
open('/work/ising.py','w').write(code)
import sys; sys.path.insert(0,'/work')
from ising import run
t=time.time()
s,M2,M4,Ma = run(2.27, 1000, 16, 10, 10, 1, 1)
print("time for 20 sweeps R=1000:", time.time()-t)
print(M2.shape, (1-M4.mean()/(3*M2.mean()**2)))
