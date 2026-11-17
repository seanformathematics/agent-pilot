import os, time
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

class SWBatch:
    def __init__(self, N, R, T, seed):
        self.N = N; self.R = R; self.T = T
        self.rng = np.random.default_rng(seed)
        self.p = 1.0 - np.exp(-2.0/T)
        self.S = self.rng.choice(np.array([-1,1],dtype=np.int8), size=(R,N,N))
        idx = np.arange(R*N*N).reshape(R,N,N)
        self.idx = idx
        # right neighbour and down neighbour indices
        self.j_right = np.roll(idx, -1, axis=2)
        self.j_down = np.roll(idx, -1, axis=1)
        self.M = R*N*N

    def sweep(self):
        S = self.S
        flat = S.reshape(-1)
        rng = self.rng
        # horizontal bonds
        ok_h = (S == np.roll(S, -1, axis=2)) & (rng.random(S.shape) < self.p)
        ok_v = (S == np.roll(S, -1, axis=1)) & (rng.random(S.shape) < self.p)
        i1 = np.concatenate([self.idx[ok_h], self.idx[ok_v]])
        j1 = np.concatenate([self.j_right[ok_h], self.j_down[ok_v]])
        data = np.ones(i1.size, dtype=np.int8)
        A = coo_matrix((data, (i1, j1)), shape=(self.M, self.M))
        ncomp, labels = connected_components(A, directed=False)
        flip = rng.random(ncomp) < 0.5
        f = flip[labels].reshape(S.shape)
        self.S = np.where(f, -S, S).astype(np.int8)

    def mags(self):
        N2 = self.N*self.N
        m = self.S.reshape(self.R, -1).sum(axis=1, dtype=np.int64) / N2
        return m.astype(float)

def run(N, R, T, seed, nequil, nmeas):
    sim = SWBatch(N, R, T, seed)
    for _ in range(nequil):
        sim.sweep()
    m2 = np.empty((nmeas, R)); m4 = np.empty((nmeas, R))
    for k in range(nmeas):
        sim.sweep()
        m = sim.mags()
        m2[k] = m*m; m4[k] = m**4
    return m2, m4
'''
open('/work/sw.py','w').write(code)

import sys
sys.path.insert(0,'/work')
import numpy as np, itertools
from sw import run, SWBatch

# validation on 4x4 vs exact enumeration
N=4
configs = np.array(list(itertools.product([-1,1], repeat=N*N)), dtype=np.int64).reshape(-1,N,N)
def energy(S):
    return -(S*np.roll(S,-1,axis=1)).sum(axis=(1,2)) - (S*np.roll(S,-1,axis=2)).sum(axis=(1,2))
E = energy(configs)
m = configs.reshape(len(configs),-1).sum(1)/(N*N)
for T in [1.5, 2.3, 3.5]:
    w = np.exp(-(E-E.min())/T); w/=w.sum()
    m2e=(w*m**2).sum(); m4e=(w*m**4).sum()
    Ue = 1-m4e/(3*m2e**2)
    t=time.time()
    m2s,m4s = run(N, 200, T, 12345, 200, 2000)
    M2=m2s.mean(); M4=m4s.mean()
    print(f"4x4 T={T}: exact <m2>={m2e:.5f} <m4>={m4e:.5f} U={Ue:.5f} | SW <m2>={M2:.5f} <m4>={M4:.5f} U={1-M4/(3*M2**2):.5f}  time {time.time()-t:.2f}s")

# timing N=16
for R in [64, 256]:
    sim = SWBatch(16, R, 2.27, 1)
    t=time.time()
    for _ in range(50): sim.sweep()
    dt=(time.time()-t)/50
    print("N=16 R=",R," time per sweep", dt, " per replica-sweep us", dt/R*1e6)
