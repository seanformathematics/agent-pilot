
code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class SWReplicas:
    """Swendsen-Wang dynamics for R independent NxN periodic 2D Ising lattices (J=1) at temperature T.
    All replicas are processed in one big sparse graph; each replica is an independent Markov chain."""

    def __init__(self, N, R, T, rng, init="hot"):
        self.N, self.R, self.T, self.rng = N, R, T, rng
        self.p = 1.0 - np.exp(-2.0 / T)          # FK bond probability for aligned neighbours, J=1
        n = R * N * N
        idx = np.arange(n, dtype=np.int32).reshape(R, N, N)
        self.idx = idx
        self.idx_r = np.roll(idx, -1, axis=2)     # neighbour (i, j+1)
        self.idx_d = np.roll(idx, -1, axis=1)     # neighbour (i+1, j)
        if init == "hot":
            self.s = (2 * rng.integers(0, 2, size=(R, N, N), dtype=np.int8) - 1).astype(np.int8)
        elif init == "cold":
            self.s = np.ones((R, N, N), dtype=np.int8)
        else:
            raise ValueError(init)

    def sweep(self):
        s, rng, p = self.s, self.rng, self.p
        n = s.size
        br = (s == np.roll(s, -1, axis=2)) & (rng.random(s.shape) < p)
        bd = (s == np.roll(s, -1, axis=1)) & (rng.random(s.shape) < p)
        rows = np.concatenate((self.idx[br], self.idx[bd]))
        cols = np.concatenate((self.idx_r[br], self.idx_d[bd]))
        g = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
        nc, lab = connected_components(g, directed=False)
        flip = rng.random(nc) < 0.5                # each cluster flipped with prob 1/2
        sf = s.reshape(-1)
        sf[flip[lab]] *= -1

    def magnetization_sum(self):
        return self.s.sum(axis=(1, 2), dtype=np.int64)     # M per replica

    def energy(self):
        s = self.s
        return -(s.astype(np.int32) * (np.roll(s, -1, axis=2) + np.roll(s, -1, axis=1))).sum(axis=(1, 2))
'''
open('/work/sw.py','w').write(code)

import time, numpy as np, sys
sys.path.insert(0,'/work')
from sw import SWReplicas
for (N,R,T) in [(16,1000,2.269),(16,1000,1.5),(16,1000,2.8)]:
    rng = np.random.default_rng(12345)
    sim = SWReplicas(N,R,T,rng,'hot')
    t=time.time()
    nsw=30
    for _ in range(nsw):
        sim.sweep()
    dt=(time.time()-t)/nsw
    M = sim.magnetization_sum()/N**2
    print(f"N={N} R={R} T={T}: {dt*1e3:.1f} ms/sweep -> {dt/R*1e6:.1f} us per replica-sweep; <m^2>={np.mean(M**2):.4f}")
