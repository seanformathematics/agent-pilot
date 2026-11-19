
code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class SWBatch:
    """R independent NxN periodic Ising replicas updated simultaneously with Swendsen-Wang."""
    def __init__(self, N, R, T, seed, start="hot"):
        self.N, self.R, self.T = N, R, float(T)
        self.rng = np.random.default_rng(seed)
        n = R * N * N
        self.n = n
        if start == "hot":
            self.s = (self.rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
        elif start == "cold":
            self.s = np.ones((R, N, N), dtype=np.int8)
        else:
            raise ValueError(start)
        idx = np.arange(n, dtype=np.int32).reshape(R, N, N)
        self.idx = idx
        self.idx_r = np.roll(idx, -1, axis=2)
        self.idx_d = np.roll(idx, -1, axis=1)
        self.p = 1.0 - np.exp(-2.0 / self.T)
        self.rep_of_node = np.repeat(np.arange(R), N * N)
        self.ones = None

    def step(self):
        """One SW update. Returns improved-estimator per-replica quantities (S2, S4) computed from the clusters
        of this update: E[m^2|bonds]=S2/N^4, E[m^4|bonds]=(3 S2^2-2 S4)/N^8."""
        s = self.s
        rng = self.rng
        R, N = self.R, self.N
        p = self.p
        br = (s == np.roll(s, -1, axis=2)) & (rng.random(s.shape) < p)
        bd = (s == np.roll(s, -1, axis=1)) & (rng.random(s.shape) < p)
        rows = np.concatenate([self.idx[br], self.idx[bd]])
        cols = np.concatenate([self.idx_r[br], self.idx_d[bd]])
        data = np.ones(rows.shape[0], dtype=np.int8)
        g = csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
        ncomp, labels = connected_components(g, directed=False)
        sizes = np.bincount(labels, minlength=ncomp).astype(np.float64)
        rep_of_comp = np.empty(ncomp, dtype=np.int64)
        rep_of_comp[labels] = self.rep_of_node
        sz2 = sizes * sizes
        S2 = np.bincount(rep_of_comp, weights=sz2, minlength=R)
        S4 = np.bincount(rep_of_comp, weights=sz2 * sz2, minlength=R)
        flip = rng.random(ncomp) < 0.5
        sf = s.reshape(-1)
        fl = flip[labels]
        self.s = np.where(fl, -sf, sf).astype(np.int8).reshape(R, N, N)
        return S2, S4

    def observables(self):
        """Plain observables of the current configuration: m (per replica) and energy per spin."""
        s = self.s.astype(np.int32)
        N = self.N
        m = s.sum(axis=(1, 2)) / float(N * N)
        e = -(s * (np.roll(s, -1, axis=2) + np.roll(s, -1, axis=1))).sum(axis=(1, 2)) / float(N * N)
        return m, e
'''
open('/work/sw.py','w').write(code)

import time, sys
sys.path.insert(0,'/work')
import importlib, sw
importlib.reload(sw)
import numpy as np
for R in [256, 1024]:
    for T in [1.5, 2.269, 2.8]:
        b = sw.SWBatch(16, R, T, seed=1, start="hot")
        for _ in range(5): b.step()
        t = time.time()
        nst = 20
        for _ in range(nst):
            S2,S4 = b.step()
            m,e = b.observables()
        dt = (time.time()-t)/nst
        print(f"R={R} T={T}: {dt*1e3:.1f} ms/step  -> {R/dt:.0f} replica-updates/s; mean m2={np.mean(m**2):.4f} E={e.mean():.4f}")
