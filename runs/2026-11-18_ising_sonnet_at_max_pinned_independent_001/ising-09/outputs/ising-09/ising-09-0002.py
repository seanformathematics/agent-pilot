
import os, time, textwrap
os.makedirs("/work", exist_ok=True)
code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class SW:
    """Swendsen-Wang for 2D Ising (J=1, E=-sum s_i s_j), R independent replicas of an NxN torus."""
    def __init__(self, N, R, seed, start="hot"):
        self.N = N; self.R = R
        self.rng = np.random.default_rng(seed)
        n = R*N*N
        self.n = n
        idx = np.arange(n, dtype=np.int32).reshape(R, N, N)
        self.idx_r = np.ascontiguousarray(np.roll(idx, -1, axis=2)).ravel()
        self.idx_d = np.ascontiguousarray(np.roll(idx, -1, axis=1)).ravel()
        self.rep_of_node = (np.arange(n) // (N*N)).astype(np.int32)
        if start == "hot":
            self.s = (self.rng.integers(0, 2, size=(R, N, N), dtype=np.int8)*2 - 1).astype(np.int8)
        elif start == "cold":
            self.s = np.ones((R, N, N), dtype=np.int8)
        else:
            raise ValueError
    def sweep(self, T, measure_improved=False):
        N, R, n = self.N, self.R, self.n
        rng = self.rng
        p = 1.0 - np.exp(-2.0/T)
        s = self.s
        sf = s.ravel()
        # horizontal / vertical bonds, bond (i, right(i)) and (i, down(i))
        eq_h = (s == np.roll(s, -1, axis=2)).ravel()
        eq_v = (s == np.roll(s, -1, axis=1)).ravel()
        u = rng.random(2*n, dtype=np.float32)
        occ_h = eq_h & (u[:n] < p)
        occ_v = eq_v & (u[n:] < p)
        ih = np.flatnonzero(occ_h)
        iv = np.flatnonzero(occ_v)
        rows = np.concatenate([ih, iv]).astype(np.int32)
        cols = np.concatenate([self.idx_r[ih], self.idx_d[iv]])
        g = csr_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(n, n))
        ncomp, labels = connected_components(g, directed=False)
        bits = rng.integers(0, 2, size=ncomp, dtype=np.int8)
        sign = (bits*2 - 1).astype(np.int8)
        out = None
        if measure_improved:
            sizes = np.bincount(labels, minlength=ncomp).astype(np.float64)
            comp_rep = np.empty(ncomp, dtype=np.int32)
            comp_rep[labels] = self.rep_of_node
            S2 = np.bincount(comp_rep, weights=sizes**2, minlength=R)
            S4 = np.bincount(comp_rep, weights=sizes**4, minlength=R)
            out = (S2/N**4, (3*S2**2 - 2*S4)/N**8)
        self.s = sign[labels].reshape(R, N, N)
        return out
    def mag(self):
        return self.s.reshape(self.R, -1).sum(axis=1, dtype=np.int64)/(self.N*self.N)
'''
open("/work/sw.py","w").write(code)
import sys
sys.path.insert(0,"/work")
import numpy as np
from sw import SW
for T in [1.5, 2.269, 2.8]:
    for R in [500, 2000]:
        sim = SW(16, R, seed=1)
        for _ in range(20): sim.sweep(T)
        t=time.time()
        ns=30
        for _ in range(ns):
            sim.sweep(T)
            m = sim.mag()
        dt=time.time()-t
        print(f"T={T} R={R}: {dt/ns*1e3:.2f} ms/sweep -> {R*ns/dt:.0f} replica-sweeps/s; <m^2>={np.mean(m**2):.4f}")
        t=time.time()
        for _ in range(ns):
            sim.sweep(T, measure_improved=True)
        dt=time.time()-t
        print(f"    with improved estimators: {R*ns/dt:.0f} replica-sweeps/s")
