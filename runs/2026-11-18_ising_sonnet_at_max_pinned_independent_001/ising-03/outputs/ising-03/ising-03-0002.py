
code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class IsingSW:
    """Swendsen-Wang for the 2D Ising model (J=1, kB=1), many independent replicas at once.
    Replicas never interact: the batched graph has no edges between replicas."""

    def __init__(self, N, R, T, seed, start="hot"):
        self.N, self.R, self.T = N, R, T
        self.rng = np.random.default_rng(seed)
        self.V = R * N * N
        idx = np.arange(self.V, dtype=np.int32).reshape(R, N, N)
        self.idx = idx
        self.right = np.roll(idx, -1, axis=2)
        self.down = np.roll(idx, -1, axis=1)
        self.p = 1.0 - np.exp(-2.0 / T)
        self.indptr = np.arange(0, 2 * self.V + 1, 2, dtype=np.int32)
        self.data = np.ones(2 * self.V)
        self.rep_of_node = np.repeat(np.arange(R, dtype=np.int32), N * N)
        if start == "hot":
            self.s = (2 * self.rng.integers(0, 2, (R, N, N), dtype=np.int8) - 1).astype(np.int8)
        elif start == "cold":
            self.s = np.ones((R, N, N), dtype=np.int8)
        elif start == "mixed":   # half hot, half cold
            s = (2 * self.rng.integers(0, 2, (R, N, N), dtype=np.int8) - 1).astype(np.int8)
            s[R // 2:] = 1
            self.s = s
        else:
            raise ValueError(start)
        self.ind = np.empty((self.V, 2), dtype=np.int32)

    def sweep(self, improved=False):
        """One SW update of all replicas. If improved: return per-replica improved
        estimators (m2, m4) computed from cluster sizes (conditional expectation over the
        random cluster flips)."""
        N, R, V = self.N, self.R, self.V
        s = self.s
        rng = self.rng
        u = rng.random((2, R, N, N), dtype=np.float32)
        bh = (s == np.roll(s, -1, axis=2)) & (u[0] < self.p)
        bv = (s == np.roll(s, -1, axis=1)) & (u[1] < self.p)
        ind = self.ind
        ind[:, 0] = np.where(bh, self.right, self.idx).ravel()
        ind[:, 1] = np.where(bv, self.down, self.idx).ravel()
        g = csr_matrix((self.data, ind.ravel(), self.indptr), shape=(V, V))
        nc, lab = connected_components(g, directed=False)
        flip = rng.integers(0, 2, nc, dtype=np.int8)
        sign = (1 - 2 * flip)[lab].astype(np.int8)
        self.s = (s.reshape(-1) * sign).reshape(R, N, N)
        if improved:
            sizes = np.bincount(lab, minlength=nc).astype(np.float64)
            rep = np.empty(nc, dtype=np.int32)
            rep[lab] = self.rep_of_node
            S2 = np.bincount(rep, weights=sizes ** 2, minlength=R)
            S4 = np.bincount(rep, weights=sizes ** 4, minlength=R)
            n2 = float(N * N)
            m2 = S2 / n2 ** 2
            m4 = (3.0 * S2 ** 2 - 2.0 * S4) / n2 ** 4
            return m2, m4
        return None

    def magnetization(self):
        return self.s.sum(axis=(1, 2), dtype=np.int32) / float(self.N ** 2)

    def energy_per_site(self):
        s = self.s.astype(np.int16)
        e = -(s * np.roll(s, -1, axis=2) + s * np.roll(s, -1, axis=1)).sum(axis=(1, 2))
        return e / float(self.N ** 2)


class IsingMetropolis:
    """Checkerboard Metropolis, many independent replicas (used only as an independent cross-check)."""

    def __init__(self, N, R, T, seed, start="hot"):
        assert N % 2 == 0
        self.N, self.R, self.T = N, R, T
        self.rng = np.random.default_rng(seed)
        if start == "hot":
            self.s = (2 * self.rng.integers(0, 2, (R, N, N), dtype=np.int8) - 1).astype(np.int8)
        else:
            self.s = np.ones((R, N, N), dtype=np.int8)
        ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        self.mask = [((ii + jj) % 2 == c) for c in (0, 1)]
        # acceptance prob for dE = 2*s*nn, nn in {-4,-2,0,2,4}: s*nn in {-4,..,4}
        self.acc = {k: min(1.0, np.exp(-2.0 * k / T)) for k in (-4, -2, 0, 2, 4)}

    def sweep(self):
        s = self.s
        R, N = self.R, self.N
        for c in (0, 1):
            nn = (np.roll(s, 1, 1).astype(np.int8) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2))
            k = s * nn
            pacc = np.where(k <= 0, 1.0, np.exp(-2.0 * k.astype(np.float64) / self.T))
            u = self.rng.random((R, N, N))
            flip = (u < pacc) & self.mask[c][None]
            s = np.where(flip, -s, s).astype(np.int8)
        self.s = s

    def magnetization(self):
        return self.s.sum(axis=(1, 2), dtype=np.int32) / float(self.N ** 2)
'''
open('/work/sw.py','w').write(code)

import sys, time
sys.path.insert(0,'/work')
import numpy as np
import importlib, sw
importlib.reload(sw)

for T in (1.5, 2.27, 2.8):
    for R in (200, 1000):
        S = sw.IsingSW(16, R, T, seed=1)
        for _ in range(5): S.sweep()
        t=time.time()
        n=20
        for _ in range(n): S.sweep()
        dt=(time.time()-t)/n
        t=time.time()
        for _ in range(n): S.sweep(improved=True)
        dt2=(time.time()-t)/n
        print(f"T={T} R={R}: {dt*1e3:.2f} ms/sweep -> {dt/R*1e6:.1f} us/replica-sweep ; improved {dt2/R*1e6:.1f} us")
