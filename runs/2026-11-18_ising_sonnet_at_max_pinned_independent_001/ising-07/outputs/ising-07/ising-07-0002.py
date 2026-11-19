
code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class SWBatch:
    """R independent N x N periodic Ising lattices (J=1, k_B=1), updated simultaneously
    with the Swendsen-Wang cluster algorithm.  Bond probability p = 1 - exp(-2/T)."""

    def __init__(self, N, R, T, seed, start="hot"):
        self.N, self.R, self.T = N, R, float(T)
        self.Ns = N * N
        self.V = R * N * N
        self.p = 1.0 - np.exp(-2.0 / self.T)
        self.rng = np.random.default_rng(seed)
        idx = np.arange(self.V, dtype=np.int32).reshape(R, N, N)
        right = np.roll(idx, -1, axis=2)
        down = np.roll(idx, -1, axis=1)
        self.cand = np.stack([right.ravel(), down.ravel()], axis=1)      # (V,2)
        self.rep_of_node = np.repeat(np.arange(R, dtype=np.int32), N * N)
        self.ones = np.ones(2 * self.V, dtype=np.int8)
        self.indptr = np.zeros(self.V + 1, dtype=np.int32)
        if isinstance(start, str):
            if start == "hot":
                s = self.rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1
            elif start == "cold":
                s = np.ones((R, N, N), dtype=np.int8)
            else:
                raise ValueError(start)
        else:  # array of per-replica flags: 1 = cold, 0 = hot
            flags = np.asarray(start, dtype=bool)
            s = self.rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1
            s[flags] = 1
        self.s = s.astype(np.int8)

    def step(self):
        """One SW sweep.  Returns (M_direct[R], S2[R], S4[R]) where M_direct is the total
        spin after the update, S2=sum_c n_c^2, S4=sum_c n_c^4 over FK clusters."""
        s, R, N, V = self.s, self.R, self.N, self.V
        rng = self.rng
        eq = np.empty((R, N, N, 2), dtype=bool)
        eq[:, :, :-1, 0] = s[:, :, :-1] == s[:, :, 1:]
        eq[:, :, -1, 0] = s[:, :, -1] == s[:, :, 0]
        eq[:, :-1, :, 1] = s[:, :-1, :] == s[:, 1:, :]
        eq[:, -1, :, 1] = s[:, -1, :] == s[:, 0, :]
        u = rng.random((R, N, N, 2), dtype=np.float32)
        act = eq & (u < self.p)
        actf = act.reshape(V, 2)
        cnt = actf.sum(axis=1, dtype=np.int32)
        np.cumsum(cnt, out=self.indptr[1:])
        indices = self.cand[actf]
        g = csr_matrix((self.ones[:indices.size], indices, self.indptr.copy()), shape=(V, V))
        ncomp, labels = connected_components(g, directed=True, connection="weak")
        sizes = np.bincount(labels, minlength=ncomp)
        rep_c = np.empty(ncomp, dtype=np.int32)
        rep_c[labels] = self.rep_of_node
        sz = sizes.astype(np.float64)
        sz2 = sz * sz
        S2 = np.bincount(rep_c, weights=sz2, minlength=R)
        S4 = np.bincount(rep_c, weights=sz2 * sz2, minlength=R)
        flip = (rng.integers(0, 2, size=ncomp, dtype=np.int8) * 2 - 1).astype(np.int8)
        s *= flip[labels].reshape(R, N, N)
        M = s.reshape(R, -1).sum(axis=1, dtype=np.int32)
        return M, S2, S4
'''
open('/work/sw.py','w').write(code)

import sys, time
sys.path.insert(0,'/work')
import numpy as np
from sw import SWBatch
for R in (100, 400, 1000):
    sim = SWBatch(16, R, 2.269, seed=1)
    for _ in range(5): sim.step()
    t=time.time(); n=20
    for _ in range(n): M,S2,S4 = sim.step()
    dt=(time.time()-t)/n
    print("R=%d: %.1f ms/step, %.0f samples/s"%(R, dt*1e3, R/dt))
