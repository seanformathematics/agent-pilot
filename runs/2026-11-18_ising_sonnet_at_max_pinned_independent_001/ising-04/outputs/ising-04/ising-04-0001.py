
import sys, time, os
import numpy as np, scipy
print(sys.version)
print("numpy", np.__version__, "scipy", scipy.__version__)
try:
    import numba
    print("numba available", numba.__version__)
except Exception as e:
    print("no numba:", e)
os.makedirs("/work", exist_ok=True)

code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class BatchSW:
    """R independent N x N periodic Ising lattices updated simultaneously with Swendsen-Wang.
    Weight exp(-E/T), E = -sum_<ij> s_i s_j  ->  bond probability p = 1 - exp(-2/T)."""
    def __init__(self, N, R, seed):
        self.N, self.R, self.V = N, R, N * N
        self.rng = np.random.default_rng(seed)
        n = R * N * N
        self.n = n
        idx = np.arange(n, dtype=np.int32).reshape(R, N, N)
        self.idx = idx
        self.idx_r = np.roll(idx, -1, axis=2)   # right neighbour (periodic)
        self.idx_d = np.roll(idx, -1, axis=1)   # down neighbour (periodic)
        self.rep_of_node = (np.arange(n) // (N * N)).astype(np.int32)

    def init_hot(self):
        return (2 * self.rng.integers(0, 2, size=(self.R, self.N, self.N), dtype=np.int8) - 1).astype(np.int8)

    def init_cold(self):
        return np.ones((self.R, self.N, self.N), dtype=np.int8)

    def measure(self, s):
        m = s.reshape(self.R, -1).sum(axis=1, dtype=np.int64) / self.V
        return m * m, m ** 4

    def step(self, s, p):
        """one SW update. returns (new_s, improved_m2, improved_m4) where the improved estimators
        are evaluated on the FK clusters generated from the *input* configuration s."""
        rng, R, N, n = self.rng, self.R, self.N, self.n
        bh = (s == np.roll(s, -1, axis=2)) & (rng.random(s.shape) < p)
        bv = (s == np.roll(s, -1, axis=1)) & (rng.random(s.shape) < p)
        a = np.concatenate((self.idx[bh], self.idx[bv]))
        b = np.concatenate((self.idx_r[bh], self.idx_d[bv]))
        g = csr_matrix((np.ones(a.size, dtype=np.float64), (a, b)), shape=(n, n))
        ncomp, labels = connected_components(g, directed=False)
        sizes = np.bincount(labels, minlength=ncomp).astype(np.float64)
        rep_c = np.empty(ncomp, dtype=np.int32)
        rep_c[labels] = self.rep_of_node
        S2 = np.bincount(rep_c, weights=sizes ** 2, minlength=R)
        S4 = np.bincount(rep_c, weights=sizes ** 4, minlength=R)
        V = float(self.V)
        im2 = S2 / V ** 2
        im4 = (3.0 * S2 ** 2 - 2.0 * S4) / V ** 4
        flip = rng.random(ncomp) < 0.5
        sign = np.where(flip, -1, 1).astype(np.int8)
        s_new = (s.reshape(-1) * sign[labels]).reshape(s.shape)
        return s_new, im2, im4
'''
with open("/work/sw.py", "w") as f:
    f.write(code)

sys.path.insert(0, "/work")
from sw import BatchSW
for R in (500, 1000, 2000):
    sim = BatchSW(16, R, 123)
    s = sim.init_hot()
    p = 1 - np.exp(-2 / 2.27)
    t = time.time()
    nst = 10
    for _ in range(nst):
        s, a, b = sim.step(s, p)
    dt = (time.time() - t) / nst
    print(f"R={R}: {dt*1e3:.1f} ms/step, {dt/R*1e6:.1f} us per replica-step")
