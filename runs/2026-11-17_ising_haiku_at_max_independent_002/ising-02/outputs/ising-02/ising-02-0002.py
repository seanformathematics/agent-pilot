import os, time
os.makedirs('/work', exist_ok=True)
src = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class SW:
    """Vectorized Swendsen-Wang over R independent replicas of an N x N periodic Ising lattice (J=1)."""
    def __init__(self, N, R, T, seed):
        self.N = N; self.R = R; self.T = T
        self.n = R * N * N
        self.rng = np.random.default_rng(seed)
        k = np.arange(self.n, dtype=np.int64)
        r = k // (N * N); rem = k % (N * N); i = rem // N; j = rem % N
        self.k = k
        self.nb_right = r * (N * N) + i * N + (j + 1) % N
        self.nb_down = r * (N * N) + ((i + 1) % N) * N + j
        self.p = 1.0 - np.exp(-2.0 / T)   # bond activation prob. for parallel spins
        self.s = self.rng.random(self.n) < 0.5  # bool, True = +1

    def sweep(self):
        s = self.s; n = self.n
        er = (s == s[self.nb_right]) & (self.rng.random(n) < self.p)
        ed = (s == s[self.nb_down]) & (self.rng.random(n) < self.p)
        u = np.concatenate([self.k[er], self.k[ed]])
        v = np.concatenate([self.nb_right[er], self.nb_down[ed]])
        g = csr_matrix((np.ones(u.size, dtype=np.int8), (u, v)), shape=(n, n))
        ncomp, lab = connected_components(g, directed=False)
        flip = self.rng.random(ncomp) < 0.5
        self.s = s ^ flip[lab]

    def mag_sum(self):
        """Per-replica sum of spins M, int64 array of length R."""
        cnt = self.s.reshape(self.R, -1).sum(axis=1).astype(np.int64)
        return 2 * cnt - self.N * self.N

    def energy(self):
        """Per-replica energy E = -(sum over nearest-neighbour pairs of s_i s_j), float array."""
        sp = np.where(self.s, 1.0, -1.0)
        e = -(sp * sp[self.nb_right]).reshape(self.R, -1).sum(axis=1) - (sp * sp[self.nb_down]).reshape(self.R, -1).sum(axis=1)
        return e
'''
with open('/work/sw.py', 'w') as f:
    f.write(src)

import sys
sys.path.insert(0, '/work')
import numpy as np
from sw import SW

def exact_small(T, N=4):
    nconf = 1 << (N * N)
    bits = (np.arange(nconf, dtype=np.int64)[:, None] >> np.arange(N * N)) & 1
    s = (2 * bits - 1).reshape(nconf, N, N).astype(np.float64)
    E = -(s * np.roll(s, -1, axis=2)).sum(axis=(1, 2)) - (s * np.roll(s, -1, axis=1)).sum(axis=(1, 2))
    M = s.sum(axis=(1, 2))
    w = np.exp(-(E - E.min()) / T)
    w /= w.sum()
    m2 = (w * M**2).sum(); m4 = (w * M**4).sum()
    return 1 - m4 / (3 * m2**2), m2, m4

for T in [1.5, 2.0, 2.5, 3.0]:
    Ue, m2e, m4e = exact_small(T)
    sim = SW(4, 4000, T, seed=12345 + int(T * 10))
    for _ in range(100):
        sim.sweep()
    S2 = 0.0; S4 = 0.0; E1 = 0.0; cnt = 0
    for t in range(300):
        sim.sweep()
        M = sim.mag_sum().astype(np.float64)
        S2 += (M**2).sum(); S4 += (M**4).sum(); cnt += 1
        E1 += sim.energy().sum()
    m2 = S2 / (cnt * sim.R); m4 = S4 / (cnt * sim.R)
    U = 1 - m4 / (3 * m2**2)
    print("T=%.2f exact U=%.5f M2avg=%.4f M4avg=%.3f | SW U=%.5f M2avg=%.4f M4avg=%.3f Eavg=%.4f" % (T, Ue, m2e, m4e, U, m2, m4, E1 / (cnt * sim.R)))

for R in [1024, 4096]:
    sim = SW(16, R, 2.3, seed=7)
    t0 = time.time()
    for _ in range(10):
        sim.sweep()
    dt = (time.time() - t0) / 10
    print("16x16 R=%d: %.1f ms per SW sweep, %.2f us per replica-sweep" % (R, dt * 1000, dt / R * 1e6))
