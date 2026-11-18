
code = r'''
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

N = 16
S = N * N
_x = np.arange(S) // N
_y = np.arange(S) % N
_right = _x * N + (_y + 1) % N
_down = ((_x + 1) % N) * N + _y
_base_src = np.concatenate([np.arange(S), np.arange(S)])
_base_dst = np.concatenate([_right, _down])

class SW:
    def __init__(self, R, T, rng):
        self.R = R; self.T = T; self.rng = rng
        self.n = R * S
        off = (np.arange(R) * S)[:, None]
        self.src = (off + _base_src[None, :]).ravel()
        self.dst = (off + _base_dst[None, :]).ravel()
        self.p = 1 - np.exp(-2.0 / T)
        self.s = np.where(rng.random(self.n) < 0.5, 1, -1).astype(np.int8)
    def sweep(self):
        s = self.s
        same = s[self.src] == s[self.dst]
        r = self.rng.random(self.src.size) < self.p
        m = same & r
        g = coo_matrix((np.ones(int(m.sum()), dtype=np.int8), (self.src[m], self.dst[m])), shape=(self.n, self.n))
        nc, lab = connected_components(g, directed=False)
        flip = self.rng.random(nc) < 0.5
        self.s = (s * (1 - 2 * flip[lab].astype(np.int8))).astype(np.int8)
    def mags(self):
        return self.s.reshape(self.R, S).mean(axis=1, dtype=np.float64)

def run(T, seed, R, burn, L, block=10):
    rng = np.random.default_rng(seed)
    sim = SW(R, T, rng)
    for _ in range(burn):
        sim.sweep()
    nb = L // block
    m2b = np.zeros((nb, R)); m4b = np.zeros((nb, R))
    for b in range(nb):
        acc2 = np.zeros(R); acc4 = np.zeros(R)
        for k in range(block):
            sim.sweep()
            m = sim.mags()
            m2 = m * m
            acc2 += m2; acc4 += m2 * m2
        m2b[b] = acc2 / block; m4b[b] = acc4 / block
    return m2b, m4b

def binder_from_chain_sums(M2, M4):
    # M2, M4: arrays (R,) of chain means of m^2, m^4
    m2 = M2.mean(); m4 = M4.mean()
    return 1 - m4 / (3 * m2 * m2)

def jackknife_U(M2, M4):
    R = len(M2)
    U = binder_from_chain_sums(M2, M4)
    Ui = np.empty(R)
    tot2 = M2.sum(); tot4 = M4.sum()
    for i in range(R):
        a2 = (tot2 - M2[i]) / (R - 1); a4 = (tot4 - M4[i]) / (R - 1)
        Ui[i] = 1 - a4 / (3 * a2 * a2)
    err = np.sqrt((R - 1) / R * np.sum((Ui - Ui.mean()) ** 2))
    return U, err
'''
open('/work/ising_sw.py', 'w').write(code)

import sys, time
sys.path.insert(0, '/work')
import numpy as np
import ising_sw as M

results = {}
t0 = time.time()
for T in [1.5, 2.27, 2.8]:
    t = time.time()
    m2b, m4b = M.run(T, seed=1000 + int(T * 100), R=256, burn=300, L=1500, block=10)
    M2 = m2b.mean(0); M4 = m4b.mean(0)
    U, err = M.jackknife_U(M2, M4)
    # first/second half consistency
    h = m2b.shape[0] // 2
    U1, e1 = M.jackknife_U(m2b[:h].mean(0), m4b[:h].mean(0))
    U2, e2 = M.jackknife_U(m2b[h:].mean(0), m4b[h:].mean(0))
    print(f"T={T}: U={U:.5f} err={err:.5f}  halves: {U1:.5f}({e1:.5f}) {U2:.5f}({e2:.5f})  time={time.time()-t:.1f}s", flush=True)
    np.savez(f'/work/pilot_T{T}.npz', m2b=m2b, m4b=m4b)
print("total", time.time() - t0)
