import os
os.makedirs("/work/data", exist_ok=True)
code = r'''
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def make_tables(N):
    idx = np.arange(N*N).reshape(N, N)
    right = np.roll(idx, -1, axis=1).ravel()   # (i,j) -> (i,j+1)
    down = np.roll(idx, -1, axis=0).ravel()    # (i,j) -> (i+1,j)
    return right, down

class SW:
    """Vectorised Swendsen-Wang for R independent N x N Ising replicas (J=1, periodic)."""
    def __init__(self, N, R, T, rng, init="random"):
        self.N = N; self.NN = N*N; self.R = R; self.T = T; self.rng = rng
        self.right, self.down = make_tables(N)
        self.pb = 1.0 - np.exp(-2.0/T)
        if init == "random":
            self.S = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, self.NN))
        else:
            self.S = np.ones((R, self.NN), dtype=np.int8)
    def sweep(self):
        R, NN = self.R, self.NN
        S = self.S
        ar = (S == S[:, self.right]) & (self.rng.random((R, NN)) < self.pb)
        ad = (S == S[:, self.down]) & (self.rng.random((R, NN)) < self.pb)
        rr, aa = np.nonzero(ar)
        rd, dd = np.nonzero(ad)
        src = np.concatenate([rr*NN + aa, rd*NN + dd])
        dst = np.concatenate([rr*NN + self.right[aa], rd*NN + self.down[dd]])
        n = R*NN
        G = coo_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n))
        ncomp, lab = connected_components(G, directed=False)
        flip = self.rng.random(ncomp) < 0.5
        f = flip[lab].reshape(R, NN)
        self.S = np.where(f, -S, S).astype(np.int8)
    def m(self):
        return self.S.sum(axis=1, dtype=np.int64) / self.NN

def exact_U(N, T):
    """Exact enumeration (only feasible for small N): returns <m^2>, <m^4>, U."""
    NN = N*N
    states = np.arange(2**NN, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(NN)) & 1).astype(np.int8)*2 - 1
    idx = np.arange(NN).reshape(N, N)
    right = np.roll(idx, -1, axis=1).ravel(); down = np.roll(idx, -1, axis=0).ravel()
    E = -(bits * bits[:, right]).sum(axis=1) - (bits * bits[:, down]).sum(axis=1)
    E = E.astype(float)
    w = np.exp(-(E - E.min())/T)
    m = bits.sum(axis=1)/NN
    Z = w.sum()
    m2 = (w*m**2).sum()/Z; m4 = (w*m**4).sum()/Z
    return m2, m4, 1 - m4/(3*m2**2)
'''
with open("/work/sw.py", "w") as f:
    f.write(code)

import sys, time
sys.path.insert(0, "/work")
import numpy as np
from sw import SW, exact_U

# 1) validation against exact enumeration on 4x4
for T in [1.5, 2.0, 2.4, 3.0]:
    m2, m4, U = exact_U(4, T)
    rng = np.random.default_rng(12345)
    s = SW(4, 2000, T, rng)
    M2 = []; M4 = []
    for t in range(300):
        s.sweep()
        if t >= 50:
            m = s.m(); M2.append((m**2)); M4.append(m**4)
    M2 = np.array(M2); M4 = np.array(M4)
    # per-replica chain averages
    c2 = M2.mean(axis=0); c4 = M4.mean(axis=0)
    B2 = c2.mean(); B4 = c4.mean(); Ust = 1 - B4/(3*B2**2)
    print(f"4x4 T={T}: exact <m2>={m2:.5f} <m4>={m4:.5f} U={U:.5f} | SW <m2>={B2:.5f} <m4>={B4:.5f} U={Ust:.5f}")

# 2) timing on 16x16
for R in [1000, 4000]:
    rng = np.random.default_rng(1)
    s = SW(16, R, 2.27, rng)
    for _ in range(3):
        s.sweep()
    t0 = time.time()
    for _ in range(10):
        s.sweep()
    dt = (time.time()-t0)/10
    print(f"16x16 R={R}: {dt*1000:.1f} ms per sweep -> {R/dt:.0f} replica-sweeps/s")
