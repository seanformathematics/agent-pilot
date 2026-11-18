import os, time
os.makedirs('/work', exist_ok=True)
src = r'''
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def neighbors(N):
    idx = np.arange(N * N)
    x = idx % N
    y = idx // N
    right = ((x + 1) % N) + N * y
    up = x + N * ((y + 1) % N)
    return right, up


class SW:
    """Swendsen-Wang updates for R independent replicas of an N x N periodic Ising lattice (J=1)."""

    def __init__(self, N, R, seed):
        self.N = N
        self.NN = N * N
        self.R = R
        self.rng = np.random.default_rng(seed)
        self.right, self.up = neighbors(N)
        self.S = (2 * self.rng.integers(0, 2, size=(R, self.NN)) - 1).astype(np.int8)
        self.V = R * self.NN

    def sweep(self, beta):
        p = -np.expm1(-2.0 * beta)  # 1 - exp(-2 beta)
        S = self.S
        eqR = S == S[:, self.right]
        eqU = S == S[:, self.up]
        aR = eqR & (self.rng.random(S.shape) < p)
        aU = eqU & (self.rng.random(S.shape) < p)
        rr, ii = np.nonzero(aR)
        ru, iu = np.nonzero(aU)
        base_r = rr * self.NN
        base_u = ru * self.NN
        src = np.concatenate([base_r + ii, base_u + iu])
        dst = np.concatenate([base_r + self.right[ii], base_u + self.up[iu]])
        G = sp.csr_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(self.V, self.V))
        ncomp, labels = connected_components(G, directed=False)
        flip = self.rng.random(ncomp) < 0.5
        f = flip[labels].reshape(self.R, self.NN)
        self.S = np.where(f, -S, S).astype(np.int8)

    def mag(self):
        return self.S.sum(axis=1, dtype=np.int64)
'''
with open('/work/sw.py', 'w') as f:
    f.write(src)

import sys
sys.path.insert(0, '/work')
import numpy as np
from sw import SW, neighbors


def exact_U(N, T):
    NN = N * N
    beta = 1.0 / T
    states = np.arange(1 << NN, dtype=np.int64)
    bits = (states[:, None] >> np.arange(NN)) & 1
    s = (2 * bits - 1).astype(np.float64)
    right, up = neighbors(N)
    Eb = (s * s[:, right]).sum(1) + (s * s[:, up]).sum(1)
    M = s.sum(1)
    w = np.exp(beta * (Eb - Eb.max()))
    Z = w.sum()
    m2 = (w * M**2).sum() / Z
    m4 = (w * M**4).sum() / Z
    return 1 - m4 / (3 * m2**2)


def run(N, T, R, nburn, nmeas, seed):
    sim = SW(N, R, seed)
    beta = 1.0 / T
    for _ in range(nburn):
        sim.sweep(beta)
    S2 = np.zeros(R, dtype=np.int64)
    S4 = np.zeros(R, dtype=np.int64)
    S1 = np.zeros(R, dtype=np.int64)
    for _ in range(nmeas):
        sim.sweep(beta)
        M = sim.mag()
        S2 += M * M
        S4 += M**4
        S1 += M
    return S2, S4, S1


def U_jack(S2r, S4r, nr):
    R = len(S2r)
    S2 = S2r.sum(dtype=np.float64)
    S4 = S4r.sum(dtype=np.float64)
    n = nr * R
    U = 1 - (S4 / n) / (3 * (S2 / n) ** 2)
    S2m = S2 - S2r
    S4m = S4 - S4r
    nm = n - nr
    Um = 1 - (S4m / nm) / (3 * (S2m / nm) ** 2)
    err = np.sqrt((R - 1) / R * np.sum((Um - Um.mean()) ** 2))
    return U, err


# validation on 4x4 against exact enumeration
for T in [2.0, 2.5, 3.0]:
    Uex = exact_U(4, T)
    t0 = time.time()
    S2, S4, S1 = run(4, T, 2000, 50, 500, seed=101 + int(T * 10))
    U, err = U_jack(S2, S4, 500)
    print(f"N=4 T={T}: exact U={Uex:.6f}  SW U={U:.6f} +- {err:.6f}  <M>={S1.sum()/(2000*500):.4f}  time={time.time()-t0:.1f}s")

# timing on 16x16
for R in [500, 1000]:
    sim = SW(16, R, seed=7)
    t0 = time.time()
    for _ in range(20):
        sim.sweep(1 / 2.269)
    dt = (time.time() - t0) / 20
    print(f"N=16 R={R}: {dt*1000:.1f} ms per sweep, {dt/R*1e6:.2f} us per replica-sweep")

# quick look at T=2.269 and low T sign-flip behaviour
for T in [1.5, 2.269, 2.8]:
    t0 = time.time()
    S2, S4, S1 = run(16, T, 300, 100, 200, seed=55)
    U, err = U_jack(S2, S4, 200)
    print(f"N=16 T={T}: U={U:.5f} +- {err:.5f}  mean M={S1.sum()/(300*200):.4f} time={time.time()-t0:.1f}s")
