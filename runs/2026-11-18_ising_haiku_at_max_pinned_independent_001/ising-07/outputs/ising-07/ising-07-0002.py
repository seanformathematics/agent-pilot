import numpy as np, time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def make_geometry(L):
    N = L*L
    idx = np.arange(N).reshape(L, L)
    right = np.roll(idx, -1, axis=1).ravel()
    down = np.roll(idx, -1, axis=0).ravel()
    src = np.concatenate([np.arange(N), np.arange(N)])
    dst = np.concatenate([right, down])
    return N, src, dst

class SW:
    def __init__(self, L, R, T, seed, init="random"):
        self.L = L
        self.N, self.src, self.dst = make_geometry(L)
        self.R = R
        self.T = T
        self.p = 1.0 - np.exp(-2.0/T)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        if init == "random":
            self.S = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, self.N))
        else:
            self.S = np.ones((R, self.N), dtype=np.int8)
    def sweep(self):
        R, N = self.S.shape
        S = self.S
        s_src = S[:, self.src]
        s_dst = S[:, self.dst]
        same = s_src == s_dst
        occ = same & (self.rng.random(same.shape) < self.p)
        r_idx, b_idx = np.nonzero(occ)
        g_src = r_idx*N + self.src[b_idx]
        g_dst = r_idx*N + self.dst[b_idx]
        n = R*N
        g = csr_matrix((np.ones(g_src.size, dtype=np.int8), (g_src, g_dst)), shape=(n, n))
        ncomp, labels = connected_components(g, directed=False)
        flip = self.rng.random(ncomp) < 0.5
        f = flip[labels].reshape(R, N)
        self.S = np.where(f, -S, S).astype(np.int8)

def measures(S, src, dst):
    M = S.sum(axis=1, dtype=np.int64)
    E = -(S[:, src].astype(np.int64) * S[:, dst]).sum(axis=1)
    return M, E

def exact_L(L, T):
    N = L*L
    confs = np.arange(2**N, dtype=np.int64)
    bits = (confs[:, None] >> np.arange(N)) & 1
    S = (2*bits - 1).astype(np.int64)
    idx = np.arange(N).reshape(L, L)
    right = np.roll(idx, -1, axis=1).ravel()
    down = np.roll(idx, -1, axis=0).ravel()
    E = -(S*S[:, right]).sum(1) - (S*S[:, down]).sum(1)
    M = S.sum(1)
    logw = -E/T
    logw -= logw.max()
    w = np.exp(logw)
    Z = w.sum()
    m2 = (w*M.astype(float)**2).sum()/Z
    m4 = (w*M.astype(float)**4).sum()/Z
    e = (w*E).sum()/Z
    U = 1 - m4/(3*m2**2)
    return m2, m4, e, U

# ---- validation on 4x4 ----
L = 4
N, src, dst = make_geometry(L)
for T in [1.5, 2.2692, 3.0]:
    m2e, m4e, ee, Ue = exact_L(L, T)
    sw = SW(L, R=400, T=T, seed=12345)
    for _ in range(200):
        sw.sweep()
    m2s, m4s, es, Us = [], [], [], []
    nsw = 1500
    acc = np.zeros(3)
    for t in range(nsw):
        sw.sweep()
        M, E = measures(sw.S, src, dst)
        M = M.astype(float)
        acc += [np.mean(M**2), np.mean(M**4), np.mean(E)]
    acc /= nsw
    print(f"L=4 T={T}: exact m2={m2e:.5f} m4={m4e:.4f} e={ee:.5f} U={Ue:.5f} | SW m2={acc[0]:.5f} m4={acc[1]:.4f} e={acc[2]/N:.5f}*N? (E sum {acc[2]:.4f} vs exact {ee:.4f})")

# ---- timing on 16x16 ----
for R in [500, 2000]:
    sw = SW(16, R=R, T=2.27, seed=7)
    t0 = time.time()
    for _ in range(20):
        sw.sweep()
    dt = (time.time()-t0)/20
    print(f"L=16 R={R}: {dt*1e3:.2f} ms per sweep -> {R/dt:.0f} replica-sweeps/s")
