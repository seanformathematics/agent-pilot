
code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class SW:
    """Swendsen-Wang for the 2D Ising model (J=1, periodic BC), vectorised over R independent replicas."""
    def __init__(self, N, R, T, rng):
        self.N, self.R, self.T, self.rng = N, R, T, rng
        n = N * N
        self.n = n
        self.p = 1.0 - np.exp(-2.0 / T)
        idx = np.arange(R * n, dtype=np.int32).reshape(R, N, N)
        right = np.roll(idx, -1, axis=2).reshape(-1)
        down = np.roll(idx, -1, axis=1).reshape(-1)
        self.nbr = np.stack([right, down], axis=1)            # (R*n, 2) neighbours (+j, +i)
        self.rep_of_node = np.repeat(np.arange(R, dtype=np.int32), n)

    def init_state(self, start):
        R, N = self.R, self.N
        if start == "cold":
            return np.ones((R, N, N), dtype=np.int8)
        elif start == "hot":
            return (self.rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
        raise ValueError(start)

    def step(self, s):
        """One SW sweep. Returns new spins and per-replica improved estimators (m2_imp, m4_imp)."""
        R, N, n, rng = self.R, self.N, self.n, self.rng
        sr = np.roll(s, -1, axis=2)
        sd = np.roll(s, -1, axis=1)
        u = rng.random((R, N, N, 2))
        same = np.stack([s == sr, s == sd], axis=-1)
        mask = (same & (u < self.p)).reshape(-1, 2)
        indices = self.nbr[mask]
        indptr = np.zeros(R * n + 1, dtype=np.int32)
        np.cumsum(mask.sum(axis=1, dtype=np.int32), out=indptr[1:])
        data = np.ones(indices.shape[0], dtype=np.float64)
        G = csr_matrix((data, indices, indptr), shape=(R * n, R * n))
        ncomp, labels = connected_components(G, directed=True, connection="weak")
        sizes = np.bincount(labels, minlength=ncomp).astype(np.float64)
        comp_rep = np.empty(ncomp, dtype=np.int32)
        comp_rep[labels] = self.rep_of_node
        x = np.bincount(comp_rep, weights=sizes ** 2, minlength=R) / n ** 2
        y = np.bincount(comp_rep, weights=sizes ** 4, minlength=R) / n ** 4
        m2_imp = x
        m4_imp = 3.0 * x * x - 2.0 * y
        eps = (rng.integers(0, 2, size=ncomp, dtype=np.int8) * 2 - 1).astype(np.int8)
        s_new = (s.reshape(-1) * eps[labels]).reshape(R, N, N)
        return s_new, m2_imp, m4_imp

def plain_moments(s):
    n = s.shape[1] * s.shape[2]
    m = s.sum(axis=(1, 2), dtype=np.int32) / n
    m2 = m * m
    return m2, m2 * m2

def U_from_sums(a2, a4, cnt):
    m2 = a2 / cnt
    m4 = a4 / cnt
    return 1.0 - m4 / (3.0 * m2 ** 2)

def jackknife_U(S2, S4, n_t):
    """S2, S4: per-replica (independent chain) sums over n_t measurements each."""
    R = len(S2)
    t2, t4 = S2.sum(), S4.sum()
    U_full = U_from_sums(t2, t4, R * n_t)
    U_loo = U_from_sums(t2 - S2, t4 - S4, (R - 1) * n_t)
    U_bar = U_loo.mean()
    var = (R - 1) / R * np.sum((U_loo - U_bar) ** 2)
    return U_full, np.sqrt(var), R * U_full - (R - 1) * U_bar

def run_sw(N, T, R, n_burn, n_meas, seed, start="cold", traj=False):
    rng = np.random.default_rng(seed)
    sim = SW(N, R, T, rng)
    s = sim.init_state(start)
    S2i = np.zeros(R); S4i = np.zeros(R); S2p = np.zeros(R); S4p = np.zeros(R)
    tr = []
    for t in range(n_burn + n_meas):
        s, m2i, m4i = sim.step(s)
        if traj:
            m2p_, _ = plain_moments(s)
            tr.append((m2i.mean(), m2p_.mean()))
        if t >= n_burn:
            m2p, m4p = plain_moments(s)
            S2i += m2i; S4i += m4i; S2p += m2p; S4p += m4p
    out = dict(S2i=S2i, S4i=S4i, S2p=S2p, S4p=S4p, n_t=n_meas, R=R, T=T, N=N, seed=seed)
    if traj:
        out["traj"] = np.array(tr)
    return out
'''
open("/work/sw.py", "w").write(code)

import sys, time
sys.path.insert(0, "/work")
import importlib, numpy as np
import sw
importlib.reload(sw)

# ---------- exact enumeration for N=4 ----------
N = 4
nn = N * N
confs = np.arange(2 ** nn, dtype=np.uint32)
bits = ((confs[:, None] >> np.arange(nn, dtype=np.uint32)[None, :]) & 1).astype(np.int8) * 2 - 1
S = bits.reshape(-1, N, N)
E = -(S * np.roll(S, -1, axis=2)).sum(axis=(1, 2)) - (S * np.roll(S, -1, axis=1)).sum(axis=(1, 2))
M = S.sum(axis=(1, 2)) / nn
def exact_U(T):
    w = np.exp(-(E - E.min()) / T)
    Z = w.sum()
    m2 = (w * M ** 2).sum() / Z
    m4 = (w * M ** 4).sum() / Z
    return 1 - m4 / (3 * m2 ** 2), m2, m4

for T in [1.5, 2.3, 3.0]:
    Ue, m2e, m4e = exact_U(T)
    for start in ["cold", "hot"]:
        t0 = time.time()
        o = sw.run_sw(4, T, R=400, n_burn=50, n_meas=500, seed=1, start=start)
        Ui, eri, _ = sw.jackknife_U(o["S2i"], o["S4i"], o["n_t"])
        Up, erp, _ = sw.jackknife_U(o["S2p"], o["S4p"], o["n_t"])
        print(f"N=4 T={T} {start}: exact U={Ue:.5f} | improved {Ui:.5f}+-{eri:.5f} (z={(Ui-Ue)/eri:+.2f}) | plain {Up:.5f}+-{erp:.5f} (z={(Up-Ue)/erp:+.2f})  [{time.time()-t0:.1f}s]")
