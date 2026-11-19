
import sys, os, time
import numpy as np, scipy
print(sys.version)
print("numpy", np.__version__, "scipy", scipy.__version__)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba:", e)
print("cpu count", os.cpu_count())

code = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class SwendsenWang:
    """Swendsen-Wang cluster algorithm for R independent NxN Ising replicas (J=1, periodic BC).
    All replicas are stored in one (R,N,N) int8 array; the bond graph of all replicas is one big
    disjoint graph, so one connected_components call labels all clusters of all replicas."""

    def __init__(self, N, R, T, seed, start="hot"):
        self.N = N
        self.R = R
        self.n = N * N
        self.T = float(T)
        self.p = 1.0 - np.exp(-2.0 / self.T)       # bond probability for aligned neighbours
        self.rng = np.random.default_rng(seed)
        M = R * self.n
        self.M = M
        idx = np.arange(M, dtype=np.int32).reshape(R, N, N)
        self.right = np.roll(idx, -1, axis=2).ravel()   # site (i, j+1)
        self.down = np.roll(idx, -1, axis=1).ravel()    # site (i+1, j)
        self.rep_of_node = np.repeat(np.arange(R, dtype=np.int64), self.n)
        if start == "hot":
            self.s = (self.rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
        elif start == "cold":
            self.s = np.ones((R, N, N), dtype=np.int8)
        else:
            raise ValueError(start)

    def sweep(self, measure=True, energy=False):
        R, N, n, M, p, rng = self.R, self.N, self.n, self.M, self.p, self.rng
        s = self.s
        sr = np.roll(s, -1, axis=2)
        sd = np.roll(s, -1, axis=1)
        u = rng.random((2, M))
        b1 = (s == sr).ravel() & (u[0] < p)
        b2 = (s == sd).ravel() & (u[1] < p)
        cnt = b1.astype(np.int32) + b2
        indptr = np.zeros(M + 1, dtype=np.int32)
        np.cumsum(cnt, out=indptr[1:])
        nnz = int(indptr[-1])
        indices = np.empty(nnz, dtype=np.int32)
        starts = indptr[:-1]
        indices[starts[b1]] = self.right[b1]
        indices[(starts + b1)[b2]] = self.down[b2]
        data = np.ones(nnz, dtype=np.float64)
        g = csr_matrix((data, indices, indptr), shape=(M, M))
        ncomp, labels = connected_components(g, directed=False)
        sign = (1 - 2 * rng.integers(0, 2, size=ncomp, dtype=np.int8)).astype(np.int8)
        snew = (s.ravel() * sign[labels]).reshape(R, N, N)
        self.s = snew
        if not measure:
            return None
        out = {}
        Mtot = snew.reshape(R, n).sum(axis=1, dtype=np.int64) / n
        m2 = Mtot ** 2
        out["m2"] = m2
        out["m4"] = m2 * m2
        # improved (cluster) estimators: given clusters, spins are iid +-1 per cluster
        counts = np.bincount(labels, minlength=ncomp).astype(np.float64)
        comp_rep = np.empty(ncomp, dtype=np.int64)
        comp_rep[labels] = self.rep_of_node
        c2 = counts * counts
        S2 = np.bincount(comp_rep, weights=c2, minlength=R)
        S4 = np.bincount(comp_rep, weights=c2 * c2, minlength=R)
        out["m2i"] = S2 / n ** 2
        out["m4i"] = (3.0 * S2 ** 2 - 2.0 * S4) / n ** 4
        if energy:
            e = -((snew * np.roll(snew, -1, axis=2)).sum(axis=(1, 2), dtype=np.int64)
                  + (snew * np.roll(snew, -1, axis=1)).sum(axis=(1, 2), dtype=np.int64)) / n
            out["e"] = e
        return out


def binder(A, B):
    return 1.0 - B / (3.0 * A * A)


def jackknife_U(a, b, G):
    """a,b: per-replica means of m^2 and m^4 (equal weights). Delete-group jackknife over G groups of replicas.
    Returns U (full-sample), jackknife error."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    R = len(a)
    assert R % G == 0
    k = R // G
    ag = a.reshape(G, k).sum(axis=1); bg = b.reshape(G, k).sum(axis=1)
    A = a.sum(); B = b.sum()
    U = binder(A / R, B / R)
    Uj = binder((A - ag) / (R - k), (B - bg) / (R - k))
    err = np.sqrt((G - 1) / G * np.sum((Uj - Uj.mean()) ** 2))
    return U, err
'''
os.makedirs('/work', exist_ok=True)
with open('/work/sw.py', 'w') as f:
    f.write(code)
print("module written")
