
import sys, os, time
import numpy as np, scipy
print(sys.version)
print("numpy", np.__version__, "scipy", scipy.__version__)
try:
    import numba
    print("numba", numba.__version__)
except Exception as e:
    print("no numba")
os.makedirs('/work', exist_ok=True)

lib = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class SWBatch:
    """R independent NxN Ising lattices (periodic b.c., J=1) updated by Swendsen-Wang."""
    def __init__(self, N, T, R, rng, init='mixed'):
        self.N = N; self.T = float(T); self.R = R; self.rng = rng
        self.p = 1.0 - np.exp(-2.0 / self.T)
        n = N * N
        self.nsite = R * n
        idx = np.arange(self.nsite, dtype=np.int32).reshape(R, N, N)
        self.idx = idx
        self.idx_h = np.roll(idx, -1, axis=2)
        self.idx_v = np.roll(idx, -1, axis=1)
        self.rep_of_site = np.repeat(np.arange(R, dtype=np.int32), n)
        s = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
        if init == 'cold':
            s[:] = 1
        elif init == 'mixed':          # first half ordered (all +1), second half random
            s[: R // 2] = 1
        self.s = s

    def step(self):
        s = self.s; rng = self.rng; N = self.N; R = self.R
        bh = (s == np.roll(s, -1, axis=2))
        bh &= (rng.random(s.shape) < self.p)
        bv = (s == np.roll(s, -1, axis=1))
        bv &= (rng.random(s.shape) < self.p)
        rows = np.concatenate((self.idx[bh], self.idx[bv]))
        cols = np.concatenate((self.idx_h[bh], self.idx_v[bv]))
        g = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)),
                       shape=(self.nsite, self.nsite))
        ncomp, labels = connected_components(g, directed=False)
        sizes = np.bincount(labels, minlength=ncomp).astype(np.float64)
        rep_c = np.empty(ncomp, dtype=np.int32)
        rep_c[labels] = self.rep_of_site
        s2 = sizes * sizes
        S2 = np.bincount(rep_c, weights=s2, minlength=R)          # sum_c |c|^2 per replica
        S4 = np.bincount(rep_c, weights=s2 * s2, minlength=R)     # sum_c |c|^4 per replica
        flip = (rng.integers(0, 2, size=ncomp, dtype=np.int8) * 2 - 1).astype(np.int8)
        self.s = (s.reshape(-1) * flip[labels]).reshape(R, N, N)
        return S2, S4

    def mag_energy(self):
        s = self.s; R = self.R; N = self.N
        M = s.reshape(R, -1).sum(axis=1, dtype=np.int64)
        bonds = ((s * np.roll(s, -1, axis=2)).reshape(R, -1).sum(axis=1, dtype=np.int64)
                 + (s * np.roll(s, -1, axis=1)).reshape(R, -1).sum(axis=1, dtype=np.int64))
        return M / float(N * N), -bonds / float(N * N)   # m, E per site


class MetroBatch:
    """R independent NxN lattices, checkerboard Metropolis (N even)."""
    def __init__(self, N, T, R, rng, init='mixed'):
        assert N % 2 == 0
        self.N = N; self.T = float(T); self.R = R; self.rng = rng
        ii, jj = np.indices((N, N))
        self.masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
        self.lut = np.exp(-np.array([-8, -4, 0, 4, 8], dtype=float) / self.T)
        s = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
        if init == 'cold':
            s[:] = 1
        elif init == 'mixed':
            s[: R // 2] = 1
        self.s = s

    def sweep(self):
        s = self.s
        for mask in self.masks:
            nn = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2))
            dE = (2 * s * nn).astype(np.int8)
            w = self.lut[(dE + 8) >> 2]
            acc = self.rng.random(s.shape) < w
            acc &= mask
            s = np.where(acc, -s, s).astype(np.int8)
        self.s = s

    def mag_energy(self):
        s = self.s; R = self.R; N = self.N
        M = s.reshape(R, -1).sum(axis=1, dtype=np.int64)
        bonds = ((s * np.roll(s, -1, axis=2)).reshape(R, -1).sum(axis=1, dtype=np.int64)
                 + (s * np.roll(s, -1, axis=1)).reshape(R, -1).sum(axis=1, dtype=np.int64))
        return M / float(N * N), -bonds / float(N * N)


def run_sw(N, T, R, n_burn, n_blocks, block, seed, init='mixed'):
    """Returns array out[q, b, r]: sum over `block` sweeps of
       q=0: m^2 (plain), 1: m^4 (plain), 2: m^2 (cluster-improved), 3: m^4 (cluster-improved), 4: E per site, 5: |m|."""
    rng = np.random.Generator(np.random.PCG64(seed))
    sw = SWBatch(N, T, R, rng, init)
    n2 = float(N) ** 2
    for _ in range(n_burn):
        sw.step()
    out = np.zeros((6, n_blocks, R))
    for b in range(n_blocks):
        acc = np.zeros((6, R))
        for k in range(block):
            S2, S4 = sw.step()
            m, E = sw.mag_energy()
            m2 = m * m
            acc[0] += m2
            acc[1] += m2 * m2
            acc[2] += S2 / n2 ** 2
            acc[3] += (3.0 * S2 * S2 - 2.0 * S4) / n2 ** 4
            acc[4] += E
            acc[5] += np.abs(m)
        out[:, b, :] = acc
    return out


def run_metro(N, T, R, n_burn, n_blocks, block, seed, init='mixed'):
    rng = np.random.Generator(np.random.PCG64(seed))
    mb = MetroBatch(N, T, R, rng, init)
    for _ in range(n_burn):
        mb.sweep()
    out = np.zeros((4, n_blocks, R))
    for b in range(n_blocks):
        acc = np.zeros((4, R))
        for k in range(block):
            mb.sweep()
            m, E = mb.mag_energy()
            m2 = m * m
            acc[0] += m2; acc[1] += m2 * m2; acc[2] += E; acc[3] += np.abs(m)
        out[:, b, :] = acc
    return out


def U_from_units(u2, u4):
    """u2,u4: arrays (nunits,) of per-unit sums (equal counts per unit). Returns U, jackknife err, bias estimate."""
    n = u2.size
    t2 = u2.sum(); t4 = u4.sum()
    U = 1.0 - (t4 / n) / (3.0 * (t2 / n) ** 2)
    l2 = (t2 - u2) / (n - 1); l4 = (t4 - u4) / (n - 1)
    Uj = 1.0 - l4 / (3.0 * l2 ** 2)
    err = np.sqrt((n - 1) / n * np.sum((Uj - Uj.mean()) ** 2))
    bias = (n - 1) * (Uj.mean() - U)
    return U, err, bias


def mean_from_units(u, count_per_unit):
    n = u.size
    mean = u.sum() / (n * count_per_unit)
    l = (u.sum() - u) / ((n - 1) * count_per_unit)
    err = np.sqrt((n - 1) / n * np.sum((l - l.mean()) ** 2))
    return mean, err


def exact_enum(N, Ts):
    """Exact <m^2>,<m^4>,<E>/N^2,U by brute-force enumeration (N=4: 65536 states)."""
    n = N * N
    states = np.arange(2 ** n, dtype=np.uint64)
    bits = ((states[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(np.int8) * 2 - 1
    spins = bits.reshape(-1, N, N)
    E = -(spins * np.roll(spins, -1, axis=2)).sum((1, 2)) - (spins * np.roll(spins, -1, axis=1)).sum((1, 2))
    M = spins.sum((1, 2)) / n
    res = {}
    for T in Ts:
        w = np.exp(-(E - E.min()) / T); Z = w.sum()
        m2 = (w * M ** 2).sum() / Z; m4 = (w * M ** 4).sum() / Z; e = (w * E).sum() / Z / n
        res[T] = (m2, m4, e, 1 - m4 / (3 * m2 ** 2))
    return res, E, M


def Z_torus(K, N):
    """Kaufman / Ferdinand-Fisher exact partition function of the NxN torus, coupling K=J/T (isotropic)."""
    K = float(K)
    r = np.arange(2 * N)
    c = np.cosh(2 * K) ** 2 / np.sinh(2 * K) - np.cos(np.pi * r / N)
    gam = np.arccosh(c)
    gam[0] = 2 * K + np.log(np.tanh(K))          # signed
    go = gam[1::2]; ge = gam[0::2]
    Z1 = np.prod(2 * np.cosh(N * go / 2)); Z2 = np.prod(2 * np.sinh(N * go / 2))
    Z3 = np.prod(2 * np.cosh(N * ge / 2)); Z4 = np.prod(2 * np.sinh(N * ge / 2))
    return 0.5 * (2 * np.sinh(2 * K)) ** (N * N / 2.0) * (Z1 + Z2 + Z3 + Z4)


def E_exact_torus(T, N, h=1e-5):
    K = 1.0 / T
    lz = lambda k: np.log(Z_torus(k, N))
    dlnZ = (lz(K + h) - lz(K - h)) / (2 * h)
    return -dlnZ / (N * N)       # <E>/N^2  (E = -sum s s)
'''
open('/work/ising_lib.py', 'w').write(lib)

sys.path.insert(0, '/work')
import importlib, ising_lib
importlib.reload(ising_lib)
from ising_lib import *

# timing of one SW step for several R and T
for T in (1.5, 2.3, 2.8):
    for R in (64, 256):
        rng = np.random.Generator(np.random.PCG64(1))
        sw = SWBatch(16, T, R, rng, 'hot')
        for _ in range(5): sw.step()
        t0 = time.time()
        n = 20
        for _ in range(n):
            sw.step(); sw.mag_energy()
        dt = (time.time() - t0) / n
        print(f"T={T} R={R}: {dt*1e3:.2f} ms/step ({dt/R*1e6:.1f} us per replica-sweep)")
