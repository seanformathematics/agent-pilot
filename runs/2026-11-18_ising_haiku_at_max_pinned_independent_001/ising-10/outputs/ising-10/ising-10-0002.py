
import time, numpy as np
src = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

class SWEngine:
    def __init__(self, L, R, T, seed):
        self.L = L; self.N = L*L; self.R = R; self.T = float(T)
        self.rng = np.random.default_rng(seed)
        idx = np.arange(self.N).reshape(L, L)
        right = np.roll(idx, -1, axis=1)   # bond (x,y)-(x,y+1)
        down = np.roll(idx, -1, axis=0)    # bond (x,y)-(x+1,y)
        self.bi = np.concatenate([idx.ravel(), idx.ravel()]).astype(np.int64)
        self.bj = np.concatenate([right.ravel(), down.ravel()]).astype(np.int64)
        self.p = 1.0 - np.exp(-2.0/self.T)   # SW bond prob for aligned pairs, J=1
        self.spins = (2*self.rng.integers(0, 2, size=(R, self.N)) - 1).astype(np.int8)
        self.V = R*self.N

    def sweep(self):
        s = self.spins
        aligned = s[:, self.bi] == s[:, self.bj]
        u = self.rng.random(aligned.shape)
        active = aligned & (u < self.p)
        rr, bb = np.nonzero(active)
        rows = rr*self.N + self.bi[bb]
        cols = rr*self.N + self.bj[bb]
        g = csr_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(self.V, self.V))
        ncomp, labels = connected_components(g, directed=False)
        flip = self.rng.random(ncomp) < 0.5
        f = flip[labels].reshape(self.R, self.N)
        self.spins = np.where(f, -s, s).astype(np.int8)

def binder_from_sums(B2, B4, n):
    R = len(B2)
    S2 = B2.sum(); S4 = B4.sum()
    m2 = S2/(R*n); m4 = S4/(R*n)
    U = 1 - m4/(3*m2*m2)
    c = (R-1)*n
    m2j = (S2 - B2)/c; m4j = (S4 - B4)/c
    Uj = 1 - m4j/(3*m2j*m2j)
    err = np.sqrt((R-1)/R*np.sum((Uj - Uj.mean())**2))
    return U, err

def run(L, R, T, seed, burn, nsweep):
    eng = SWEngine(L, R, T, seed)
    for _ in range(burn):
        eng.sweep()
    B2 = np.zeros(R); B4 = np.zeros(R)
    for _ in range(nsweep):
        eng.sweep()
        m = eng.spins.sum(axis=1)/eng.N
        m2 = m*m
        B2 += m2; B4 += m2*m2
    return B2, B4
'''
open('/work/sw.py', 'w').write(src)
import importlib, sys
sys.path.insert(0, '/work')
import sw
importlib.reload(sw)

# exact enumeration for L=4
def exact_U(L, Ts):
    N = L*L
    nconf = 2**N
    bits = ((np.arange(nconf)[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int64)
    S = (2*bits - 1).reshape(-1, L, L)
    E = -((S*np.roll(S, -1, axis=1)).sum(axis=(1, 2)) + (S*np.roll(S, -1, axis=2)).sum(axis=(1, 2)))
    M = S.sum(axis=(1, 2))
    m = M/N
    out = {}
    for T in Ts:
        w = np.exp(-(E - E.min())/T)
        Z = w.sum()
        m2 = (w*m*m).sum()/Z
        m4 = (w*m**4).sum()/Z
        out[T] = (1 - m4/(3*m2*m2), m2, m4)
    return out

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
ex = exact_U(4, Ts)
t0 = time.time()
res = {}
for k, T in enumerate(Ts):
    B2, B4 = sw.run(4, 4000, T, seed=12345+k, burn=50, nsweep=500)
    U, err = sw.binder_from_sums(B2, B4, 500)
    res[T] = (U, err)
print("L=4 check (exact vs SW):")
for T in Ts:
    print(f"T={T:.2f} exact U={ex[T][0]:.5f}  SW U={res[T][0]:.5f} +- {res[T][1]:.5f}  diff/err={(res[T][0]-ex[T][0])/res[T][1]:.2f}")
print("time L4 check", time.time()-t0)

# timing for L=16
for R in [500, 1000]:
    eng = sw.SWEngine(16, R, 2.269, seed=7)
    t0 = time.time()
    for _ in range(40):
        eng.sweep()
    dt = (time.time()-t0)/40
    print(f"L=16 R={R}: {dt*1e3:.2f} ms per sweep, {dt/R*1e6:.2f} us per replica-sweep")
