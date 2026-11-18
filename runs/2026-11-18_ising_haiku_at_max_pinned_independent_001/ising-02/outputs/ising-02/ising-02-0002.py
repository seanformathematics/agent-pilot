
import os, time
os.makedirs("/work", exist_ok=True)
src = r'''
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def neighbor_tables(N):
    n = N*N
    ii = np.arange(n)
    x = ii // N; y = ii % N
    right = ((x+1) % N)*N + y
    down = x*N + (y+1) % N
    return right, down

class SWEnsemble:
    """R independent periodic Ising lattices NxN, updated with Swendsen-Wang in parallel."""
    def __init__(self, N, R, T, seed):
        self.N = N; self.R = R; self.T = float(T); self.n = N*N
        self.p = 1.0 - np.exp(-2.0/self.T)
        self.rng = np.random.default_rng(seed)
        right, down = neighbor_tables(N)
        off = (np.arange(R, dtype=np.int64)*self.n)[:, None]
        self.gR = (off + right[None, :]).ravel().astype(np.int64)
        self.gD = (off + down[None, :]).ravel().astype(np.int64)
        self.gi = np.arange(R*self.n, dtype=np.int64)
        self.S = (self.rng.integers(0, 2, size=R*self.n, dtype=np.int8)*2 - 1).astype(np.int8)
    def step(self):
        S = self.S; rng = self.rng; p = self.p
        actR = (S == S[self.gR]) & (rng.random(S.size) < p)
        actD = (S == S[self.gD]) & (rng.random(S.size) < p)
        rows = np.concatenate((self.gi[actR], self.gi[actD]))
        cols = np.concatenate((self.gR[actR], self.gD[actD]))
        G = sp.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(S.size, S.size))
        ncomp, labels = connected_components(G, directed=False)
        flip = rng.random(ncomp) < 0.5
        self.S = np.where(flip[labels], -S, S).astype(np.int8)
    def magsums(self):
        return self.S.reshape(self.R, self.n).sum(axis=1, dtype=np.int64)
    def energies(self):
        S = self.S
        b = (S*S[self.gR]).astype(np.int64) + (S*S[self.gD]).astype(np.int64)
        return -b.reshape(self.R, self.n).sum(axis=1)

def exact_stats(N, T):
    n = N*N
    cfg = np.arange(2**n, dtype=np.int64)
    S = (((cfg[:, None] >> np.arange(n)[None, :]) & 1)*2 - 1).astype(np.int64)
    right, down = neighbor_tables(N)
    ii = np.arange(n)
    E = -(S[:, ii]*S[:, right]).sum(1) - (S[:, ii]*S[:, down]).sum(1)
    w = np.exp(-(E - E.min())/T)
    Z = w.sum()
    m = S.sum(1)/n
    m2 = (w*m**2).sum()/Z; m4 = (w*m**4).sum()/Z
    e = (w*E).sum()/Z/n
    return dict(m2=m2, m4=m4, U=1 - m4/(3*m2**2), e=e)
'''
with open("/work/sw.py", "w") as f:
    f.write(src)

import sys
sys.path.insert(0, "/work")
import numpy as np
from sw import SWEnsemble, exact_stats

# ---- validation on 4x4 against exact enumeration ----
t0 = time.time()
for T in [1.5, 2.0, 2.5, 3.0]:
    ex = exact_stats(4, T)
    ens = SWEnsemble(4, 400, T, seed=2024)
    for _ in range(100):
        ens.step()
    A2 = []; A4 = []; EE = []
    for k in range(1500):
        ens.step()
        m = ens.magsums()/16.0
        A2.append((m*m).mean()); A4.append((m**4).mean())
        EE.append(ens.energies().mean()/16.0)
    A2 = np.array(A2); A4 = np.array(A4); EE = np.array(EE)
    U_mc = 1 - A4.mean()/(3*A2.mean()**2)
    print(f"T={T}: exact m2={ex['m2']:.5f} m4={ex['m4']:.5f} U={ex['U']:.5f} e={ex['e']:.5f} | "
          f"MC m2={A2.mean():.5f} m4={A4.mean():.5f} U={U_mc:.5f} e={EE.mean():.5f} "
          f"(se e={EE.std()/np.sqrt(EE.size):.5f}, approx se m2 {A2.std()/np.sqrt(A2.size):.5f})")
print("validation time", round(time.time()-t0, 2))

# ---- timing at N=16 ----
for R in [200, 1000]:
    ens = SWEnsemble(16, R, 2.27, seed=7)
    t = time.time()
    for _ in range(20):
        ens.step()
    dt = (time.time()-t)/20
    print(f"N=16 R={R}: {dt*1000:.1f} ms per SW step ({R/dt:.0f} replica-steps/s)")
