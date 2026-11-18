import os, time, sys
module_src = r'''
import numpy as np
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
        self.N, self.src, self.dst = make_geometry(L)
        self.R = R; self.T = float(T)
        self.p = 1.0 - np.exp(-2.0/self.T)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        if init == "random":
            self.S = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, self.N))
        else:
            self.S = np.ones((R, self.N), dtype=np.int8)
    def sweep(self):
        R, N = self.S.shape
        S = self.S
        same = S[:, self.src] == S[:, self.dst]
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

def run_chain(L, R, T, nsweeps, seed, init="random"):
    N, src, dst = make_geometry(L)
    sw = SW(L, R, T, seed, init)
    Ms = np.empty((nsweeps, R), dtype=np.int16)
    Es = np.empty((nsweeps, R), dtype=np.int16)
    for t in range(nsweeps):
        sw.sweep()
        S = sw.S
        Ms[t] = S.sum(axis=1, dtype=np.int64)
        Es[t] = -(S[:, src].astype(np.int64) * S[:, dst]).sum(axis=1)
    return Ms, Es

def jack_U(s2, s4, n_per_bin):
    K = s2.size
    S2 = s2.sum(); S4 = s4.sum(); n = n_per_bin*K
    U_full = 1 - n*S4/(3*S2**2)
    S2k = S2 - s2; S4k = S4 - s4; nk = n - n_per_bin
    Uk = 1 - nk*S4k/(3*S2k**2)
    var = (K-1)/K*np.sum((Uk - Uk.mean())**2)
    return U_full, np.sqrt(var)

def U_from_series(Ms, burn=0):
    x = Ms[burn:].astype(np.float64)
    m2 = np.mean(x**2); m4 = np.mean(x**4)
    return 1 - m4/(3*m2**2)

def U_jack(Ms, B, burn):
    x = Ms[burn:].astype(np.float64)
    nt, R = x.shape
    nb = nt // B
    x = x[:nb*B].reshape(nb, B, R)
    s2 = (x**2).sum(axis=1).ravel()
    s4 = (x**4).sum(axis=1).ravel()
    return jack_U(s2, s4, B)
'''
with open("/work/sw_ising.py", "w") as f:
    f.write(module_src)
sys.path.insert(0, "/work")
from sw_ising import run_chain, U_from_series, U_jack
import numpy as np

L = 16; R = 2000; NSW = 1000; BURN = 200; B = 20
Tlist = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
base_seed = 20250000
sel = [0, 1, 2, 3, 4, 5, 6]
log = []
t_start = time.time()
for i in sel:
    T = Tlist[i]
    seed = base_seed + i
    t0 = time.time()
    Ms, Es = run_chain(L, R, T, NSW, seed)
    dt = time.time() - t0
    np.savez_compressed(f"/work/prod_T{T:.1f}_s{seed}.npz", Ms=Ms, Es=Es, T=T, seed=seed, R=R, NSW=NSW)
    U, Ue = U_jack(Ms, B, BURN)
    U1 = U_from_series(Ms[BURN:600]); U2 = U_from_series(Ms[600:])
    G = 10
    gU = [U_from_series(Ms[BURN:, g*R//G:(g+1)*R//G]) for g in range(G)]
    gerr = np.std(gU, ddof=1)/np.sqrt(G)
    line = (f"T={T:.2f} seed={seed} time={dt:.1f}s U={U:.5f} Uerr_jack(B={B})={Ue:.5f} "
            f"Uerr_groups={gerr:.5f} U[200:600]={U1:.5f} U[600:]={U2:.5f} E/N={Es[BURN:].mean()/L**2:.4f}")
    print(line, flush=True)
    log.append(line)
    if time.time() - t_start > 520:
        print("stopping early for time budget", flush=True)
        break
with open("/work/log_batch1.txt", "w") as f:
    f.write("\n".join(log) + "\n")
print("total", time.time()-t_start)
