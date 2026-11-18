import numpy as np, time, os
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import ellipk

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
        self.R = R; self.T = T
        self.p = 1.0 - np.exp(-2.0/T)
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

def onsager_u(T):
    K = 1.0/T
    t2 = np.tanh(2*K)
    kap = 2*np.sinh(2*K)/np.cosh(2*K)**2
    Kc = ellipk(kap**2)
    return -(1/t2)*(1 + (2/np.pi)*(2*t2**2 - 1)*Kc)

def jack_U(s2, s4, n_per_bin):
    # s2, s4: arrays of bin sums (K bins); n_per_bin scalar
    K = s2.size
    S2 = s2.sum(); S4 = s4.sum(); n = n_per_bin*K
    U_full = 1 - n*S4/(3*S2**2)
    S2k = S2 - s2; S4k = S4 - s4; nk = n - n_per_bin
    Uk = 1 - nk*S4k/(3*S2k**2)
    var = (K-1)/K*np.sum((Uk - Uk.mean())**2)
    return U_full, np.sqrt(var)

def block_U_err(Ms, B, burn):
    x = Ms[burn:].astype(float)
    nt, R = x.shape
    nb = nt//B
    x = x[:nb*B].reshape(nb, B, R)
    s2 = (x**2).sum(axis=1).ravel()
    s4 = (x**4).sum(axis=1).ravel()
    return jack_U(s2, s4, B)

def acf_tau(Ms, burn, maxlag=200):
    x = Ms[burn:].astype(float)
    nt, R = x.shape
    x = x - x.mean(axis=0)
    f = np.fft.rfft(x, n=2*nt, axis=0)
    ac = np.fft.irfft(f*np.conj(f), axis=0)[:nt]
    ac = ac/ac[0:1]
    rho = ac.mean(axis=1)
    tau = 0.5
    for W in range(1, maxlag):
        tau += rho[W]
        if W >= 5*tau:
            break
    return tau, W

L = 16
R = 200; nsw = 1000
print("Onsager energy/site check and pilot")
for T in [1.5, 1.8, 2.0, 2.2692, 2.5, 2.8]:
    t0 = time.time()
    Ms, Es = run_chain(L, R, T, nsw, seed=int(1000*T)+1)
    dt = time.time()-t0
    burn = 100
    M = Ms.astype(float)
    # equilibration check: ensemble U in windows of 50 sweeps over first 300
    m2w = [np.mean(M[w:w+50]**2) for w in range(0, 300, 50)]
    uw = [1-np.mean(M[w:w+50]**4)/(3*np.mean(M[w:w+50]**2)**2) for w in range(0, 300, 50)]
    U = 1 - np.mean(M[burn:]**4)/(3*np.mean(M[burn:]**2)**2)
    tau2, W2 = acf_tau(Ms**2 if False else Ms, burn)
    x4 = (Ms.astype(float)**2)
    tau_m2, Wm2 = acf_tau(x4.astype(np.float64).astype(np.int64) if False else Ms, burn)
    errs = {}
    for B in [5, 20, 50, 100]:
        Uf, Ue = block_U_err(Ms, B, burn)
        errs[B] = Ue
    e_site = Es[burn:].mean()/L**2
    print(f"T={T}: time {dt:.1f}s U={U:.5f} | U_err by block B: " + ", ".join(f"B{B}:{errs[B]:.5f}" for B in errs))
    print(f"   |M|-based tau_int ~ {tau_m2:.2f} (W={Wm2}); E/N={e_site:.5f} Onsager={onsager_u.__name__ and (-np.sqrt(2) if False else 0):.0f}" if False else f"   tau_int(M)={tau_m2:.2f} (W={Wm2}); E/N={e_site:.5f}; Onsager e/N={-(1/np.tanh(2/T))*(1+(2/np.pi)*(2*np.tanh(2/T)**2-1)*ellipk((2*np.sinh(2/T)/np.cosh(2/T)**2)**2)):.5f}")
    print("   equilibration windows (U, m2) first 300 sweeps:", " ".join(f"{u:.4f}/{m:.1f}" for u, m in zip(uw, m2w)))
