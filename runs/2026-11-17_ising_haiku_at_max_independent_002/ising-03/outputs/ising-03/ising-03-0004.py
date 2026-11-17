analysis = r'''
import numpy as np

def U_from_chains(M2, M4):
    """M2, M4: arrays (R, n) of per-sweep m^2 and m^4 values (already burn-in removed).
    Returns full-sample U and jackknife error over replica groups."""
    R = M2.shape[0]
    B2 = M2.mean(); B4 = M4.mean()
    U = 1 - B4/(3*B2**2)
    G = min(100, R)
    edges = np.linspace(0, R, G+1).astype(int)
    s2 = np.array([M2[:a].sum()+M2[b:].sum() for a, b in zip(edges[:-1], edges[1:])])
    s4 = np.array([M4[:a].sum()+M4[b:].sum() for a, b in zip(edges[:-1], edges[1:])])
    n_all = M2.size
    n_g = np.array([M2.size - M2[a:b].size for a, b in zip(edges[:-1], edges[1:])], dtype=float)
    Ug = 1 - (s4/n_g)/(3*(s2/n_g)**2)
    Uj = G*U - (G-1)*Ug.mean()
    err = np.sqrt((G-1)/G*np.sum((Ug - Ug.mean())**2))
    return U, err, Uj

def tau_int(X, maxlag=50):
    """Integrated autocorrelation time (in sweeps) of series X (R, n), replica-averaged ACF."""
    X = X - X.mean()
    R, n = X.shape
    f = np.fft.rfft(X, n=2*n, axis=1)
    acf = np.fft.irfft(f*np.conj(f), n=2*n, axis=1)[:, :n]
    acf = acf.sum(axis=0)
    acf = acf / acf[0]
    tau = 0.5
    for t in range(1, maxlag):
        tau += acf[t]
    return tau
'''
with open("/work/analysis.py", "w") as f:
    f.write(analysis)

import sys, time
sys.path.insert(0, "/work")
import numpy as np
from sw import SW
from analysis import U_from_chains, tau_int

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R = 2000; nsw = 300; burn = 100
t_start = time.time()
for k, T in enumerate(Ts):
    rng = np.random.default_rng(1000 + k)
    s = SW(16, R, T, rng, init="random")
    M2 = np.zeros((R, nsw), dtype=np.float32); M4 = np.zeros((R, nsw), dtype=np.float32)
    for t in range(nsw):
        s.sweep()
        m = s.m()
        M2[:, t] = m**2; M4[:, t] = m**4
    np.savez(f"/work/data/pilot_T{T:.2f}.npz", M2=M2, M4=M4)
    early = M2[:, :20].mean(); mid = M2[:, 20:burn].mean(); late = M2[:, burn:].mean()
    U, err, Uj = U_from_chains(M2[:, burn:].astype(float), M4[:, burn:].astype(float))
    tau = tau_int(M2[:, burn:].astype(float))
    tau4 = tau_int(M4[:, burn:].astype(float))
    print(f"T={T:.2f} U={U:.5f} err={err:.5f} (Uj-U={Uj-U:+.1e}) <m2>early={early:.4f} mid={mid:.4f} late={late:.4f} tau_m2={tau:.2f} tau_m4={tau4:.2f} elapsed={time.time()-t_start:.0f}s")
