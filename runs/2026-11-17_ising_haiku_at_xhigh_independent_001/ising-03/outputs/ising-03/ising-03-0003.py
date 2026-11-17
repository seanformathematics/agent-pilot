import sys, time
sys.path.insert(0,'/work')
import numpy as np
from sw import run

def U_from(m2s, m4s):
    # m2s, m4s shape (nmeas, R); pooled estimate
    M2 = m2s.mean(); M4 = m4s.mean()
    return 1 - M4/(3*M2**2)

def jack_err(m2s, m4s):
    R = m2s.shape[1]
    # per-chain sums
    s2 = m2s.sum(0); s4 = m4s.sum(0); n = m2s.shape[0]
    tot2 = s2.sum(); tot4 = s4.sum()
    Uj = np.empty(R)
    for r in range(R):
        M2 = (tot2 - s2[r])/((R-1)*n); M4 = (tot4 - s4[r])/((R-1)*n)
        Uj[r] = 1 - M4/(3*M2**2)
    err = np.sqrt((R-1)/R * ((Uj-Uj.mean())**2).sum())
    return err

def tau_int(x, maxlag=200):
    x = x - x.mean()
    n = len(x)
    c0 = (x*x).mean()
    tau = 0.5
    for k in range(1, maxlag):
        ck = (x[:-k]*x[k:]).mean()
        rho = ck/c0
        if rho < 0.05: break
        tau += rho
    return tau

for T in [1.5, 2.0, 2.27, 2.6, 2.8]:
    t=time.time()
    m2s, m4s = run(16, 64, T, 2024, 300, 2000)
    U = U_from(m2s, m4s)
    err = jack_err(m2s, m4s)
    # tau for m2 averaged over chains
    taus = [tau_int(m2s[:,r]) for r in range(0,64,8)]
    print(f"T={T}: U={U:.5f} err={err:.5f}  tau(m2)~{np.mean(taus):.2f}  elapsed {time.time()-t:.1f}s")
