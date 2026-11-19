
code = r'''
import numpy as np

def U_of(m2bar, m4bar):
    return 1.0 - m4bar/(3.0*m2bar**2)

def moments(M, N=16):
    m = M.astype(np.float64)/(N*N)
    m2 = m*m
    return m2, m2*m2

def influence_series(m2, m4):
    """Linearised (delta-method) series u_t such that mean(u_t) - U ~ dU, so Var(mean) gives the error."""
    a = m2.mean(); b = m4.mean()
    # U = 1 - b/(3 a^2);  dU = -(1/(3a^2)) db + (2 b/(3 a^3)) da
    return -(m4 - b)/(3*a*a) + (2*b/(3*a**3))*(m2 - a)

def tau_int_sokal(x, c=6.0):
    """Integrated autocorrelation time with Sokal automatic windowing. Returns (tau, window)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for W in range(1, n):
        tau = 1.0 + 2.0*np.sum(acf[1:W+1])
        if W >= c*tau:
            return tau, W
    return tau, n

def jackknife_U(S2, S4, cnt):
    """S2,S4,cnt: per-block sums of m^2, m^4 and number of samples. Returns (U_full, jackknife_err, U_bias_corrected)."""
    S2 = np.asarray(S2, float); S4 = np.asarray(S4, float); cnt = np.asarray(cnt, float)
    B = len(S2)
    T2 = S2.sum(); T4 = S4.sum(); Tn = cnt.sum()
    U_full = U_of(T2/Tn, T4/Tn)
    loo = U_of((T2 - S2)/(Tn - cnt), (T4 - S4)/(Tn - cnt))
    mean_loo = loo.mean()
    err = np.sqrt((B-1)/B*np.sum((loo-mean_loo)**2))
    U_bc = B*U_full - (B-1)*mean_loo
    return U_full, err, U_bc

def block_sums(M, nblocks, N=16):
    m2, m4 = moments(M, N)
    L = len(M)//nblocks
    m2 = m2[:L*nblocks].reshape(nblocks, L); m4 = m4[:L*nblocks].reshape(nblocks, L)
    return m2.sum(1), m4.sum(1), np.full(nblocks, L)
'''
open("/work/analysis.py","w").write(code)

import sys, time
sys.path.insert(0,"/work")
import numpy as np
from ising import wolff_run
from analysis import *

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N=16
nmeas=100000
res = {}
t0=time.time()
for T in Ts:
    t=time.time()
    M, cbar, th = wolff_run(N, T, seed=2000+int(round(T*100)), n_therm=2000, n_meas=nmeas, init="cold")
    dt=time.time()-t
    m2,m4 = moments(M)
    U = U_of(m2.mean(), m4.mean())
    u = influence_series(m2,m4)
    tau,W = tau_int_sokal(u)
    err_delta = np.sqrt(u.var()*tau/len(u))
    S2,S4,cnt = block_sums(M, 50)
    Uf, err_jk, Ubc = jackknife_U(S2,S4,cnt)
    tau2,_ = tau_int_sokal(m2); tau4,_=tau_int_sokal(m4)
    print(f"T={T:.1f} U={U:.4f} err_delta={err_delta:.5f} err_jk50={err_jk:.5f} tau_u={tau:6.2f} flips (={tau*cbar/256:5.2f} sweeps)  tau_m2={tau2:6.2f} tau_m4={tau4:6.2f} sd_u={u.std():.4f}  t={dt:.1f}s")
    res[T]=(U,err_delta)
print("total", time.time()-t0)
