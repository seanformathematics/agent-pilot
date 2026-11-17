import sys, time, numpy as np
sys.path.insert(0, '/work')
import ising

def tau_int(x, maxlag=400):
    # x: (n, R) ; estimate integrated autocorr time of the chain-averaged series via FFT on each chain, averaged
    n, R = x.shape
    x = x - x.mean(axis=0)
    f = np.fft.rfft(x, n=2*n, axis=0)
    ac = np.fft.irfft(f*np.conj(f), axis=0)[:n]
    ac = ac.mean(axis=1)
    ac = ac / ac[0]
    # Sokal-style windowing
    tau = 0.5
    for t in range(1, min(maxlag, n//4)):
        tau += ac[t]
        if t >= 6 * tau:
            break
    return tau

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N = 16
res = {}
t0 = time.time()
for idx, T in enumerate(Ts):
    M, M2, M4, E, s = ising.run_chains(N, T, R=400, n_equil=1000, n_meas=2000, seed=1000+idx, init='random')
    U, err = ising.binder_jack(M2, M4)
    tau2 = tau_int(M2, )
    tau4 = tau_int(M4)
    # per-sample sd of U-ish: sd of m2 and m4 pooled
    res[T] = (U, err, tau2, tau4, E.mean())
    print(f"T={T:.2f}  U={U:.5f}  err={err:.5f}  tau_m2={tau2:.1f} tau_m4={tau4:.1f}  E={E.mean():.4f} sd(m2)={M2.std():.4f} sd(m4)={M4.std():.4f} time={time.time()-t0:.1f}s")
