import sys, time, math
sys.path.insert(0, '/work')
import numpy as np
import ising

def tau_int(x, maxlag=2000):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    # FFT autocorrelation
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    W = 0
    for t in range(1, min(maxlag, n//10)):
        tau += acf[t]
        if t >= 6*tau:
            break
        W = t
    return tau, W

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
res = {}
t_start = time.time()
for idx, T in enumerate(Ts):
    t0 = time.time()
    M, _ = ising.wolff_chain(16, T, 200000, 5000, seed=1000+idx)
    dt = time.time()-t0
    U, Ujk, Ue = ising.binder_jk(M, 200)
    M2 = (M.astype(float))**2
    tau, W = tau_int(M2)
    # tau for the Binder-like quantity via M^2 and M^4 combined is roughly similar
    res[T] = (U, Ujk, Ue, tau)
    print(f"T={T:.2f}: U={U:.5f} jk={Ujk:.5f} err={Ue:.5f}  tau_int(M^2)={tau:.2f}  time={dt:.1f}s  <|m|>={np.mean(np.abs(M))/256:.4f}", flush=True)
print("total time", time.time()-t_start)
