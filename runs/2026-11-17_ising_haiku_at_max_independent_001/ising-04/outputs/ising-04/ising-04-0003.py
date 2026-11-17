import sys, time
sys.path.insert(0, '/work')
import numpy as np
from swlib import *

def tau_int(x, c=6.0):
    """Integrated autocorrelation time (Sokal automatic windowing) of a 1D series."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf = acf/acf[0]
    tau = 1.0
    for W in range(1, n):
        tau = 1 + 2*acf[1:W+1].sum()
        if W >= c*tau:
            break
    return tau

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
N = 16
R = 128; nsw = 1000; nb = 200
res = {}
t0 = time.time()
for i, T in enumerate(Ts):
    S1, S2, S4, ser = run(N, T, R=R, nsweep=nsw, nburn=nb, seed=1000+i, series_reps=4)
    U, err = jackknife_U(S2, S4, nsw)
    taus = [tau_int(ser[r]) for r in range(4)]
    m2 = S2.sum()/(nsw*R); m1 = S1.sum()/(nsw*R)
    res[T] = (U, err)
    print(f"T={T:.2f} U={U:.5f} err={err:.5f} <|m|>={m1:.4f} <m2>={m2:.5f} tau(m2) reps={np.round(taus,2)}", flush=True)
print("total time", time.time()-t0)
