import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from ising import run_chains, jackknife_U

def tau_int(x, maxlag=None):
    # x: (T, K) time series for K chains; returns integrated autocorr time (Sokal-style window)
    x = x - x.mean(axis=0, keepdims=True)
    T_, K = x.shape
    nfft = 1 << (2 * T_ - 1).bit_length()
    f = np.fft.rfft(x, n=nfft, axis=0)
    ac = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:T_]
    ac = ac.mean(axis=1)
    if ac[0] <= 0:
        return np.nan
    rho = ac / ac[0]
    tau = 0.5
    for t in range(1, T_):
        tau += rho[t]
        if t >= 6 * tau:
            break
    return tau

temps = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
res = {}
t_start = time.time()
for k, T in enumerate(temps):
    t0 = time.time()
    S2, S4, eqt, ts = run_chains(16, T, R=1024, n_eq=300, n_meas=1000, seed=1000 + k, ts_reps=16)
    Uf, Ue, Ubc = jackknife_U(S2, S4, 1000, 64)
    m2 = ts.astype(float)**2
    tau_m2 = tau_int(m2)
    tau_m = tau_int(np.abs(ts.astype(float)))
    res[T] = dict(U=Uf, err=Ue, Ubc=Ubc, tau_m2=float(tau_m2), tau_absm=float(tau_m),
                  eq_first=float(eqt[0]), eq_50=float(eqt[50]), eq_100=float(eqt[100]),
                  eq_200=float(eqt[200]), eq_299=float(eqt[299]))
    print(f"T={T}: U={Uf:.5f} +- {Ue:.5f} (bc {Ubc:.5f}) tau(M2)={tau_m2:.2f} tau(|M|)={tau_m:.2f} "
          f"<M2>_eq trace: t0={eqt[0]:.0f} t50={eqt[50]:.1f} t100={eqt[100]:.1f} t200={eqt[200]:.1f} t299={eqt[299]:.1f} [{time.time()-t0:.1f}s]", flush=True)
with open("/work/pilot_results.json", "w") as f:
    json.dump({str(k): v for k, v in res.items()}, f, indent=1)
print("total", time.time() - t_start)
