
import sys, time
sys.path.insert(0, "/work")
import numpy as np
from sw import SWEnsemble

def tau_int_pooled(X, cmax=None):
    # X: (R, K) array; returns integrated autocorrelation time (Sokal automatic window), pooled over replicas
    R, K = X.shape
    X = X - X.mean()
    nfft = 1
    while nfft < 2*K: nfft *= 2
    F = np.fft.rfft(X, n=nfft, axis=1)
    C = np.fft.irfft(F*np.conj(F), n=nfft, axis=1)[:, :K]
    C = C.mean(axis=0)
    if C[0] <= 0: return np.nan
    rho = C/C[0]
    tau = 0.5
    for W in range(1, K):
        tau += rho[W]
        if W >= 5*tau:
            return tau
    return tau

def binder_jk(A2, A4):
    # A2, A4: (R, K) arrays of m^2, m^4 per replica & step. Returns U, jackknife err over replicas, bias-corrected U
    R, K = A2.shape
    a2 = A2.sum(axis=1); a4 = A4.sum(axis=1)
    S2 = a2.sum(); S4 = a4.sum(); Nt = R*K
    U = 1 - (S4/Nt)/(3*(S2/Nt)**2)
    S2j = S2 - a2; S4j = S4 - a4; Nj = (R-1)*K
    Uj = 1 - (S4j/Nj)/(3*(S2j/Nj)**2)
    err = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    Ubc = R*U - (R-1)*Uj.mean()
    return U, err, Ubc

N = 16; R = 1000; burn = 100; K = 1000
results = {}
t_all = time.time()
for T in [1.5, 1.9, 2.1, 2.27, 2.5, 2.8]:
    t0 = time.time()
    ens = SWEnsemble(N, R, T, seed=int(1000*T)+11)
    # burn-in with recording of early trajectory for equilibration check
    early = []
    for t in range(burn):
        ens.step()
        if t < 30 or t % 10 == 0:
            m = ens.magsums()/256.0
            early.append((t, float((m*m).mean())))
    Ms = np.empty((R, K), dtype=np.int16)
    for k in range(K):
        ens.step()
        Ms[:, k] = ens.magsums()
    m = Ms.astype(np.float64)/256.0
    A2 = m*m; A4 = A2*A2
    U, err, Ubc = binder_jk(A2, A4)
    tau2 = tau_int_pooled(A2); tau4 = tau_int_pooled(A4)
    # equilibrium check: replica-averaged m^2 in first steps vs later measurement
    m2_meas_first = A2[:, :200].mean(); m2_meas_last = A2[:, -200:].mean()
    results[T] = dict(U=U, err=err, Ubc=Ubc, tau2=tau2, tau4=tau4)
    print(f"T={T}: U={U:.5f} (bias-corr {Ubc:.5f}) jk_err={err:.5f}  <m2>={A2.mean():.5f}  tau_int(m2)={tau2:.2f} tau_int(m4)={tau4:.2f}  "
          f"m2 first200={m2_meas_first:.5f} last200={m2_meas_last:.5f}  time {time.time()-t0:.1f}s")
    print("   burn-in trajectory (step, replica-avg m2):", [(a, round(b,4)) for a,b in early[:12]], "...", [(a, round(b,4)) for a,b in early[-4:]])
    np.save(f"/work/pilot_T{T}.npy", Ms)
print("total", round(time.time()-t_all,1))
