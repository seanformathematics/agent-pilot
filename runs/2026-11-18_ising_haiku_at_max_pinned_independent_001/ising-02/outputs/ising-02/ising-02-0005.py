
import sys, time, json, os
sys.path.insert(0, "/work")
import numpy as np
from sw import SWEnsemble

def tau_int_pooled(X):
    R, K = X.shape
    X = X - X.mean()
    nfft = 1
    while nfft < 2*K: nfft *= 2
    F = np.fft.rfft(X, n=nfft, axis=1)
    C = np.fft.irfft(F*np.conj(F), n=nfft, axis=1)[:, :K].mean(axis=0)
    rho = C/C[0]
    tau = 0.5
    for W in range(1, K):
        tau += rho[W]
        if W >= 5*tau:
            return float(tau)
    return float(tau)

def binder_jk(A2, A4):
    R, K = A2.shape
    a2 = A2.sum(axis=1); a4 = A4.sum(axis=1)
    S2 = a2.sum(); S4 = a4.sum(); Nt = R*K
    U = 1 - (S4/Nt)/(3*(S2/Nt)**2)
    S2j = S2 - a2; S4j = S4 - a4; Nj = (R-1)*K
    Uj = 1 - (S4j/Nj)/(3*(S2j/Nj)**2)
    err = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    Ubc = R*U - (R-1)*Uj.mean()
    return float(U), float(err), float(Ubc)

TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R = 1000; BURN = 200; K = 3000

def run_T(T, idx):
    t0 = time.time()
    seed = 900000 + 7*idx
    ens = SWEnsemble(16, R, T, seed=seed)
    for _ in range(BURN):
        ens.step()
    Ms = np.empty((R, K), dtype=np.int16)
    for k in range(K):
        ens.step()
        Ms[:, k] = ens.magsums()
    np.save(f"/work/prod_M_T{T:.2f}.npy", Ms)
    m = Ms.astype(np.float64)/256.0
    A2 = m*m; A4 = A2*A2
    U, err, Ubc = binder_jk(A2, A4)
    U1, e1, _ = binder_jk(A2[:, :K//2], A4[:, :K//2])
    U2, e2, _ = binder_jk(A2[:, K//2:], A4[:, K//2:])
    tau2 = tau_int_pooled(A2); tau4 = tau_int_pooled(A4)
    res = dict(T=T, idx=idx, seed=seed, R=R, burn=BURN, K=K, n_samples=R*K, U=U, err=err, Ubc=Ubc,
               U_half1=U1, U_half2=U2, err_half=e1, tau_int_m2=tau2, tau_int_m4=tau4,
               mean_m2=float(A2.mean()), mean_m4=float(A4.mean()), time_s=time.time()-t0)
    json.dump(res, open(f"/work/prod_T{T:.2f}.json", "w"), indent=1)
    print(f"T={T}: U={U:.5f} err={err:.5f} (Ubc={Ubc:.5f}) halves {U1:.5f}/{U2:.5f} tau(m2)={tau2:.2f} tau(m4)={tau4:.2f} "
          f"<m2>={A2.mean():.5f} n={R*K} time={time.time()-t0:.1f}s", flush=True)
    return res

t_all = time.time()
for idx, T in enumerate(TS[:7]):
    run_T(T, idx)
print("part 1 total", round(time.time()-t_all, 1))
