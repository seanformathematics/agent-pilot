
import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from sw import SWEnsemble

def binder_jk(A2, A4):
    R, K = A2.shape
    a2 = A2.sum(axis=1); a4 = A4.sum(axis=1)
    S2 = a2.sum(); S4 = a4.sum(); Nt = R*K
    U = 1 - (S4/Nt)/(3*(S2/Nt)**2)
    S2j = S2 - a2; S4j = S4 - a4; Nj = (R-1)*K
    Uj = 1 - (S4j/Nj)/(3*(S2j/Nj)**2)
    err = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return float(U), float(err)

TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R = 1000; BURN = 200; K = 3000
t_all = time.time()
for idx, T in enumerate(TS):
    if idx >= 7: continue
    t0 = time.time()
    seed = 7000 + idx
    ens = SWEnsemble(16, R, T, seed=seed)
    for _ in range(BURN):
        ens.step()
    Ms = np.empty((R, K), dtype=np.int16)
    for k in range(K):
        ens.step()
        Ms[:, k] = ens.magsums()
    m = Ms.astype(np.float64)/256.0
    A2 = m*m; A4 = A2*A2
    U, err = binder_jk(A2, A4)
    np.savez(f"/work/prod2_sums_T{T:.2f}.npz", a2=A2.sum(axis=1), a4=A4.sum(axis=1), K=K, seed=seed)
    print(f"T={T} seed={seed}: U={U:.5f} jk_err={err:.5f} ({time.time()-t0:.1f}s)", flush=True)
print("part A total", round(time.time()-t_all, 1))
