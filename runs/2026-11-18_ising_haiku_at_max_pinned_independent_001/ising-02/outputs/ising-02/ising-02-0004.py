
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
    return U, err

out = {}
t_all = time.time()
for T in [2.27, 2.5, 2.8]:
    Us = []; errs = []
    t0 = time.time()
    for rep in range(20):
        seed = 555000 + 100*int(T*100) + rep
        ens = SWEnsemble(16, 100, T, seed=seed)
        for _ in range(200):
            ens.step()
        Ms = np.empty((100, 1000), dtype=np.int16)
        for k in range(1000):
            ens.step()
            Ms[:, k] = ens.magsums()
        m = Ms.astype(np.float64)/256.0
        A2 = m*m; A4 = A2*A2
        U, err = binder_jk(A2, A4)
        Us.append(U); errs.append(err)
    Us = np.array(Us); errs = np.array(errs)
    sd_across = Us.std(ddof=1)
    rms_jk = np.sqrt(np.mean(errs**2))
    out[T] = dict(mean_U=float(Us.mean()), sd_across=float(sd_across), mean_jk=float(errs.mean()), rms_jk=float(rms_jk))
    print(f"T={T}: 20 repeats (100 rep x 1000 steps each): mean U={Us.mean():.5f}, "
          f"sd across repeats={sd_across:.5f}, mean in-repeat jackknife err={errs.mean():.5f} (rms {rms_jk:.5f}), time {time.time()-t0:.1f}s")
print("total", round(time.time()-t_all, 1))
json.dump({str(k): v for k, v in out.items()}, open("/work/validation_repeats.json", "w"), indent=1)
