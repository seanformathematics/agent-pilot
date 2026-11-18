
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

out = {}
t_all = time.time()
plan = [(2.5, 3101), (2.5, 3102), (2.5, 3103), (2.5, 3104), (2.27, 3201), (2.27, 3202)]
for T, s in plan:
    t0 = time.time()
    ens = SWEnsemble(16, 1000, T, seed=s)
    for _ in range(200):
        ens.step()
    Ms = np.empty((1000, 3000), dtype=np.int16)
    for k in range(3000):
        ens.step()
        Ms[:, k] = ens.magsums()
    m = Ms.astype(np.float64)/256.0
    A2 = m*m; A4 = A2*A2
    U, err = binder_jk(A2, A4)
    out.setdefault(str(T), []).append((s, U, err))
    print(f"T={T} seed={s}: U={U:.5f} jk_err={err:.5f}  time {time.time()-t0:.1f}s", flush=True)
json.dump(out, open("/work/extra_runs.json", "w"), indent=1)
for T in out:
    Us = np.array([u for _, u, _ in out[T]]); es = np.array([e for _, _, e in out[T]])
    print(f"T={T}: runs={len(Us)} mean U={Us.mean():.5f} sd across runs={Us.std(ddof=1):.5f} mean jk err={es.mean():.5f}")
print("total", round(time.time()-t_all, 1))
