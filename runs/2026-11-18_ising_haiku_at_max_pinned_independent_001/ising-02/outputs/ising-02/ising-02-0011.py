
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

res = []
t_all = time.time()
for rep in range(6):
    s = 5101 + rep
    t0 = time.time()
    ens = SWEnsemble(16, 1000, 2.5, seed=s)
    for _ in range(200):
        ens.step()
    Ms = np.empty((1000, 3000), dtype=np.int16)
    for k in range(3000):
        ens.step()
        Ms[:, k] = ens.magsums()
    m = Ms.astype(np.float64)/256.0
    A2 = m*m; A4 = A2*A2
    U, err = binder_jk(A2, A4)
    res.append((s, U, err))
    print(f"seed {s}: U={U:.5f} jk_err={err:.5f} ({time.time()-t0:.1f}s)", flush=True)
json.dump(res, open("/work/rep1000_K3000_T2.5.json", "w"))
Us = np.array([r[1] for r in res]); es = np.array([r[2] for r in res])
print(f"6 new runs R=1000,K=3000: mean U={Us.mean():.5f} sd={Us.std(ddof=1):.5f} mean jk err={es.mean():.5f}")
allU = np.array([0.38417, 0.38390, 0.38237, 0.38312, 0.38230] + list(Us))
print(f"all 11 K=3000 runs: mean={allU.mean():.5f} sd={allU.std(ddof=1):.5f}; jk ~0.00037; ratio={allU.std(ddof=1)/0.00037:.2f}")
print("total", round(time.time()-t_all, 1))
