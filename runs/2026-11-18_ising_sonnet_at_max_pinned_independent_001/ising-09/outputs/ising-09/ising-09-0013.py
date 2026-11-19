
import sys, time
sys.path.insert(0, "/work")
import numpy as np
from run import run_T, jackknife_U

t0=time.time()
K = 40
for T in [1.5, 2.0, 2.5]:
    us=[]; es=[]
    for k in range(K):
        acc = run_T(16, T, R=100, E=200, S=200, L=20, seed=700000 + 1000*int(round(T*10)) + k, improved=False)
        U, e, _ = jackknife_U(acc[0].mean(axis=1), acc[1].mean(axis=1))
        us.append(U); es.append(e)
    u=np.array(us); e=np.array(es)
    print(f"T={T}: mean U={u.mean():.6f} std over repeats={u.std(ddof=1):.6f}  rms reported err={np.sqrt(np.mean(e**2)):.6f}  ratio={u.std(ddof=1)/np.sqrt(np.mean(e**2)):.3f} (expect 1 +- {1/np.sqrt(2*(K-1)):.2f})  std z={((u-u.mean())/e).std(ddof=1):.3f}   [{time.time()-t0:.0f}s]")
