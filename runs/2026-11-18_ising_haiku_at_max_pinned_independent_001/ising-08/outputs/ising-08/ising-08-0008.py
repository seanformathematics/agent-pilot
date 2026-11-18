import sys, time
sys.path.insert(0, "/work")
import numpy as np
from ising import run_chains, jackknife_U

T = 2.8
t_start = time.time()
Us, Es = [], []
for r in range(12):
    t0 = time.time()
    S2, S4, eqt, _ = run_chains(16, T, R=1024, n_eq=500, n_meas=1000, seed=3000 + r)
    Uf, Ue, _ = jackknife_U(S2, S4, 1000, 64)
    Us.append(Uf); Es.append(Ue)
    print(f"rep {r} seed {3000+r}: U={Uf:.6f} jk_err={Ue:.6f} [{time.time()-t0:.1f}s]", flush=True)
Us = np.array(Us); Es = np.array(Es)
print("mean U over reps: %.6f" % Us.mean())
print("scatter (std over reps, ddof=1): %.6f" % Us.std(ddof=1))
print("mean within-run jackknife err: %.6f  (rms %.6f)" % (Es.mean(), np.sqrt((Es**2).mean())))
print("std error of scatter estimate ~ %.1f%%" % (100/np.sqrt(2*(len(Us)-1))))
print("total time", time.time() - t_start)
