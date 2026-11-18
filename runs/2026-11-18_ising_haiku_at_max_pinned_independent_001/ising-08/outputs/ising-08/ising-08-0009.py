import sys, json
sys.path.insert(0, "/work")
import numpy as np
from ising import jackknife_U

temps = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
temps_all = temps
out = []
for T in temps:
    d = np.load(f"/work/prod/T{T:.1f}.npz")
    S2, S4, n = d["S2"], d["S4"], int(d["n"])
    seed = int(d["seed"])
    Uf, Ue, Ubc = jackknife_U(S2, S4, n, 64)
    nsamp = int(S2.shape[0] * n)
    out.append(dict(T=T, U=Uf, U_err=Ue, n_samples=nsamp, seed=seed, Ubc=Ubc))
    print(f"T={T:.1f}  U={Uf:.8f}  U_err={Ue:.3e}  n_samples={nsamp}  seed={seed}  bias_corr_diff={Ubc-Uf:.1e}")
ok = all(np.isfinite(o["U"]) and np.isfinite(o["U_err"]) and o["U_err"] > 0 and o["U"] <= 2/3 for o in out)
print("all finite, U_err>0, U<=2/3:", ok)
json.dump(out, open("/work/final_results.json", "w"), indent=1)
print("seeds:", [o["seed"] for o in out])
print("max U_err:", max(o["U_err"] for o in out))
