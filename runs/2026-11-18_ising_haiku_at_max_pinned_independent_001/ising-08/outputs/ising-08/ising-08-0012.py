import sys, json
sys.path.insert(0, "/work")
import numpy as np
from ising import jackknife_U
from scipy.interpolate import CubicSpline

temps = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
rows = []
for T in temps:
    d = np.load(f"/work/prod/T{T:.1f}.npz")
    S2, S4, n = d["S2"], d["S4"], int(d["n"])
    Uf, _, _ = jackknife_U(S2, S4, n, 1024)
    # leave-one-replica-out jackknife error (G = 1024)
    R = S2.shape[0]
    tot2 = S2.sum(); tot4 = S4.sum(); Ntot = float(n) * R
    Uloo = np.array([1.0 - ((tot4 - S4[r]) / (Ntot - n)) / (3.0 * ((tot2 - S2[r]) / (Ntot - n)) ** 2) for r in range(R)])
    err = np.sqrt((R - 1) / R * np.sum((Uloo - Uloo.mean()) ** 2))
    rows.append(dict(T=T, U=float(Uf), U_err=float(err), n_samples=int(R * n), seed=int(d["seed"])))
    print(f"T={T:.1f} U={Uf:.8f} U_err(LOO jackknife)={err:.4e} n={R*n} seed={int(d['seed'])}")
json.dump(rows, open("/work/final_results_loo.json", "w"), indent=1)

Tt = np.array([r["T"] for r in rows]); Uu = np.array([r["U"] for r in rows])
Tc = 2.0 / np.log(1 + np.sqrt(2))
cs = CubicSpline(Tt, Uu)
print("Tc = %.6f ; spline estimate of U_16(Tc) = %.5f" % (Tc, cs(Tc)))
# local polynomial around Tc using 2.1..2.5 (degree 3)
mask = (Tt >= 2.1) & (Tt <= 2.5)
p = np.polyfit(Tt[mask] - Tc, Uu[mask], 3)
print("cubic fit (2.1..2.5) estimate at Tc = %.5f" % np.polyval(p, 0.0))
print("finite:", all(np.isfinite(r["U"]) and np.isfinite(r["U_err"]) and r["U_err"] > 0 for r in rows))
