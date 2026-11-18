import sys, json
sys.path.insert(0, "/work")
import numpy as np
from ising import jackknife_U

temps = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
pilot = json.load(open("/work/pilot_results.json"))
rows = []
for T in temps:
    d = np.load(f"/work/prod/T{T:.1f}.npz")
    S2, S4, n = d["S2"], d["S4"], int(d["n"])
    Uf, Ue, Ubc = jackknife_U(S2, S4, n, 64)
    # also a 32-group and 128-group jackknife for stability of the error estimate
    _, Ue32, _ = jackknife_U(S2, S4, n, 32)
    _, Ue128, _ = jackknife_U(S2, S4, n, 128)
    p = pilot[str(T)]
    pull = (Uf - p["U"]) / np.sqrt(Ue**2 + p["err"]**2)
    rows.append((T, Uf, Ue, Ubc, Ue32, Ue128, p["U"], p["err"], pull))
    print(f"T={T}: U_prod={Uf:.6f} +- {Ue:.6f} (G32 {Ue32:.6f}, G128 {Ue128:.6f}) bias-corr={Ubc:.6f} | pilot U={p['U']:.6f} +- {p['err']:.6f} | pull={pull:+.2f}")
pulls = np.array([r[-1] for r in rows])
print("pull mean %.2f std %.2f" % (pulls.mean(), pulls.std(ddof=1)))
