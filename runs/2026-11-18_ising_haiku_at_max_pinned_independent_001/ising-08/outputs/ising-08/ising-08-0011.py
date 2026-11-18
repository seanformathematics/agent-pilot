import sys, json
sys.path.insert(0, "/work")
import numpy as np
from ising import jackknife_U

final = json.load(open("/work/final_results.json"))
print("T     U(full)      jk G64     jk G256    jk G1024(leave-one-replica)   boot(4000)")
rng = np.random.default_rng(777)
B = 4000
for o in final:
    T = o["T"]
    d = np.load(f"/work/prod/T{T:.1f}.npz")
    S2, S4, n = d["S2"], d["S4"], int(d["n"])
    Uf, e64, _ = jackknife_U(S2, S4, n, 64)
    _, e256, _ = jackknife_U(S2, S4, n, 256)
    _, e1024, _ = jackknife_U(S2, S4, n, 1024)
    S2f = S2.astype(np.float64); S4f = S4.astype(np.float64)
    R = S2f.shape[0]
    idx = rng.integers(0, R, size=(B, R))
    b2 = S2f[idx].sum(axis=1); b4 = S4f[idx].sum(axis=1)
    Ub = 1.0 - (b4 / (R * n)) / (3.0 * (b2 / (R * n)) ** 2)
    print(f"{T:.1f}  {Uf:.8f}  {e64:.3e}  {e256:.3e}  {e1024:.3e}  {Ub.std(ddof=1):.3e}")
