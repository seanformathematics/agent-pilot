import sys, json
sys.path.insert(0, "/work")
import numpy as np

temps = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
rng = np.random.default_rng(777)
B = 4000
final = json.load(open("/work/final_results.json"))
print("T        U_jk(G64)     err_jk        err_boot(replica bootstrap)   ratio")
for o in final:
    T = o["T"]
    d = np.load(f"/work/prod/T{T:.1f}.npz")
    S2 = d["S2"].astype(np.float64); S4 = d["S4"].astype(np.float64); n = float(d["n"])
    R = S2.shape[0]
    idx = rng.integers(0, R, size=(B, R))
    b2 = S2[idx].sum(axis=1); b4 = S4[idx].sum(axis=1)
    Ub = 1.0 - (b4 / (R * n)) / (3.0 * (b2 / (R * n)) ** 2)
    eb = Ub.std(ddof=1)
    print(f"{T:.1f}  {o['U']:.8f}  {o['U_err']:.3e}  {eb:.3e}  {eb/o['U_err']:.3f}")
