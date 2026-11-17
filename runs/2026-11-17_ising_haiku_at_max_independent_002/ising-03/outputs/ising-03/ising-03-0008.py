import sys, glob, json
sys.path.insert(0, "/work")
import numpy as np
from prod import U_jack

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
final = []
print("T   U(full)            U_err          U(first half)  U(second half)  z_halves")
for k, T in enumerate(Ts):
    seed = 777001 + k
    f = glob.glob(f"/work/data/prod_T{T:.2f}_seed{seed}.npz")[0]
    d = np.load(f)
    S2 = d["S2"]; S4 = d["S4"]; nmeas = int(d["nmeas"]); assert int(d["seed"]) == seed
    U, err = U_jack(S2, S4, nmeas, G=100)
    nb = S2.shape[1]
    Uh1, eh1 = U_jack(S2[:, :nb//2], S4[:, :nb//2], nmeas//2, G=100)
    Uh2, eh2 = U_jack(S2[:, nb//2:], S4[:, nb//2:], nmeas//2, G=100)
    zh = (Uh1 - Uh2)/np.sqrt(eh1**2 + eh2**2)
    print(f"{T:.1f}  {U:.10f}  {err:.6e}  {Uh1:.8f}  {Uh2:.8f}  {zh:+.2f}")
    final.append({"T": T, "U": float(U), "U_err": float(err), "n": int(S2.size*20), "seed": seed})
json.dump(final, open("/work/final_results.json", "w"), indent=1)
print("max U_err =", max(x["U_err"] for x in final), " min U =", min(x["U"] for x in final), " max U =", max(x["U"] for x in final))
