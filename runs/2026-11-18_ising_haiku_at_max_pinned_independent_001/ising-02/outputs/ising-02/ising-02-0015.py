
import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from sw import SWEnsemble

prod = {1.5: 0.66645, 2.5: 0.38309, 2.8: 0.14622}
out = {}
t_all = time.time()
for T in [1.5, 2.5, 2.8]:
    ens = SWEnsemble(16, 1000, T, seed=int(31337 + 100*T))
    ens.S[:] = 1  # cold start: all spins up
    steps = 600
    Ms = np.empty((1000, steps), dtype=np.int16)
    for k in range(steps):
        ens.step()
        Ms[:, k] = ens.magsums()
    m = Ms.astype(np.float64)/256.0
    A2 = m*m; A4 = A2*A2
    def U_of(sl):
        a2 = A2[:, sl]; a4 = A4[:, sl]
        return 1 - a4.mean()/(3*a2.mean()**2)
    w = {"0-5": slice(0,5), "5-20": slice(5,20), "20-100": slice(20,100), "100-600": slice(100,600)}
    print(f"T={T} cold start: m2 by window:", {k: round(float(A2[:, s].mean()),5) for k,s in w.items()},
          " U by window:", {k: round(float(U_of(s)),5) for k,s in w.items()}, f"(random-start production U={prod[T]})", flush=True)
    out[str(T)] = {k: float(U_of(s)) for k,s in w.items()}
json.dump(out, open("/work/cold_start_check.json","w"), indent=1)
print("total", round(time.time()-t_all,1))
