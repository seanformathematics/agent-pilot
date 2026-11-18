import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from sw import run_sw, binder_jackknife

sc = json.load(open("/work/scatter_check.json"))
for T in ["2.3", "2.8"]:
    print("scatter U values T=", T, np.round(sc[T]["Us"], 5))

extra = {}
for T, seed in [(2.8, 31337), (2.3, 31338)]:
    t0 = time.time()
    M2s, M4s, nit = run_sw(16, T, R=1024, n_therm=200, n_meas=1000, seed=seed)
    U, err, Ujk = binder_jackknife(M2s, M4s, 1000)
    extra[str(T)] = dict(seed=seed, U=float(U), U_err=float(err))
    print(f"extra run T={T} seed={seed}: U={U:.5f} +- {err:.5f}  time={time.time()-t0:.1f}s", flush=True)
json.dump(extra, open("/work/extra_check.json", "w"))
