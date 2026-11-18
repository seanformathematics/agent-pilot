import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from sw import run_sw, binder_jackknife

out = {}
for T in [2.3, 2.8]:
    Us, errs = [], []
    t0 = time.time()
    for k in range(8):
        seed = 880000 + 100*int(T*10) + k
        M2s, M4s, nit = run_sw(16, T, R=256, n_therm=200, n_meas=1000, seed=seed)
        U, err, Ujk = binder_jackknife(M2s, M4s, 1000)
        Us.append(U); errs.append(err)
    Us = np.array(Us); errs = np.array(errs)
    print(f"T={T}: seed-scatter std(U)={Us.std(ddof=1):.5f}  mean U={Us.mean():.5f} (se of mean {Us.std(ddof=1)/np.sqrt(8):.5f}) "
          f"mean jackknife err={errs.mean():.5f}  ratio={Us.std(ddof=1)/errs.mean():.2f}  time={time.time()-t0:.1f}s", flush=True)
    out[str(T)] = dict(Us=Us.tolist(), errs=errs.tolist())
json.dump(out, open("/work/scatter_check.json", "w"))
