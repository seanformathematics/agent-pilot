
import sys, time
import numpy as np
sys.path.insert(0, "/work")
from prod import run_T

Ts = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t00 = time.time()
for T in Ts:
    seed = 300000 + int(round(T * 10))
    t0 = time.time()
    out, blen = run_T(T, seed, start="cold")
    np.savez(f"/work/prod2_T{T:.1f}.npz", blen=blen, seed=seed, **out)
    print(f"T={T} seed={seed} done in {time.time()-t0:.1f}s (total {time.time()-t00:.0f}s)", flush=True)
