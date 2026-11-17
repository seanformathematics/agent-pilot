import sys, time
sys.path.insert(0, "/work")
import numpy as np
from prod import run_T, U_jack

Ts = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t0 = time.time()
for k, T in enumerate(Ts):
    seed = 777008 + k
    S2, S4 = run_T(T, seed)
    U, err = U_jack(S2, S4, 1000)
    print(f"T={T:.2f} seed={seed} U={U:.6f} err={err:.3e} n={S2.size*20} elapsed={time.time()-t0:.0f}s", flush=True)
