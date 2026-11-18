import sys, time, os
sys.path.insert(0, "/work")
import numpy as np
from ising import run_chains

os.makedirs("/work/prod", exist_ok=True)
temps_all = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
batch = [2.5, 2.6, 2.7, 2.8]
t_start = time.time()
for T in batch:
    k = temps_all.index(T)
    t0 = time.time()
    S2, S4, eqt, ts = run_chains(16, T, R=1024, n_eq=500, n_meas=5000, seed=2000 + k, ts_reps=16)
    np.savez(f"/work/prod/T{T:.1f}.npz", S2=S2, S4=S4, n=5000, eqt=eqt, ts=ts, seed=2000 + k)
    print(f"T={T} done in {time.time()-t0:.1f}s, mean<M>={ts.astype(float).mean():.3f}, eq_last={eqt[-1]:.1f}", flush=True)
print("batch total", time.time() - t_start)
