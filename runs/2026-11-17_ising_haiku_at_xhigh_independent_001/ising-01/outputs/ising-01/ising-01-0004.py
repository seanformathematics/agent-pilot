import numpy as np, time, sys
sys.path.insert(0, "/work")
from ising_mc2 import run
temps = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t0 = time.time()
for i, T in enumerate(temps):
    seed = 2000 + i
    out = run(T, seed)
    np.savez(f"/work/res_T{T:.2f}_s{seed}.npz", T=T, seed=seed, **out)
    M2 = out["M2"].mean(); M4 = out["M4"].mean()
    print(f"T={T} seed={seed} U={1-M4/(3*M2*M2):.5f}  E/N={out['EN'].mean():.4f} <|m|>={out['M1'].mean():.4f} time={time.time()-t0:.1f}", flush=True)