import numpy as np, time, sys
sys.path.insert(0, "/work")
from ising_mc2 import run
def U_of(M2, M4):
    return 1 - M4.mean() / (3 * M2.mean()**2)
t0=time.time()
for i,T in enumerate([2.3, 2.6, 2.8]):
    seed = 3000 + i
    out = run(T, seed)
    np.savez(f"/work/check_T{T:.2f}_s{seed}.npz", T=T, seed=seed, **out)
    U2 = U_of(out["M2"], out["M4"])
    d = np.load(f"/work/res_T{T:.2f}_s{2000+[2.3,2.6,2.8].index(T)*0 + (0 if T==2.3 else (4 if T==2.6 else 6))}.npz") if False else None
    print(f"T={T} seed={seed} U_check={U2:.5f} time={time.time()-t0:.1f}", flush=True)
import glob
for T, s0 in [(2.3, 2001), (2.6, 2004), (2.8, 2006)]:
    d = np.load(f"/work/res_T{T:.2f}_s{s0}.npz")
    print(f"T={T} original U={U_of(d['M2'], d['M4']):.5f}")