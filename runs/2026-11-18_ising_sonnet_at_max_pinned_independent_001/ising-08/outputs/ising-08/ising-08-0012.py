
import sys, time, os
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *
os.makedirs('/work/prod2', exist_ok=True)

N = 16; R = 128; burn = 2000; nb = 80; bl = 100
allT = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
Tlist = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t0 = time.time()
for T in Tlist:
    i = allT.index(T)
    out = run_sw(N, T, R, burn, nb, bl, seed=40000 + i, init='hot')
    np.save(f'/work/prod2/sums_T{T}.npy', out)
    per = out.sum(axis=1); n = nb * bl
    Ub, eb, bb = U_from_units(per[2], per[3], n)
    Upb, epb, _ = U_from_units(per[0], per[1], n)
    A = np.load(f'/work/prod/sums_T{T}.npy').sum(axis=1)
    Ua, ea, _ = U_from_units(A[2], A[3], 4000)
    Em, Ee = mean_from_units(per[4], n); Eex = E_exact_torus(T, N)
    print(f"T={T}: B: U_imp={Ub:.6f}+-{eb:.2e} (plain {Upb:.6f}+-{epb:.2e}) | A: {Ua:.6f}+-{ea:.2e} | z(A-B)={(Ua-Ub)/np.hypot(ea,eb):+.2f} | E_B z vs exact={(Em-Eex)/Ee:+.2f} | {time.time()-t0:.0f}s", flush=True)
