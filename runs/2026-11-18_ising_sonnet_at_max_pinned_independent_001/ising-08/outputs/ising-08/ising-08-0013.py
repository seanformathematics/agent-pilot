
import sys, time, os
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *
os.makedirs('/work/hb', exist_ok=True)

N = 16
allT = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def pooled_sw(T):
    A = np.load(f'/work/prod/sums_T{T}.npy').sum(axis=1)       # (6, 256) 4000 sweeps each
    B = np.load(f'/work/prod2/sums_T{T}.npy').sum(axis=1)      # (6, 128) 8000 sweeps each
    # pair up A replicas -> 128 units with 8000 measurements, concatenate with B units
    Ap = A[:, 0::2] + A[:, 1::2]
    units = np.concatenate([Ap, B], axis=1)
    return units

R, burn, nb, bl = 128, 2000, 50, 200
n = nb * bl
t0 = time.time()
for T in (1.5, 2.0, 2.3, 2.5, 2.8):
    i = allT.index(T)
    init = 'cold' if T < 2.25 else 'mixed'
    out = run_hb(N, T, R, burn, nb, bl, seed=50000 + i, init=init)
    np.save(f'/work/hb/sums_T{T}.npy', out)
    per = out.sum(axis=1)
    Uh, eh, _ = U_from_units(per[0], per[1], n)
    Em, Ee = mean_from_units(per[2], n); Eex = E_exact_torus(T, N)
    u = pooled_sw(T)
    Us, es, _ = U_from_units(u[2], u[3], 8000)
    print(f"T={T}: heat-bath U={Uh:.5f}+-{eh:.2e} | SW pooled (improved) U={Us:.6f}+-{es:.1e} | z={(Uh-Us)/np.hypot(eh,es):+.2f} | HB E z vs exact={(Em-Eex)/Ee:+.2f} | {time.time()-t0:.0f}s", flush=True)
