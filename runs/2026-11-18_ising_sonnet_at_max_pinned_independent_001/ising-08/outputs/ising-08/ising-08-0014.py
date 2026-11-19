
import sys, time, os
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *

N = 16
allT = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
def pooled_sw(T):
    A = np.load(f'/work/prod/sums_T{T}.npy').sum(axis=1)
    B = np.load(f'/work/prod2/sums_T{T}.npy').sum(axis=1)
    return np.concatenate([A[:, 0::2] + A[:, 1::2], B], axis=1)

def tau_int(x, c=6.0):
    x = np.asarray(x, float); x = x - x.mean(); n = x.size
    f = np.fft.rfft(x, 2 * n); acf = np.fft.irfft(f * np.conj(f))[:n]; acf /= acf[0]
    tau = 0.5
    for W in range(1, n):
        tau = 0.5 + acf[1:W + 1].sum()
        if W >= c * tau: break
    return tau

t0 = time.time()
configs = [(2.3, 256, 5000, 100, 400, 61000), (2.2, 128, 3000, 50, 400, 61001), (2.4, 128, 3000, 50, 400, 61002)]
for (T, R, burn, nb, bl, seed) in configs:
    n = nb * bl
    out = run_hb(N, T, R, burn, nb, bl, seed=seed, init='mixed')
    np.save(f'/work/hb/sums2_T{T}.npy', out)
    per = out.sum(axis=1)
    Uh, eh, _ = U_from_units(per[0], per[1], n)
    Uc, ec, _ = U_from_units(per[0][:R//2], per[1][:R//2], n)
    Uhot, ehot, _ = U_from_units(per[0][R//2:], per[1][R//2:], n)
    Em, Ee = mean_from_units(per[2], n); Eex = E_exact_torus(T, N)
    u = pooled_sw(T); Us, es, _ = U_from_units(u[2], u[3], 8000)
    # block-level error check: use blocks (400 sweeps) as units
    bu0 = out[0].reshape(-1); bu1 = out[1].reshape(-1)
    Ub, eb, _ = U_from_units(bu0, bu1, bl)
    # tau_int of the block-summed m2 series of replica 0 (block 400 sweeps - only rough)
    print(f"T={T}: HB U={Uh:.5f}+-{eh:.2e} (block-jackknife err {eb:.2e}) | cold-start {Uc:.5f}+-{ec:.1e}, hot-start {Uhot:.5f}+-{ehot:.1e} (z={(Uc-Uhot)/np.hypot(ec,ehot):+.2f}) | SW pooled {Us:.5f}+-{es:.1e} | z(HB-SW)={(Uh-Us)/np.hypot(eh,es):+.2f} | HB E z vs exact={(Em-Eex)/Ee:+.2f} | {time.time()-t0:.0f}s", flush=True)
