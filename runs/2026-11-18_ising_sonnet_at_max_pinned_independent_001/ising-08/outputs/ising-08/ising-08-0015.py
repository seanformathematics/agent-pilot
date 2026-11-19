
import sys, json
sys.path.insert(0, '/work')
import numpy as np
from ising_lib import *

N = 16
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
res = []
zE = []; zAB = []
print(" T   |   U (improved) +- jackknife err | block-jk err | invvar(A,B) U +- err | U plain +- err (z vs impr) | z_E | bias est")
for T in Ts:
    outA = np.load(f'/work/prod/sums_T{T}.npy')      # (6, 40 blocks, 256 reps), blocks of 100 sweeps
    outB = np.load(f'/work/prod2/sums_T{T}.npy')     # (6, 80 blocks, 128 reps)
    A = outA.sum(axis=1); B = outB.sum(axis=1)
    units = np.concatenate([A[:, 0::2] + A[:, 1::2], B], axis=1)     # 256 independent units x 8000 measurements
    cnt = 8000
    U, e, bias = U_from_units(units[2], units[3], cnt)
    Up, ep, _ = U_from_units(units[0], units[1], cnt)
    # block-level jackknife (units = blocks of 100 sweeps)
    b2 = np.concatenate([outA[2].reshape(-1), outB[2].reshape(-1)]); b4 = np.concatenate([outA[3].reshape(-1), outB[3].reshape(-1)])
    Ub, eb, _ = U_from_units(b2, b4, 100)
    # inverse-variance combination of the two independent productions
    Ua, ea, _ = U_from_units(A[2], A[3], 4000); Ubb, ebb, _ = U_from_units(B[2], B[3], 8000)
    w = np.array([1/ea**2, 1/ebb**2]); Uc = (w * np.array([Ua, Ubb])).sum() / w.sum(); ec = 1/np.sqrt(w.sum())
    zAB.append((Ua - Ubb) / np.hypot(ea, ebb))
    Em, Ee = mean_from_units(units[4], cnt); Eex = E_exact_torus(T, N); zE.append((Em - Eex) / Ee)
    nsamp = units.shape[1] * cnt
    print(f"{T:.1f} | {U:.6f} +- {e:.2e}          | {eb:.2e}     | {Uc:.6f} +- {ec:.2e} | {Up:.6f} +- {ep:.2e} ({(Up-U)/ep:+.2f}) | {zE[-1]:+.2f} | {bias:+.1e}")
    res.append(dict(T=T, U=float(U), U_err=float(e), n_samples=int(nsamp), U_plain=float(Up), U_plain_err=float(ep),
                    U_err_block_jackknife=float(eb), U_invvar=float(Uc), U_invvar_err=float(ec),
                    E=float(Em), E_err=float(Ee), E_exact=float(Eex)))
zE = np.array(zE); zAB = np.array(zAB)
print("chi2/14 for energy vs exact Kaufman (pooled):", np.sum(zE**2) / 14, " ; chi2/14 for A-vs-B U:", np.sum(zAB**2) / 14)
print("max U_err:", max(r['U_err'] for r in res), " min U_err:", min(r['U_err'] for r in res))
json.dump(res, open('/work/final_results.json', 'w'), indent=1)
