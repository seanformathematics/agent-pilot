import sys, time, numpy as np
sys.path.insert(0, '/work')
from prodlib import run_T, U_jack

Ts = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R, nburn, nmeas = 2000, 100, 1000
for k, T in enumerate(Ts):
    seed = 1007 + k
    t0 = time.time()
    d = run_T(T, R, nburn, nmeas, seed)
    np.savez(f'/work/res_T{T:.2f}_s{seed}.npz', **{kk: np.asarray(v) for kk, v in d.items()})
    U, err = U_jack(d['S2'], d['S4'], nmeas)
    bU = 1 - (d['bS4'] / (R * d['block'])) / (3 * (d['bS2'] / (R * d['block'])) ** 2)
    mM = d['S1'].sum() / (R * nmeas)
    m2 = d['S2'].sum() / (R * nmeas)
    print(f"T={T:.2f} seed={seed}: U={U:.6f} +- {err:.6f}  n={R*nmeas}  <M>={mM:+.3f}  <M^2>={m2:.2f}  signflip/sweep={d['flips']/(R*nmeas):.3f}  blockU={np.round(bU,4).tolist()}  time={time.time()-t0:.1f}s", flush=True)
