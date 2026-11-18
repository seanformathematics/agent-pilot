import sys, time, numpy as np
sys.path.insert(0, '/work')
from sw import SW

lib = r'''
import numpy as np
import sys
sys.path.insert(0, '/work')
from sw import SW


def run_T(T, R, nburn, nmeas, seed, N=16, block=100):
    sim = SW(N, R, seed)
    beta = 1.0 / T
    for _ in range(nburn):
        sim.sweep(beta)
    S2 = np.zeros(R, dtype=np.int64)
    S4 = np.zeros(R, dtype=np.int64)
    S1 = np.zeros(R, dtype=np.int64)
    nb = nmeas // block
    bS2 = np.zeros(nb)
    bS4 = np.zeros(nb)
    flips = 0
    Mprev = sim.mag()
    for t in range(nmeas):
        sim.sweep(beta)
        M = sim.mag()
        M2 = M * M
        M4 = M2 * M2
        S2 += M2
        S4 += M4
        S1 += M
        flips += int(np.count_nonzero((M > 0) != (Mprev > 0)))
        Mprev = M
        b = t // block
        if b < nb:
            bS2[b] += float(M2.sum())
            bS4[b] += float(M4.sum())
    return dict(S2=S2, S4=S4, S1=S1, bS2=bS2, bS4=bS4, flips=flips,
                R=R, nburn=nburn, nmeas=nmeas, seed=seed, T=T, N=N, block=block)


def U_jack(S2r, S4r, nr):
    R = len(S2r)
    S2 = float(S2r.sum())
    S4 = float(S4r.sum())
    n = float(nr) * R
    U = 1 - (S4 / n) / (3 * (S2 / n) ** 2)
    S2m = S2 - S2r.astype(float)
    S4m = S4 - S4r.astype(float)
    nm = n - nr
    Um = 1 - (S4m / nm) / (3 * (S2m / nm) ** 2)
    err = np.sqrt((R - 1) / R * np.sum((Um - Um.mean()) ** 2))
    return U, err
'''
with open('/work/prodlib.py', 'w') as f:
    f.write(lib)
from prodlib import run_T, U_jack

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
R, nburn, nmeas = 2000, 100, 1000
for k, T in enumerate(Ts):
    seed = 1000 + k
    t0 = time.time()
    d = run_T(T, R, nburn, nmeas, seed)
    np.savez(f'/work/res_T{T:.2f}_s{seed}.npz', **{kk: np.asarray(v) for kk, v in d.items()})
    U, err = U_jack(d['S2'], d['S4'], nmeas)
    nb = nmeas // d['block']
    bU = 1 - (d['bS4'] / (R * d['block'])) / (3 * (d['bS2'] / (R * d['block'])) ** 2)
    mM = d['S1'].sum() / (R * nmeas)
    m2 = d['S2'].sum() / (R * nmeas)
    print(f"T={T:.2f} seed={seed}: U={U:.6f} +- {err:.6f}  n={R*nmeas}  <M>={mM:+.3f}  <M^2>={m2:.2f}  signflip/sweep={d['flips']/(R*nmeas):.3f}  blockU={np.round(bU,4).tolist()}  time={time.time()-t0:.1f}s", flush=True)
