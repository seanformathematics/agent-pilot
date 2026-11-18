import sys, time
sys.path.insert(0, '/work')
import numpy as np
from sw import sw_setup, sw_sweep

Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
R = 512; N = 16
geo = sw_setup(R, N)
t0 = time.time()
for k, T in enumerate(Ts):
    seed = 2000 + k
    rng = np.random.default_rng(seed)
    p = 1 - np.exp(-2.0/T)
    s = np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)
    for it in range(300):
        s = sw_sweep(s, p, rng, geo)
    nmeas = 1500
    B = 10
    nb = nmeas // B
    bm2 = np.zeros((nb, R)); bm4 = np.zeros((nb, R))
    for it in range(nmeas):
        s = sw_sweep(s, p, rng, geo)
        m = s.reshape(R, -1).mean(axis=1)
        bm2[it//B] += m**2 / B
        bm4[it//B] += m**4 / B
    np.savez(f'/work/prod_T{T:.1f}.npz', bm2=bm2, bm4=bm4, seed=seed)
    print(f"T={T} done t={time.time()-t0:.1f}s", flush=True)
