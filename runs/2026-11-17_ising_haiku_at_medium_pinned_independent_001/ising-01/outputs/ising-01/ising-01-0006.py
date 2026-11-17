
import sys, time, os
sys.path.insert(0, '/work')
import numpy as np
from sw import make_state, sw_step, measure

def run_blocks(T, N=16, R=1024, n_burn=200, n_blocks=12, steps=100, seed=1):
    rng = np.random.default_rng(seed)
    S = make_state(R, N, rng)
    for _ in range(n_burn):
        S = sw_step(S, T, rng, N)
    m2 = np.zeros((R, n_blocks)); m4 = np.zeros((R, n_blocks)); am = np.zeros((R, n_blocks)); en = np.zeros((R, n_blocks))
    for b in range(n_blocks):
        for k in range(steps):
            S = sw_step(S, T, rng, N)
            m, e = measure(S, N)
            q = m * m
            m2[:, b] += q; m4[:, b] += q * q; am[:, b] += np.abs(m); en[:, b] += e
    m2 /= steps; m4 /= steps; am /= steps; en /= steps
    return dict(m2=m2, m4=m4, am=am, en=en)

os.makedirs('/work/res', exist_ok=True)
temps = [1.5, 1.6, 1.7, 1.8]
t0 = time.time()
for i, T in enumerate(temps):
    d = run_blocks(T, seed=20240 + i)
    np.savez(f'/work/res/T{T:.2f}_s{20240+i}.npz', T=T, seed=20240+i, **d)
    print(f"done T={T} t={time.time()-t0:.1f}s", flush=True)
