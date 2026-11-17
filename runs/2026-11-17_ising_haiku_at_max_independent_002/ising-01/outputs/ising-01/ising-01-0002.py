
import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
src = r'''
import numpy as np

def wolff_run(N, T, R, nsteps, seed, s0=None):
    """Vectorized Wolff: R independent replicas advanced in lockstep.
    Returns magnetization sums M_t = sum_i s_i (int16), shape (nsteps, R), and final spins."""
    rng = np.random.default_rng(seed)
    p = 1.0 - np.exp(-2.0 / T)
    if s0 is None:
        s = rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1
    else:
        s = s0.copy()
    s = s.astype(np.int8)
    ar = np.arange(R)
    Mt = np.empty((nsteps, R), dtype=np.int16)
    for t in range(nsteps):
        sr = np.roll(s, -1, axis=2)
        sd = np.roll(s, -1, axis=1)
        open_r = (s == sr) & (rng.random((R, N, N)) < p)
        open_d = (s == sd) & (rng.random((R, N, N)) < p)
        seeds = rng.integers(0, N * N, size=R)
        C = np.zeros((R, N * N), dtype=bool)
        C[ar, seeds] = True
        C = C.reshape(R, N, N)
        while True:
            newC = (C | np.roll(C & open_r, 1, axis=2) | (open_r & np.roll(C, -1, axis=2))
                    | np.roll(C & open_d, 1, axis=1) | (open_d & np.roll(C, -1, axis=1)))
            if np.array_equal(newC, C):
                break
            C = newC
        s = np.where(C, -s, s).astype(np.int8)
        Mt[t] = s.reshape(R, -1).sum(axis=1, dtype=np.int32)
    return Mt, s
'''
with open('/work/ising.py', 'w') as f:
    f.write(src)

import sys
sys.path.insert(0, '/work')
from ising import wolff_run

def exact_UU(N, T):
    n = N * N
    idx = np.arange(2 ** n, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)
    S = (2 * bits - 1).reshape(-1, N, N).astype(np.int64)
    E = -((S * np.roll(S, -1, axis=2)).sum(axis=(1, 2)) + (S * np.roll(S, -1, axis=1)).sum(axis=(1, 2)))
    w = np.exp(-(E - E.min()) / T)
    M = S.sum(axis=(1, 2)) / n
    Z = w.sum()
    m2 = (w * M ** 2).sum() / Z
    m4 = (w * M ** 4).sum() / Z
    e = (w * E).sum() / Z / (2 * n)
    return 1 - m4 / (3 * m2 ** 2), m2, m4, e

N = 4
print("Validation N=4 (exact vs Wolff):")
t0 = time.time()
for T in [1.5, 2.0, 2.269, 3.0]:
    Ue, m2e, m4e, ee = exact_UU(N, T)
    R, nsteps, burn = 256, 3000, 200
    Mt, _ = wolff_run(N, T, R, nsteps, seed=12345 + int(T * 1000))
    Mf = Mt[burn:].astype(np.float64) / (N * N)
    m2 = (Mf ** 2).mean(); m4 = (Mf ** 4).mean()
    U = 1 - m4 / (3 * m2 ** 2)
    m2r = (Mf ** 2).mean(axis=0); m4r = (Mf ** 4).mean(axis=0)
    Ur = 1 - m4r / (3 * m2r ** 2)
    eU = Ur.std(ddof=1) / np.sqrt(R)
    print(f"T={T}: exact U={Ue:.5f}, m2={m2e:.5f}, m4={m4e:.5f} | wolff U={U:.5f} +- {eU:.5f}, m2={m2:.5f}, m4={m4:.5f}, z={(U-Ue)/eU:.2f}")
print("time for validation", time.time() - t0)

for R in [256, 1024]:
    t0 = time.time()
    Mt, _ = wolff_run(16, 2.269, R, 50, seed=7)
    dt = time.time() - t0
    print(f"N=16 R={R}: 50 steps took {dt:.3f}s -> {dt/50*1e3:.2f} ms/step, {R*50/dt:.0f} cluster updates/s")
