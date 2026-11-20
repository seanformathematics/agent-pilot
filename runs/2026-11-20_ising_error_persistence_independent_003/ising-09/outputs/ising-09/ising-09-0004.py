import time, json
import numpy as np

N = 16
R = 1000
EQ = 2000
MEAS = 4000
Ts = [1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
MASK0 = ((ii + jj) % 2 == 0)
MASK1 = ~MASK0


def sweep(s, beta, rng):
    for mask in (MASK0, MASK1):
        nb = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
        dE = 2 * s * nb
        p = np.exp(-beta * np.clip(dE, 0, None))
        u = rng.random(s.shape)
        acc = (dE <= 0) | (u < p)
        s[acc & mask] *= -1
    return s

out = {}
t0 = time.time()
for k, T in enumerate(Ts):
    seed = 1000 + k
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
    for _ in range(EQ):
        sweep(s, beta, rng)
    S2 = np.zeros(R)
    S4 = np.zeros(R)
    series2 = np.zeros(MEAS)
    series4 = np.zeros(MEAS)
    for t in range(MEAS):
        sweep(s, beta, rng)
        M = s.sum(axis=(1, 2)).astype(np.float64) / (N * N)
        m2 = M * M
        m4 = m2 * m2
        S2 += m2
        S4 += m4
        series2[t] = m2.mean()
        series4[t] = m4.mean()
    np.savez(f"/work/raw_T{T}.npz", S2=S2, S4=S4, series2=series2, series4=series4, seed=seed)
    print(f"T={T} done, elapsed {time.time()-t0:.1f}s", flush=True)
