
import sys, time, os
sys.path.insert(0, "/work")
import numpy as np
import sw

N = 16
def load(tag, T):
    d = np.load(f"/work/{tag}_T{T:.1f}.npz")
    return {k: d[k] for k in d.files}

def metropolis_run(N, T, R, n_burn, n_meas, seed, start="cold"):
    rng = np.random.default_rng(seed)
    if start == "cold":
        s = np.ones((R, N, N), dtype=np.int8)
    else:
        s = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
    ii, jj = np.indices((N, N))
    par = (ii + jj) % 2
    masks = [(par == 0)[None], (par == 1)[None]]
    ks = np.arange(-4, 5)
    acc_table = np.minimum(1.0, np.exp(-2.0 * ks / T))
    n = N * N
    S2 = np.zeros(R); S4 = np.zeros(R)
    for t in range(n_burn + n_meas):
        for mk in masks:
            nn = np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1) + np.roll(s, 1, axis=2) + np.roll(s, -1, axis=2)
            k = s * nn
            flip = (rng.random(s.shape) < acc_table[k + 4]) & mk
            np.negative(s, out=s, where=flip)
        if t >= n_burn:
            m = s.sum(axis=(1, 2), dtype=np.int32) / n
            m2 = m * m
            S2 += m2; S4 += m2 * m2
    return S2, S4

# timing
t0 = time.time()
metropolis_run(N, 2.3, 500, 0, 40, 1)
print("Metropolis: %.1f ms per sweep for R=500" % ((time.time() - t0) / 40 * 1e3), flush=True)

R, n_burn, n_meas = 500, 1000, 3000
zs = []
for T in [1.5, 1.9, 2.1, 2.3, 2.5, 2.8]:
    seed = 3000 + int(round(T * 10))
    t0 = time.time()
    S2, S4 = metropolis_run(N, T, R, n_burn, n_meas, seed, "cold")
    Um, em, _ = sw.jackknife_U(S2, S4, n_meas)
    d1, d2 = load("run1", T), load("run2", T)
    S2s = np.concatenate([d1["S2i"].sum(1), d2["S2i"].sum(1)]); S4s = np.concatenate([d1["S4i"].sum(1), d2["S4i"].sum(1)])
    Us, es, _ = sw.jackknife_U(S2s, S4s, n_meas)
    z = (Um - Us) / np.hypot(em, es)
    zs.append(z)
    print(f"T={T:.1f}: Metropolis(seed {seed}) U={Um:.5f}+-{em:.5f}   SW pooled U={Us:.5f}+-{es:.5f}   z={z:+.2f}   [{time.time()-t0:.0f}s]", flush=True)
zs = np.array(zs)
print("z:", np.round(zs, 2), " chi2 =", round((zs ** 2).sum(), 2), "for", len(zs), "dof")
