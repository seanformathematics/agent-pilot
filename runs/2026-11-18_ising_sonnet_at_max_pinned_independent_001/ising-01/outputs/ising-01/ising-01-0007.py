
import sys, time, os
sys.path.insert(0, "/work")
import numpy as np
import sw
from scipy import stats

N = 16
TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
SW_NT = 1000   # measured sweeps per replica in SW runs 1 and 2

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

R_met, nb_met, nm_met = 1000, 1000, 4000
zs = []
t_all = time.time()
for T in TS:
    fn = f"/work/met_T{T:.1f}.npz"
    seed = 3000 + int(round(T * 10))
    if not os.path.exists(fn):
        S2, S4 = metropolis_run(N, T, R_met, nb_met, nm_met, seed, "cold")
        np.savez(fn, T=T, seed=seed, S2=S2, S4=S4, n_t=nm_met)
    dm = load("met", T)
    Um, em, _ = sw.jackknife_U(dm["S2"], dm["S4"], nm_met)
    d1, d2 = load("run1", T), load("run2", T)
    S2s = np.concatenate([d1["S2i"].sum(1), d2["S2i"].sum(1)]); S4s = np.concatenate([d1["S4i"].sum(1), d2["S4i"].sum(1)])
    Us, es, _ = sw.jackknife_U(S2s, S4s, SW_NT)
    z = (Um - Us) / np.hypot(em, es)
    zs.append(z)
    print(f"T={T:.1f}: Metropolis U={Um:.5f}+-{em:.5f} | SW pooled(2x10^6 meas) U={Us:.5f}+-{es:.5f} | z={z:+.2f}  [{time.time()-t_all:.0f}s]", flush=True)
zs = np.array(zs)
print("z:", np.round(zs, 2))
print("mean z %.2f, rms z %.2f, chi2 = %.1f for %d dof, p = %.3f" % (zs.mean(), np.sqrt((zs ** 2).mean()), (zs ** 2).sum(), len(zs), stats.chi2.sf((zs ** 2).sum(), len(zs))))
