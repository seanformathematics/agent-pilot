import sys, time, json, math
sys.path.insert(0, '/work')
import ising_lib as L
import numpy as np

def exact_N(N, T):
    nS = N * N
    idx = np.arange(2 ** nS)
    bits = (idx[:, None] >> np.arange(nS)[None, :]) & 1
    spins = (bits * 2 - 1).reshape(-1, N, N)
    E = -(spins * np.roll(spins, -1, axis=1)).sum(axis=(1, 2)) - (spins * np.roll(spins, -1, axis=2)).sum(axis=(1, 2))
    m = spins.sum(axis=(1, 2)) / nS
    w = np.exp(-(E - E.min()) / T)
    Z = w.sum()
    m2 = (w * m ** 2).sum() / Z; m4 = (w * m ** 4).sum() / Z
    return 1 - m4 / (3 * m2 ** 2)

def metro_run(N, R, T, nburn, nmeas, seed):
    rng = np.random.default_rng(seed)
    beta = 1.0 / T
    p4 = np.float32(math.exp(-4 * beta)); p8 = np.float32(math.exp(-8 * beta))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    par = ((ii + jj) % 2)[:, :, None]
    masks = [par == 0, par == 1]
    s = np.where(rng.random((N, N, R)) < 0.5, 1, -1).astype(np.int8)
    m2 = np.empty((nmeas, R)); m4 = np.empty((nmeas, R))
    for t in range(nburn + nmeas):
        for mk in masks:
            h = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
            dE = 2 * s * h
            acc = np.where(dE >= 8, p8, np.where(dE >= 4, p4, np.float32(1.0)))
            u = rng.random((N, N, R), dtype=np.float32)
            flip = (u < acc) & mk
            s = np.where(flip, -s, s).astype(np.int8)
        if t >= nburn:
            m = s.sum(axis=(0, 1), dtype=np.int32) / (N * N)
            m2[t - nburn] = m * m; m4[t - nburn] = m ** 4
    return m2.T.copy(), m4.T.copy()

t_all = time.time()
# (1) exact vs Wolff on N=4 at T=2.8
Ue = exact_N(4, 2.8)
m2, m4, hist, el, itm = L.run_chain(4, 2000, 2.8, 300, 2000, 777)
Uw, ew = L.jack_U_rep(m2.sum(1), m4.sum(1), m2.shape[1])
print(f"N=4 T=2.8: exact U={Ue:.5f}, Wolff U={Uw:.5f} +- {ew:.5f}, pull={(Uw-Ue)/ew:+.2f}", flush=True)

# (2) Metropolis on N=16
for T, nb, nm in [(2.8, 500, 1500), (2.0, 1000, 1500)]:
    t0 = time.time()
    m2, m4 = metro_run(16, 2000, T, nb, nm, seed=31337 + int(T * 10))
    U, e = L.jack_U_rep(m2.sum(1), m4.sum(1), nm)
    tau2 = L.tau_int(m2)
    print(f"Metropolis N=16 T={T}: U={U:.5f} +- {e:.5f}, tau_int(m2)={tau2:.2f}, <m2>={m2.mean():.4f}, time={time.time()-t0:.0f}s", flush=True)
print("total", time.time() - t_all, flush=True)
