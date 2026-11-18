import sys, time, math, random
sys.path.insert(0, "/work")
import numpy as np, ising

def metropolis_run(L, T, nsweeps, seed, nburn_sweeps=500, bin_size=100):
    rng = random.Random(seed)
    rnd = rng.random
    N = L * L
    nb = ising.build_neighbors(L)
    sp = [1 if rnd() < 0.5 else -1 for _ in range(N)]
    M = sum(sp)
    # acceptance table for dE = 2 s_i * sum_nb s_j in {-8,-4,0,4,8}
    acc = {dE: (1.0 if dE <= 0 else math.exp(-dE / T)) for dE in (4, 8)}
    series = []
    for sw in range(nburn_sweeps + nsweeps):
        for _ in range(N):
            i = int(rnd() * N)
            r, l, u, d = nb[i]
            h = sp[r] + sp[l] + sp[u] + sp[d]
            dE = 2 * sp[i] * h
            if dE <= 0 or rnd() < acc.get(dE, 0.0):
                M -= 2 * sp[i]
                sp[i] = -sp[i]
        if sw >= nburn_sweeps:
            series.append(M / N)
    return np.array(series)

for T in [2.8, 2.269]:
    t0 = time.time()
    m = metropolis_run(16, T, 60000, seed=77)
    dt = time.time() - t0
    m2 = m * m
    # bin-jackknife with bins of 200 sweeps
    nbins = len(m) // 200
    S2 = m2[:nbins * 200].reshape(nbins, 200).sum(1)
    S4 = (m2 * m2)[:nbins * 200].reshape(nbins, 200).sum(1)
    U, err = ising.binder_from_bins(S2, S4, 200)
    tau2, _ = (0, 0)
    print(f"Metropolis L=16 T={T}: U={U:.4f} +- {err:.4f}  <|m|>={np.mean(np.abs(m)):.4f}  <m2>={m2.mean():.5f}  time {dt:.1f}s")
