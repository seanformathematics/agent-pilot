import sys, time, math
sys.path.insert(0, '/work')
import numpy as np
import ising

def metropolis_M(N, T, R, nsweeps, burn, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
    ii, jj = np.indices((N, N))
    maskA = ((ii + jj) % 2 == 0)
    maskB = ~maskA
    out = np.empty((nsweeps, R), dtype=np.int32)
    for sw in range(burn + nsweeps):
        for mask in (maskA, maskB):
            nsum = (np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1) +
                    np.roll(s, 1, axis=2) + np.roll(s, -1, axis=2))
            dE = (2 * s * nsum).astype(np.float64)          # energy change if flipped
            acc = np.exp(-np.maximum(dE, 0.0) / T)           # =1 when dE<=0
            flip = (rng.random((R, N, N)) < acc) & mask
            s = np.where(flip, -s, s).astype(np.int8)
        if sw >= burn:
            out[sw - burn] = s.reshape(R, -1).sum(axis=1)
    return out.reshape(-1).astype(np.int64)

for T, seed in [(2.3, 4242), (2.8, 4343)]:
    t0 = time.time()
    M = metropolis_M(16, T, 256, 20000, 2000, seed)
    # block jackknife over sweeps: blocks of 200 sweeps x 256 replicas (concatenated sweep-major)
    U, Ujk, e = ising.binder_jk(M, 500)   # 500 blocks of 10240 samples
    print(f"Metropolis T={T}: U={U:.5f} +- {e:.5f} (n={M.size}), time {time.time()-t0:.1f}s", flush=True)
    wolff = np.load(f'/work/M_T{T:.2f}.npy').astype(np.int64)
    Uw = ising.binder_from_M(wolff, 256)
    print(f"   Wolff (production, 2e6 samples) U={Uw:.5f}", flush=True)
