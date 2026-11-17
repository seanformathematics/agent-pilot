import numpy as np, time, json

def metro_run(T, R, n_sweeps, n_burn, seed, N=16):
    rng = np.random.default_rng(seed)
    S = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * np.int8(2) - np.int8(1)).astype(np.int8)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    even = ((ii + jj) % 2 == 0)
    odd = ~even
    M = np.empty((n_sweeps, R), dtype=np.int16)
    for sw in range(n_burn + n_sweeps):
        for sub in (even, odd):
            nb = (np.roll(S, 1, axis=1) + np.roll(S, -1, axis=1) +
                  np.roll(S, 1, axis=2) + np.roll(S, -1, axis=2)).astype(np.int8)
            dE = (2 * S * nb).astype(np.int16)           # in {-8,-4,0,4,8}
            prob = np.where(dE <= 0, 1.0, np.exp(-np.maximum(dE, 0) / T))
            flip = (rng.random((R, N, N)) < prob) & sub
            S = np.where(flip, -S, S).astype(np.int8)
        if sw >= n_burn:
            M[sw - n_burn] = S.sum(axis=(1, 2), dtype=np.int32)
    return M

def jack_replicas(A, B, n):
    R = A.shape[0]
    At, Bt = A.sum(), B.sum()
    tot = n * R
    U_full = 1 - (Bt / tot) / (3 * (At / tot) ** 2)
    Ak = (At - A) / (n * (R - 1)); Bk = (Bt - B) / (n * (R - 1))
    Uk = 1 - Bk / (3 * Ak ** 2)
    se = np.sqrt((R - 1) / R * np.sum((Uk - Uk.mean()) ** 2))
    return U_full, se

out = {}
for T, seed in [(2.8, 4242801), (2.3, 4242302)]:
    t0 = time.time()
    M = metro_run(T, R=256, n_sweeps=8000, n_burn=1000, seed=seed, N=16)
    m = M / 256.0
    A = (m ** 2).sum(axis=0); B = (m ** 4).sum(axis=0)
    U, se = jack_replicas(A, B, M.shape[0])
    out[T] = dict(U=float(U), se=float(se), seed=seed, n=int(M.size))
    print(f"Metropolis T={T}: U={U:.5f} +- {se:.5f} (R=256, 8000 sweeps), time {time.time()-t0:.0f}s")
json.dump(out, open('/work/results/metropolis_check.json', 'w'), indent=1)
