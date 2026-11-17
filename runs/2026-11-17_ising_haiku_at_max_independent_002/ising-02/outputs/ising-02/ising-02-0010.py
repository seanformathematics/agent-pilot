import sys, time, json
import numpy as np

N = 16
def metro_U(T, seed, R=512, burn=500, K=3000):
    rng = np.random.default_rng(seed)
    s = np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii + jj) % 2 == 0)[None], ((ii + jj) % 2 == 1)[None]]
    # acceptance table indexed by dE+8, dE = 2*s*nn in {-8..8}
    dEs = np.arange(-8, 9)
    table = np.minimum(1.0, np.exp(-dEs / T))
    def half(s, mask):
        nn = np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1) + np.roll(s, 1, axis=2) + np.roll(s, -1, axis=2)
        dE = (2 * s * nn).astype(np.int16) + 8
        acc = (rng.random(s.shape) < table[dE]) & mask
        return np.where(acc, -s, s).astype(np.int8)
    S2 = np.zeros(R, dtype=np.int64); S4 = np.zeros(R, dtype=np.int64)
    for t in range(burn + K):
        s = half(s, masks[0]); s = half(s, masks[1])
        if t >= burn:
            M = s.reshape(R, -1).sum(axis=1, dtype=np.int64)
            M2 = M * M
            S2 += M2; S4 += M2 * M2
    return S2, S4, K

def U_jk(S2, S4, K):
    R = S2.size
    A2 = float(S2.sum()); A4 = float(S4.sum()); n = K * R
    U = 1 - (A4 / n) / (3 * (A2 / n) ** 2)
    B2 = A2 - S2.astype(float); B4 = A4 - S4.astype(float); nb = n - K
    Uj = 1 - (B4 / nb) / (3 * (B2 / nb) ** 2)
    return U, np.sqrt((R - 1) / R * np.sum((Uj - Uj.mean()) ** 2))

t0 = time.time()
res = {}
for i, T in enumerate([2.0, 2.3, 2.8]):
    S2, S4, K = metro_U(T, seed=9100 + i)
    U, e = U_jk(S2, S4, K)
    res[str(T)] = dict(U=U, err=e, seed=9100 + i)
    print("Metropolis T=%.2f: U=%.5f +- %.5f  (%.1f s elapsed)" % (T, U, e, time.time() - t0), flush=True)
json.dump(res, open('/work/res/metropolis_check.json', 'w'))
