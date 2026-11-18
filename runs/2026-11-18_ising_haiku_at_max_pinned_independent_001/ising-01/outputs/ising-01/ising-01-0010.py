import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from sw import binder_jackknife

def run_metropolis(L, T, R, n_therm, n_meas, seed):
    """Checkerboard Metropolis, vectorized over R independent replicas."""
    rng = np.random.default_rng(seed)
    S = (rng.integers(0, 2, size=(R, L, L), dtype=np.int8) * 2 - 1).astype(np.int8)
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    black = ((ii + jj) % 2 == 0)[None, :, :]
    white = ~black
    def half(S, mask):
        h = np.roll(S, 1, axis=1) + np.roll(S, -1, axis=1) + np.roll(S, 1, axis=2) + np.roll(S, -1, axis=2)
        dE = 2 * S.astype(np.int16) * h
        u = rng.random(S.shape, dtype=np.float32)
        acc = np.exp(-np.maximum(dE, 0) / T).astype(np.float32)
        flip = mask & ((dE <= 0) | (u < acc))
        return np.where(flip, -S, S).astype(np.int8)
    for _ in range(n_therm):
        S = half(S, black); S = half(S, white)
    M2s = np.zeros(R, dtype=np.int64); M4s = np.zeros(R, dtype=np.int64)
    for _ in range(n_meas):
        S = half(S, black); S = half(S, white)
        M = S.sum(axis=(1, 2), dtype=np.int64)
        M2 = M * M
        M2s += M2; M4s += M2 * M2
    return M2s, M4s

res = json.load(open("/work/results_prod.json"))
for T, seed in [(2.3, 555001), (2.8, 555002)]:
    t0 = time.time()
    M2s, M4s = run_metropolis(16, T, R=512, n_therm=500, n_meas=2000, seed=seed)
    U, err, Ujk = binder_jackknife(M2s, M4s, 2000)
    sw = res[str(T)]
    print(f"Metropolis T={T}: U={U:.5f} +- {err:.5f} | SW production U={sw['U']:.5f} +- {sw['U_err']:.5f} | diff={U-sw['U']:.5f} z={(U-sw['U'])/np.hypot(err, sw['U_err']):.2f} time={time.time()-t0:.1f}s", flush=True)
