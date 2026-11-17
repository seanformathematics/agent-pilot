import numpy as np, time, os
os.makedirs('/work', exist_ok=True)
src = r'''
import numpy as np

def sw_run(T, R, n_steps, n_burn, seed, N=16, S0=None):
    """Swendsen-Wang for R independent replicas of an NxN periodic Ising lattice.
    Returns M (n_steps, R) int16 total magnetization sums (sum of spins), and final spins."""
    rng = np.random.default_rng(seed)
    p = 1.0 - np.exp(-2.0 / T)
    NN = N * N
    if S0 is None:
        S = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * np.int8(2) - np.int8(1)).astype(np.int8)
    else:
        S = S0.astype(np.int8).copy()
    base = np.arange(NN, dtype=np.int16).reshape(1, N, N)
    BIG = np.int16(NN + 100)
    M = np.empty((n_steps, R), dtype=np.int16)
    total = n_burn + n_steps
    for step in range(total):
        eqh = S == np.roll(S, -1, axis=2)
        eqv = S == np.roll(S, -1, axis=1)
        bh = eqh & (rng.random((R, N, N)) < p)
        bv = eqv & (rng.random((R, N, N)) < p)
        mR = bh
        mL = np.roll(bh, 1, axis=2)
        mD = bv
        mU = np.roll(bv, 1, axis=1)
        lab = np.broadcast_to(base, (R, N, N)).copy()
        while True:
            nR = np.where(mR, np.roll(lab, -1, axis=2), BIG)
            nL = np.where(mL, np.roll(lab, 1, axis=2), BIG)
            nD = np.where(mD, np.roll(lab, -1, axis=1), BIG)
            nU = np.where(mU, np.roll(lab, 1, axis=1), BIG)
            new = np.minimum(lab, np.minimum(np.minimum(nR, nL), np.minimum(nD, nU)))
            flat = new.reshape(R, NN)
            new = np.take_along_axis(flat, flat.astype(np.intp), axis=1).reshape(R, N, N)
            if np.array_equal(new, lab):
                break
            lab = new
        flags = rng.random((R, NN)) < 0.5
        flip = np.take_along_axis(flags, lab.reshape(R, NN).astype(np.intp), axis=1).reshape(R, N, N)
        S = np.where(flip, -S, S).astype(np.int8)
        if step >= n_burn:
            M[step - n_burn] = S.sum(axis=(1, 2), dtype=np.int32)
    return M, S
'''
open('/work/sw.py', 'w').write(src)
import sys
sys.path.insert(0, '/work')
import importlib, sw
importlib.reload(sw)

def exact_U(N, T):
    NN = N * N
    nconf = 1 << NN
    configs = np.arange(nconf, dtype=np.int64)
    bits = (configs[:, None] >> np.arange(NN)) & 1
    S = (2 * bits - 1).astype(np.int32)
    idx = np.arange(NN).reshape(N, N)
    E = np.zeros(nconf)
    for i in range(N):
        for j in range(N):
            a = idx[i, j]
            E -= S[:, a] * S[:, idx[i, (j + 1) % N]]
            E -= S[:, a] * S[:, idx[(i + 1) % N, j]]
    M = S.sum(axis=1)
    m = M / NN
    logw = -E / T
    logw -= logw.max()
    w = np.exp(logw)
    Z = w.sum()
    m2 = (w * m ** 2).sum() / Z
    m4 = (w * m ** 4).sum() / Z
    return 1 - m4 / (3 * m2 ** 2), m2, m4 / Z * Z

def U_from_M(M):
    NN = 16 * 16
    m = M.astype(np.float64) / NN
    m2 = (m ** 2).mean()
    m4 = (m ** 4).mean()
    return 1 - m4 / (3 * m2 ** 2)

# 1) validation on 4x4 against exact enumeration
for T in [1.8, 2.27, 3.0]:
    t0 = time.time()
    Ue, m2e, _ = exact_U(4, T)
    M, _ = sw.sw_run(T, R=400, n_steps=1500, n_burn=100, seed=12345, N=4)
    Us = U_from_M(M)
    # replica-level spread
    Ur = []
    for r in range(M.shape[1]):
        Ur.append(U_from_M(M[:, r:r+1]))
    Ur = np.array(Ur)
    print(f"N=4 T={T}: exact U={Ue:.5f}, SW U={Us:.5f}, std over replicas/sqrt(R)={Ur.std(ddof=1)/np.sqrt(len(Ur)):.5f}, time {time.time()-t0:.1f}s")

# 2) timing on 16x16
t0 = time.time()
M, S = sw.sw_run(2.2692, R=512, n_steps=50, n_burn=10, seed=7, N=16)
dt = time.time() - t0
print("N=16 timing: 60 steps x 512 replicas in", round(dt, 2), "s ->", round(dt/60*1000, 2), "ms per step")
