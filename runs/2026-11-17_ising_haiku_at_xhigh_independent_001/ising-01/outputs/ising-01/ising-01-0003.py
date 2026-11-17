import numpy as np, time, sys, json
sys.path.insert(0, "/work")
code = r'''
import numpy as np

def run(T, seed, L=16, R=1000, nburn=2000, nmeas=10000, every=10, blk=100):
    rng = np.random.default_rng(seed)
    s = (rng.integers(0, 2, size=(R, L, L), dtype=np.int8) * 2 - 1).astype(np.int8)
    idx = np.arange(L)
    par = ((idx[:, None] + idx[None, :]) % 2).astype(bool)
    masks = [par, ~par]
    pacc = np.zeros(5, dtype=np.float32)
    for k in range(5):
        dE = 2 * (2 * k - 4)
        pacc[k] = 1.0 if dE <= 0 else np.exp(-dE / T)
    def sweep(s):
        for m_ in masks:
            nb = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2))
            sn = s * nb
            p = pacc[((sn + 4) >> 1)]
            u = rng.random(s.shape, dtype=np.float32)
            flip = (u < p) & m_
            s = np.where(flip, -s, s).astype(np.int8)
        return s
    for _ in range(nburn):
        s = sweep(s)
    nS = nmeas // every
    nblk = nS // blk
    M2 = np.zeros((R, nblk)); M4 = np.zeros((R, nblk)); M1 = np.zeros((R, nblk)); EN = np.zeros((R, nblk))
    k = 0
    for it in range(nmeas):
        s = sweep(s)
        if (it + 1) % every == 0:
            b = k // blk
            if b < nblk:
                m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
                M2[:, b] += m * m; M4[:, b] += m ** 4; M1[:, b] += np.abs(m)
                # energy per spin: bonds right and down
                e = -(s * np.roll(s, -1, 1) + s * np.roll(s, -1, 2)).reshape(R, -1).sum(axis=1, dtype=np.float64) / (L * L)
                EN[:, b] += e
            k += 1
    return dict(M2=M2 / blk, M4=M4 / blk, M1=M1 / blk, EN=EN / blk)
'''
open("/work/ising_mc2.py", "w").write(code)
from ising_mc2 import run
temps = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
t0 = time.time()
for i, T in enumerate(temps):
    seed = 1000 + i
    out = run(T, seed)
    np.savez(f"/work/res_T{T:.2f}_s{seed}.npz", T=T, seed=seed, **out)
    M2 = out["M2"].mean(); M4 = out["M4"].mean()
    print(f"T={T} seed={seed} U={1-M4/(3*M2*M2):.5f}  E/N={out['EN'].mean():.4f} <|m|>={out['M1'].mean():.4f} time={time.time()-t0:.1f}", flush=True)