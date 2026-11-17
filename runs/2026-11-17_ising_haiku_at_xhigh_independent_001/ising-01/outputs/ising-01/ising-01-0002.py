import numpy as np, time, os
os.makedirs("/work", exist_ok=True)
code = r'''
import numpy as np, time, sys

def run(T, L=16, R=1000, nburn=1000, nmeas=2000, every=5, seed=1, init_random=True):
    rng = np.random.default_rng(seed)
    if init_random:
        s = rng.integers(0, 2, size=(R, L, L), dtype=np.int8) * 2 - 1
    else:
        s = np.ones((R, L, L), dtype=np.int8)
    idx = np.arange(L)
    par = ((idx[:, None] + idx[None, :]) % 2).astype(bool)
    masks = [par, ~par]
    # acceptance table indexed by sn = s*nb in {-4,-2,0,2,4} -> (sn+4)//2
    pacc = np.zeros(5, dtype=np.float32)
    for k in range(5):
        sn = 2 * k - 4
        dE = 2 * sn
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
    S2 = np.zeros(R); S4 = np.zeros(R); SE = np.zeros(R); SM = np.zeros(R); cnt = 0
    for it in range(nmeas):
        s = sweep(s)
        if (it + 1) % every == 0:
            m = s.reshape(R, -1).mean(axis=1, dtype=np.float64)
            S2 += m * m; S4 += m ** 4; SM += np.abs(m)
            cnt += 1
    return dict(S2=S2 / cnt, S4=S4 / cnt, SM=SM / cnt, cnt=cnt)
'''
open("/work/ising_mc.py", "w").write(code)
import sys
sys.path.insert(0, "/work")
from ising_mc import run
t0 = time.time()
out = run(2.269, R=1000, nburn=200, nmeas=400, every=5, seed=11)
dt = time.time() - t0
print("time for 600 sweeps, R=1000:", dt, "s; per sweep", dt / 600)
S2 = out["S2"].mean(); S4 = out["S4"].mean()
print("U pilot", 1 - S4 / (3 * S2 * S2), "<|m|>", out["SM"].mean())