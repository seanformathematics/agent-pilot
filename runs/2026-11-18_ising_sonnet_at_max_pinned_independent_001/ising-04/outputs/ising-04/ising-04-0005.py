
import sys, time
import numpy as np
sys.path.insert(0, "/work")

prod = r'''
import numpy as np
from sw import BatchSW

def energy_per_site(s):
    N = s.shape[1]
    return -((s * np.roll(s, -1, axis=2)).sum(axis=(1, 2), dtype=np.int64)
             + (s * np.roll(s, -1, axis=1)).sum(axis=(1, 2), dtype=np.int64)) / (N * N)

def run_T(T, seed, N=16, R=1000, neq=200, nblocks=10, blen=200, start="hot"):
    sim = BatchSW(N, R, seed)
    s = sim.init_hot() if start == "hot" else sim.init_cold()
    p = 1.0 - np.exp(-2.0 / T)
    for _ in range(neq):
        s, _, _ = sim.step(s, p)
    out = {k: np.zeros((nblocks, R)) for k in ("p2", "p4", "i2", "i4", "e")}
    for b in range(nblocks):
        for t in range(blen):
            m2, m4 = sim.measure(s)
            e = energy_per_site(s)
            s, i2, i4 = sim.step(s, p)
            out["p2"][b] += m2; out["p4"][b] += m4
            out["i2"][b] += i2; out["i4"][b] += i4
            out["e"][b] += e
    return out, blen
'''
open("/work/prod.py", "w").write(prod)
from prod import run_T

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
t00 = time.time()
for T in Ts:
    seed = 100000 + int(round(T * 10))
    t0 = time.time()
    out, blen = run_T(T, seed)
    np.savez(f"/work/prod_T{T:.1f}.npz", blen=blen, seed=seed, **out)
    print(f"T={T} seed={seed} done in {time.time()-t0:.1f}s (total {time.time()-t00:.0f}s)", flush=True)
