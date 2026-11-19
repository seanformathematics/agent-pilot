
code = r'''
import numpy as np

def metropolis_batch(N, T, R, ntherm, nsweeps, seed, init="hot", record_every=1):
    """R independent NxN Ising replicas updated with checkerboard Metropolis. Returns per-replica sums of m^2, m^4 and count."""
    rng = np.random.default_rng(seed)
    if init == "hot":
        s = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8)*2 - 1).astype(np.int8)
    else:
        s = np.ones((R, N, N), dtype=np.int8)
    ii, jj = np.indices((N, N))
    masks = [((ii+jj) % 2 == 0), ((ii+jj) % 2 == 1)]
    masks = [np.broadcast_to(m, (R, N, N)) for m in masks]
    table = np.array([min(1.0, np.exp(-2.0*k/T)) for k in (-4, -2, 0, 2, 4)], dtype=np.float32)
    S2 = np.zeros(R); S4 = np.zeros(R); cnt = 0
    for sweep in range(ntherm + nsweeps):
        for mask in masks:
            nb = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
            idx = (s*nb + 4) >> 1
            prob = table[idx]
            u = rng.random(s.shape, dtype=np.float32)
            flip = (u < prob) & mask
            s = np.where(flip, -s, s)
        if sweep >= ntherm and ((sweep - ntherm) % record_every == 0):
            M = s.sum(axis=(1, 2), dtype=np.int32).astype(np.float64)/(N*N)
            m2 = M*M
            S2 += m2; S4 += m2*m2; cnt += 1
    return S2, S4, cnt
'''
open("/work/metro.py","w").write(code)
import sys, time
sys.path.insert(0,"/work")
import numpy as np
from metro import metropolis_batch
t=time.time()
S2,S4,cnt = metropolis_batch(16, 2.3, 256, 100, 200, seed=7)
dt=time.time()-t
print("time for 300 sweeps with 256 replicas:", dt, "-> per sweep ms:", dt/300*1e3)
m2=S2.sum()/(cnt*256); m4=S4.sum()/(cnt*256)
print("U (tiny run) =", 1-m4/(3*m2**2))
