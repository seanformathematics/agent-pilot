
import os, time
os.makedirs('/work', exist_ok=True)
code = r'''
import numpy as np, time

N = 16

def make_masks(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    black = ((i + j) % 2 == 0)
    return black, ~black

def run_chain(T, R, nsweep, seed, nburn=0, init=None, rec_E=False):
    rng = np.random.default_rng(seed)
    black, white = make_masks(N)
    bm = black[:, :, None]
    wm = white[:, :, None]
    if init is None:
        S = rng.choice(np.array([-1, 1], dtype=np.int8), size=(N, N, R))
    else:
        S = init.copy()
    # acceptance table indexed by x = S*nn + 4 (x in 0..8), dE = 2*S*nn
    xs = np.arange(9) - 4
    tab = np.where(xs > 0, np.exp(-2.0 * xs / T), 1.0)
    tab = tab.astype(np.float64)
    m_rec = np.empty((nsweep, R), dtype=np.float64)
    E_rec = np.empty((nsweep, R), dtype=np.float64) if rec_E else None
    for t in range(nsweep + nburn):
        for mask in (bm, wm):
            nn = (np.roll(S, 1, 0) + np.roll(S, -1, 0) +
                  np.roll(S, 1, 1) + np.roll(S, -1, 1))
            x = (S * nn).astype(np.int8) + 4
            p = tab[x]
            u = rng.random(S.shape)
            flip = (u < p) & mask
            S = np.where(flip, -S, S).astype(np.int8)
        if t >= nburn:
            k = t - nburn
            m_rec[k] = S.reshape(N * N, R).mean(axis=0)
            if rec_E:
                nnr = np.roll(S, 1, 0) + np.roll(S, 1, 1)
                E_rec[k] = -(S.astype(np.float64) * nnr).reshape(N*N, R).sum(axis=0)
    return S, m_rec, E_rec

def binder_from_m(m):
    m2 = m**2
    return 1.0 - np.mean(m**4) / (3.0 * np.mean(m2)**2)
'''
open('/work/ising.py', 'w').write(code)
import sys
sys.path.insert(0, '/work')
import importlib, ising
importlib.reload(ising)
import numpy as np
t0 = time.time()
S, m, E = ising.run_chain(2.3, 256, 50, seed=1, rec_E=True)
dt = time.time() - t0
print("time for 50 sweeps x 256 replicas:", dt, "-> per replica-sweep us:", dt / (50*256) * 1e6)
