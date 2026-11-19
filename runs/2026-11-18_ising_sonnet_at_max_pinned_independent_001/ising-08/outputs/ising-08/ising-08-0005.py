
import sys, time
sys.path.insert(0, '/work')
src = open('/work/ising_lib.py').read()
add = r'''

class HeatBathBatch:
    """R independent NxN lattices, checkerboard heat-bath (N even). Ergodic, no deterministic moves."""
    def __init__(self, N, T, R, rng, init='mixed'):
        assert N % 2 == 0
        self.N = N; self.T = float(T); self.R = R; self.rng = rng
        ii, jj = np.indices((N, N))
        self.masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
        h = np.array([-4, -2, 0, 2, 4], dtype=float)
        self.pup = 1.0 / (1.0 + np.exp(-2.0 * h / self.T))      # P(s=+1 | local field h)
        s = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
        if init == 'cold':
            s[:] = 1
        elif init == 'mixed':
            s[: R // 2] = 1
        self.s = s

    def sweep(self):
        s = self.s
        for mask in self.masks:
            nn = (np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2))
            pu = self.pup[(nn + 4) >> 1]
            new = np.where(self.rng.random(s.shape) < pu, 1, -1).astype(np.int8)
            s = np.where(mask, new, s).astype(np.int8)
        self.s = s

    def mag_energy(self):
        s = self.s; R = self.R; N = self.N
        M = s.reshape(R, -1).sum(axis=1, dtype=np.int64)
        bonds = ((s * np.roll(s, -1, axis=2)).reshape(R, -1).sum(axis=1, dtype=np.int64)
                 + (s * np.roll(s, -1, axis=1)).reshape(R, -1).sum(axis=1, dtype=np.int64))
        return M / float(N * N), -bonds / float(N * N)


def run_hb(N, T, R, n_burn, n_blocks, block, seed, init='mixed'):
    rng = np.random.Generator(np.random.PCG64(seed))
    mb = HeatBathBatch(N, T, R, rng, init)
    for _ in range(n_burn):
        mb.sweep()
    out = np.zeros((4, n_blocks, R))
    for b in range(n_blocks):
        acc = np.zeros((4, R))
        for k in range(block):
            mb.sweep()
            m, E = mb.mag_energy()
            m2 = m * m
            acc[0] += m2; acc[1] += m2 * m2; acc[2] += E; acc[3] += np.abs(m)
        out[:, b, :] = acc
    return out
'''
if 'HeatBathBatch' not in src:
    open('/work/ising_lib.py', 'w').write(src + add)

import numpy as np
import importlib, ising_lib
importlib.reload(ising_lib)
from ising_lib import *

# Equilibration diagnostic for SW at N=16: hot (random) vs cold (all +1) groups
N = 16
R = 256
for T in (1.5, 2.0, 2.3, 2.8):
    Eex = E_exact_torus(T, N)
    rng = np.random.Generator(np.random.PCG64(123))
    sw = SWBatch(N, T, R, rng, 'mixed')
    rec = {}
    nsw = 60
    Eser = np.zeros((nsw, R)); m2ser = np.zeros((nsw, R))
    for k in range(nsw):
        sw.step()
        m, E = sw.mag_energy()
        Eser[k] = E; m2ser[k] = m * m
    print(f"\nT={T}  exact E/N^2={Eex:.5f}")
    print(" sweep |  E cold   E hot   | m2 cold   m2 hot")
    for k in (0, 1, 2, 4, 9, 19, 39, 59):
        print(f" {k+1:4d}  | {Eser[k,:R//2].mean():.5f} {Eser[k,R//2:].mean():.5f} | {m2ser[k,:R//2].mean():.5f} {m2ser[k,R//2:].mean():.5f}")
