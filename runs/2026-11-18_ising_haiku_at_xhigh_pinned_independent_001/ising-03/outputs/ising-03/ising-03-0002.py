mod = """
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def sw_setup(R, N):
    idx = np.arange(R*N*N, dtype=np.int64).reshape(R, N, N)
    idr = np.roll(idx, -1, axis=2)
    idd = np.roll(idx, -1, axis=1)
    return idx, idr, idd

def sw_sweep(s, p, rng, geo):
    R, N, _ = s.shape
    idx, idr, idd = geo
    sr = np.roll(s, -1, axis=2)
    sd = np.roll(s, -1, axis=1)
    br = (s == sr) & (rng.random(s.shape) < p)
    bd = (s == sd) & (rng.random(s.shape) < p)
    u = np.concatenate([idx[br], idx[bd]])
    v = np.concatenate([idr[br], idd[bd]])
    M = R*N*N
    g = csr_matrix((np.ones(u.shape[0], dtype=np.int8), (u, v)), shape=(M, M))
    ncomp, labels = connected_components(g, directed=False)
    flip = rng.random(ncomp) < 0.5
    f = flip[labels].reshape(R, N, N)
    return np.where(f, -s, s).astype(np.int8)
"""
open('/work/sw.py', 'w').write(mod)

import sys
sys.path.insert(0, '/work')
import numpy as np, time
from sw import sw_setup, sw_sweep

def exact_4x4(T, N=4):
    confs = np.array([[1 if (k >> b) & 1 else -1 for b in range(N*N)] for k in range(2**(N*N))], dtype=np.int64)
    s = confs.reshape(-1, N, N)
    E = -((s*np.roll(s, -1, axis=1)).sum(axis=(1, 2)) + (s*np.roll(s, -1, axis=2)).sum(axis=(1, 2)))
    w = np.exp(-(E - E.min())/T)
    Z = w.sum()
    m = s.sum(axis=(1, 2))/(N*N)
    m2 = (w*m**2).sum()/Z
    m4 = (w*m**4).sum()/Z
    return 1 - m4/(3*m2**2)

for T in [1.5, 2.27, 3.0]:
    print("exact 4x4 U at T", T, round(exact_4x4(T), 5))

R = 2000; N = 4
rng = np.random.default_rng(12345)
geo = sw_setup(R, N)
for T in [1.5, 2.27, 3.0]:
    p = 1 - np.exp(-2/T)
    s = np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)
    m2s = []; m4s = []
    for it in range(1000):
        s = sw_sweep(s, p, rng, geo)
        if it >= 100:
            m = s.reshape(R, -1).mean(axis=1)
            m2s.append((m**2).mean()); m4s.append((m**4).mean())
    m2 = np.mean(m2s); m4 = np.mean(m4s)
    print("SW 4x4 T", T, round(1 - m4/(3*m2**2), 5))

R = 512; N = 16
geo = sw_setup(R, N)
rng = np.random.default_rng(7)
s = np.where(rng.random((R, N, N)) < 0.5, 1, -1).astype(np.int8)
p = 1 - np.exp(-2/2.27)
t0 = time.time()
for it in range(100):
    s = sw_sweep(s, p, rng, geo)
t1 = time.time()
print("time per sweep R=512 N=16:", (t1 - t0)/100)
