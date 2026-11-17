import os, time
os.makedirs('/work', exist_ok=True)
src = r'''
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

TC = 2.0/np.log(1.0+np.sqrt(2.0))

def sw_step(s, T, rng):
    """One Swendsen-Wang sweep on R independent replicas. s: int8 array (R,N,N) of +-1."""
    R, N, _ = s.shape
    M = R*N*N
    p = -np.expm1(-2.0/T)  # 1 - exp(-2/T): bond activation prob for aligned neighbours (J=1)
    right = np.roll(s, -1, axis=2)
    down = np.roll(s, -1, axis=1)
    ur = rng.random((R, N, N))
    ud = rng.random((R, N, N))
    bh = (s == right) & (ur < p)
    bv = (s == down) & (ud < p)
    idx = np.arange(M, dtype=np.int64).reshape(R, N, N)
    ridx = np.roll(idx, -1, axis=2)
    didx = np.roll(idx, -1, axis=1)
    rows = np.concatenate((idx[bh], idx[bv]))
    cols = np.concatenate((ridx[bh], didx[bv]))
    g = csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(M, M))
    ncomp, lab = connected_components(g, directed=False)
    flip = rng.random(ncomp) < 0.5
    mask = flip[lab].reshape(R, N, N)
    s = np.where(mask, -s, s).astype(np.int8)
    return s

def init_spins(R, N, rng, init='random'):
    if init == 'random':
        return (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
    return np.ones((R, N, N), dtype=np.int8)

def run(N, T, R, nsweep, nburn, seed, init='random', series_reps=0):
    """Returns per-replica sums S1=sum|m|, S2=sum m^2, S4=sum m^4 over nsweep measured sweeps,
    and optionally the m^2 time series of the first series_reps replicas."""
    rng = np.random.default_rng(seed)
    s = init_spins(R, N, rng, init)
    for _ in range(nburn):
        s = sw_step(s, T, rng)
    S1 = np.zeros(R); S2 = np.zeros(R); S4 = np.zeros(R)
    ser = np.zeros((series_reps, nsweep)) if series_reps > 0 else None
    for t in range(nsweep):
        s = sw_step(s, T, rng)
        m = s.reshape(R, -1).sum(axis=1, dtype=np.int64) / (N*N)
        m2 = m*m
        S1 += np.abs(m); S2 += m2; S4 += m2*m2
        if series_reps > 0:
            ser[:, t] = m2[:series_reps]
    return S1, S2, S4, ser

def exact_moments(N, T):
    L = N*N
    configs = ((np.arange(2**L, dtype=np.int64)[:, None] >> np.arange(L, dtype=np.int64)[None, :]) & 1)*2 - 1
    spins = configs.reshape(-1, N, N)
    E = -(spins*np.roll(spins, -1, axis=2) + spins*np.roll(spins, -1, axis=1)).sum(axis=(1, 2))
    w = np.exp(-(E - E.min())/T)
    m = spins.reshape(-1, L).sum(axis=1)/L
    Z = w.sum()
    m2 = (w*m**2).sum()/Z
    m4 = (w*m**4).sum()/Z
    return m2, m4, 1 - m4/(3*m2**2)

def binder_from_sums(S2, S4, n):
    """Pooled Binder cumulant from per-replica sums (equal counts n per replica)."""
    a2 = S2.sum()/(n*len(S2)); a4 = S4.sum()/(n*len(S4))
    return 1 - a4/(3*a2*a2)

def jackknife_U(S2, S4, n):
    R = len(S2)
    tot2 = S2.sum(); tot4 = S4.sum()
    Uk = np.empty(R)
    for k in range(R):
        a2 = (tot2 - S2[k])/(n*(R-1)); a4 = (tot4 - S4[k])/(n*(R-1))
        Uk[k] = 1 - a4/(3*a2*a2)
    Ubar = Uk.mean()
    err = np.sqrt((R-1)/R * np.sum((Uk-Ubar)**2))
    U = binder_from_sums(S2, S4, n)
    return U, err
'''
open('/work/swlib.py', 'w').write(src)

import sys, time
sys.path.insert(0, '/work')
import numpy as np
from swlib import *

# 1) Validate SW sampler on N=4 against exact enumeration
t0 = time.time()
for T in [2.0, 2.269, 2.6]:
    m2e, m4e, Ue = exact_moments(4, T)
    S1, S2, S4, _ = run(4, T, R=2000, nsweep=300, nburn=100, seed=12345 + int(T*1000))
    n = 300
    U, err = jackknife_U(S2, S4, n)
    m2s = S2.sum()/(n*len(S2)); m4s = S4.sum()/(n*len(S4))
    print(f"N=4 T={T}: exact m2={m2e:.6f} m4={m4e:.6f} U={Ue:.6f} | SW m2={m2s:.6f} m4={m4s:.6f} U={U:.6f} +- {err:.6f}")
print("validation time", time.time()-t0)

# 2) Timing on N=16
for R in [64, 256]:
    t0 = time.time()
    rng = np.random.default_rng(1)
    s = init_spins(R, 16, rng)
    for _ in range(50):
        s = sw_step(s, 2.269, rng)
    dt = (time.time()-t0)/50
    print(f"N=16 R={R}: {dt*1e3:.2f} ms per sweep -> {dt/R*1e6:.1f} us per replica-sweep")
