import os
os.makedirs("/work", exist_ok=True)
lib = r'''
import numpy as np

def wolff_step(s, p, rng):
    """One Wolff cluster update, applied independently to each of R replicas.
    s: int8 array (R, N, N) of +-1 spins. Returns a new int8 array.
    Bonds are pre-sampled (active with prob p if the two spins are equal),
    and the cluster is the connected component of a random seed site."""
    R, N, _ = s.shape
    sr = np.roll(s, -1, axis=2)   # right neighbour s[i, j+1]
    sd = np.roll(s, -1, axis=1)   # down neighbour s[i+1, j]
    act_r = (s == sr) & (rng.random((R, N, N), dtype=np.float32) < p)
    act_d = (s == sd) & (rng.random((R, N, N), dtype=np.float32) < p)
    idx = rng.integers(0, N * N, size=R)
    mark = np.zeros((R, N * N), dtype=bool)
    mark[np.arange(R), idx] = True
    mark = mark.reshape(R, N, N)
    cnt = 1
    while True:
        g = (np.roll(mark, -1, axis=2) & act_r) | np.roll(mark & act_r, 1, axis=2)
        g |= (np.roll(mark, -1, axis=1) & act_d) | np.roll(mark & act_d, 1, axis=1)
        new = mark | g
        c = int(np.count_nonzero(new))
        if c == cnt:
            break
        cnt = c
        mark = new
    return np.where(mark, -s, s).astype(np.int8)

def run_chains(N, T, R, n_eq, n_meas, seed, ts_reps=0):
    """R independent Wolff chains at temperature T. Returns per-replica sums of M^2 and M^4
    over the n_meas measurement steps (after n_eq equilibration steps), the equilibration
    trace of the replica-averaged M^2, and optionally a time series of M for ts_reps replicas."""
    rng = np.random.default_rng(seed)
    p = 1.0 - np.exp(-2.0 / T)
    s = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
    S2 = np.zeros(R, dtype=np.int64)
    S4 = np.zeros(R, dtype=np.int64)
    eq_trace = np.zeros(n_eq, dtype=np.float64)
    ts = np.zeros((n_meas, ts_reps), dtype=np.int16) if ts_reps else None
    for t in range(n_eq + n_meas):
        s = wolff_step(s, p, rng)
        M = s.sum(axis=(1, 2), dtype=np.int64)
        M2 = M * M
        if t < n_eq:
            eq_trace[t] = M2.mean()
        else:
            S2 += M2
            S4 += M2 * M2
            if ts_reps:
                ts[t - n_eq] = M[:ts_reps]
    return S2, S4, eq_trace, ts

def U_from_sums(S2sum, S4sum, n):
    m2 = S2sum / n
    m4 = S4sum / n
    return 1.0 - m4 / (3.0 * m2 * m2)

def jackknife_U(S2, S4, n_per_rep, G):
    """Replica-group jackknife for U = 1 - <M^4>/(3<M^2>^2). Replicas are independent runs."""
    R = S2.shape[0]
    g = R // G
    G2 = S2.reshape(G, g).sum(axis=1).astype(np.float64)
    G4 = S4.reshape(G, g).sum(axis=1).astype(np.float64)
    Ntot = float(n_per_rep) * R
    U_full = U_from_sums(G2.sum(), G4.sum(), Ntot)
    U_del = np.empty(G)
    for k in range(G):
        U_del[k] = U_from_sums(G2.sum() - G2[k], G4.sum() - G4[k], Ntot - n_per_rep * g)
    Ubar = U_del.mean()
    var = (G - 1) / G * np.sum((U_del - Ubar) ** 2)
    bias = (G - 1) * (Ubar - U_full)
    return U_full, float(np.sqrt(var)), U_full - bias
'''
with open("/work/ising.py", "w") as f:
    f.write(lib)

import sys, time
sys.path.insert(0, "/work")
import numpy as np
from ising import run_chains, jackknife_U

def make_bonds(N):
    sites = np.arange(N * N).reshape(N, N)
    a = np.concatenate([sites.ravel(), sites.ravel()])
    b = np.concatenate([np.roll(sites, -1, axis=1).ravel(), np.roll(sites, -1, axis=0).ravel()])
    return a, b

def exact_U(N, T):
    nsite = N * N
    cfg = np.arange(2 ** nsite, dtype=np.int64)
    bits = (cfg[:, None] >> np.arange(nsite)[None, :]) & 1
    s = (2 * bits - 1).astype(np.float64)
    a, b = make_bonds(N)
    E = -np.sum(s[:, a] * s[:, b], axis=1)
    w = np.exp(-(E - E.min()) / T)
    M = s.sum(axis=1)
    Z = w.sum()
    m2 = (w * M ** 2).sum() / Z
    m4 = (w * M ** 4).sum() / Z
    return 1 - m4 / (3 * m2 ** 2)

# Validation on 4x4 against exact enumeration
for T in [1.8, 2.3, 2.8]:
    Ue = exact_U(4, T)
    t0 = time.time()
    S2, S4, eqt, _ = run_chains(4, T, R=1024, n_eq=200, n_meas=2000, seed=12345)
    Us, Uerr, Ubc = jackknife_U(S2, S4, 2000, 64)
    print(f"N=4 T={T}: exact U={Ue:.6f}  wolff U={Us:.6f} +- {Uerr:.6f}  (bias-corr {Ubc:.6f})  pull={(Us-Ue)/Uerr:.2f}  [{time.time()-t0:.1f}s]")

# Timing on N=16
for R in [1024, 4096]:
    t0 = time.time()
    S2, S4, eqt, _ = run_chains(16, 2.27, R=R, n_eq=0, n_meas=20, seed=1)
    dt = time.time() - t0
    print(f"N=16 R={R}: {dt/20*1000:.1f} ms per Wolff step for all replicas -> {R*20/dt:.0f} updates/s")
