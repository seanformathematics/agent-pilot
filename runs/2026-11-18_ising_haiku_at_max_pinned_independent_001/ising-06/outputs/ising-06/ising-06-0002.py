import os, math, random, time
os.makedirs("/work", exist_ok=True)
src = r'''
import math, random
import numpy as np

def build_neighbors(L):
    N = L * L
    nb = [None] * N
    for x in range(L):
        for y in range(L):
            i = x * L + y
            r = x * L + (y + 1) % L
            l = x * L + (y - 1) % L
            u = ((x + 1) % L) * L + y
            d = ((x - 1) % L) * L + y
            nb[i] = (r, l, u, d)
    return nb

def wolff_run(L, T, nsteps, seed, nburn=2000, bin_size=1000, init='cold', keep_series=False):
    """Wolff cluster updates for the 2D Ising model, E = -sum_<ij> s_i s_j, weight exp(-E/T).
    One measurement after every cluster update.
    Returns dict with per-bin sums of m^2, m^4, |m| and optionally the full m series."""
    rng = random.Random(seed)
    rnd = rng.random
    N = L * L
    nb = build_neighbors(L)
    p = 1.0 - math.exp(-2.0 / T)
    if init == 'cold':
        sp = [1] * N
    else:
        sp = [1 if rnd() < 0.5 else -1 for _ in range(N)]
    M = sum(sp)
    for _ in range(nburn):
        s_site = int(rnd() * N)
        s0 = sp[s_site]
        sp[s_site] = -s0
        stack = [s_site]
        size = 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if sp[j] == s0 and rnd() < p:
                    sp[j] = -s0
                    stack.append(j)
                    size += 1
        M -= 2 * s0 * size
    S2 = []; S4 = []; SA = []
    series = [] if keep_series else None
    b2 = b4 = ba = 0.0
    cnt = 0
    invN = 1.0 / N
    for step in range(nsteps):
        s_site = int(rnd() * N)
        s0 = sp[s_site]
        sp[s_site] = -s0
        stack = [s_site]
        size = 1
        while stack:
            i = stack.pop()
            for j in nb[i]:
                if sp[j] == s0 and rnd() < p:
                    sp[j] = -s0
                    stack.append(j)
                    size += 1
        M -= 2 * s0 * size
        m = M * invN
        m2 = m * m
        b2 += m2
        b4 += m2 * m2
        ba += abs(m)
        cnt += 1
        if keep_series:
            series.append(m)
        if cnt == bin_size:
            S2.append(b2); S4.append(b4); SA.append(ba)
            b2 = b4 = ba = 0.0
            cnt = 0
    out = dict(S2=np.array(S2), S4=np.array(S4), SA=np.array(SA), bin_size=bin_size,
               nbins=len(S2))
    if keep_series:
        out['series'] = np.array(series)
    return out

def binder_from_bins(S2, S4, n_per_bin):
    """Binder cumulant and bin-jackknife error from per-bin sums (equal bin sizes)."""
    S2 = np.asarray(S2, float); S4 = np.asarray(S4, float)
    nb = len(S2)
    n = nb * n_per_bin
    m2 = S2.sum() / n; m4 = S4.sum() / n
    U = 1.0 - m4 / (3.0 * m2 * m2)
    # leave-one-bin-out
    Ub = np.empty(nb)
    for b in range(nb):
        a2 = (S2.sum() - S2[b]) / (n - n_per_bin)
        a4 = (S4.sum() - S4[b]) / (n - n_per_bin)
        Ub[b] = 1.0 - a4 / (3.0 * a2 * a2)
    Ubar = Ub.mean()
    err = math.sqrt((nb - 1) / nb * np.sum((Ub - Ubar) ** 2))
    return U, err
'''
with open("/work/ising.py", "w") as f:
    f.write(src)

import sys
sys.path.insert(0, "/work")
import importlib, ising
importlib.reload(ising)
import numpy as np

# ---- exact enumeration for L=4 ----
def exact_U(L, T):
    N = L * L
    nst = 1 << N
    idx = np.arange(nst, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)
    s = 1 - 2 * bits.astype(np.int16)  # bit 1 -> -1
    E = np.zeros(nst)
    nb = ising.build_neighbors(L)
    # each bond counted once: right and up neighbours
    for i in range(N):
        r = nb[i][0]; u = nb[i][2]
        E -= s[:, i] * s[:, r]
        E -= s[:, i] * s[:, u]
    M = s.sum(axis=1)
    w = np.exp(-(E - E.min()) / T)
    Z = w.sum()
    m = M / N
    m2 = (w * m**2).sum() / Z
    m4 = (w * m**4).sum() / Z
    return 1 - m4 / (3 * m2 * m2), m2

for T in [2.0, 2.5, 3.0]:
    Ue, m2e = exact_U(4, T)
    t0 = time.time()
    out = ising.wolff_run(4, T, 400000, seed=1001, nburn=2000, bin_size=1000)
    U, err = ising.binder_from_bins(out['S2'], out['S4'], 1000)
    print(f"L=4 T={T}: exact U={Ue:.5f}  wolff U={U:.5f} +- {err:.5f}  (exact <m2>={m2e:.5f}, time {time.time()-t0:.1f}s)")
