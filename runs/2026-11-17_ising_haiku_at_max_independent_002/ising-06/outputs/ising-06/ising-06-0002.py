import os
os.makedirs('/work', exist_ok=True)
src = r'''
import random, math
import numpy as np

def build_nbr(N):
    nbr = []
    for i in range(N):
        for j in range(N):
            nbr.append((((i+1) % N)*N + j, ((i-1) % N)*N + j, i*N + (j+1) % N, i*N + (j-1) % N))
    return nbr

def wolff_chain(N, T, nupd, burn, seed):
    """Wolff single-cluster updates. Returns int64 array of total magnetization sum M (= N^2 m)
    recorded after every cluster update (after burn-in). Spins start random."""
    NN = N*N
    nbr = build_nbr(N)
    rnd = random.Random(seed).random
    p = 1.0 - math.exp(-2.0/T)
    spins = [1 if rnd() < 0.5 else -1 for _ in range(NN)]
    M = sum(spins)
    out = [0]*nupd
    for k in range(burn + nupd):
        x0 = int(rnd()*NN)
        s0 = spins[x0]
        spins[x0] = -s0
        stack = [x0]
        pop = stack.pop
        push = stack.append
        c = 1
        while stack:
            x = pop()
            for y in nbr[x]:
                if spins[y] == s0 and rnd() < p:
                    spins[y] = -s0
                    push(y)
                    c += 1
        M -= 2*s0*c
        if k >= burn:
            out[k-burn] = M
    return np.array(out, dtype=np.int64), spins

def binder_from_M(M, NN):
    M = np.asarray(M, dtype=np.int64)
    n = M.size
    S2 = int(np.sum(M*M))
    S4 = int(np.sum((M*M)*(M*M)))
    # U = 1 - <m^4>/(3<m^2>^2); m = M/NN, normalisation cancels
    return 1.0 - (n*float(S4))/(3.0*float(S2)**2)

def binder_jk(M, nblocks=100):
    """Block-jackknife estimate of U and its standard error."""
    M = np.asarray(M, dtype=np.int64)
    n = M.size
    L = n // nblocks
    M = M[:L*nblocks]
    Mb = M.reshape(nblocks, L)
    q = Mb*Mb
    S2b = q.sum(axis=1).astype(np.float64)
    S4b = (q*q).sum(axis=1).astype(np.float64)
    S2 = S2b.sum(); S4 = S4b.sum()
    nn = float(L*nblocks)
    U_full = 1.0 - nn*S4/(3.0*S2*S2)
    S2_l = S2 - S2b; S4_l = S4 - S4b; n_l = nn - L
    U_l = 1.0 - n_l*S4_l/(3.0*S2_l*S2_l)
    B = nblocks
    Ubar = U_l.mean()
    var = (B-1)/B * np.sum((U_l-Ubar)**2)
    U_jk = B*U_full - (B-1)*Ubar  # bias-corrected jackknife
    return U_full, U_jk, math.sqrt(var)
'''
with open('/work/ising.py', 'w') as f:
    f.write(src)

import sys, time, math, itertools
sys.path.insert(0, '/work')
import numpy as np
import importlib
import ising
importlib.reload(ising)

# ---- exact enumeration for 4x4 ----
def exact_moments(N, T):
    NN = N*N
    nbr = ising.build_nbr(N)
    bonds = set()
    for x in range(NN):
        for y in nbr[x]:
            bonds.add((min(x,y), max(x,y)))
    bonds = np.array(sorted(bonds))
    configs = np.arange(2**NN, dtype=np.int64)
    bits = ((configs[:, None] >> np.arange(NN)) & 1)
    s = 2*bits - 1  # (2^NN, NN)
    E = -np.sum(s[:, bonds[:,0]] * s[:, bonds[:,1]], axis=1)
    M = np.sum(s, axis=1)
    w = np.exp(-(E - E.min())/T)
    Z = w.sum()
    m2 = np.sum(w*M**2)/Z
    m4 = np.sum(w*M**4)/Z
    U = 1 - (m4/NN**4)/(3*(m2/NN**2)**2)
    return U, m2/NN**2, m4/NN**4

for T in [1.5, 2.3, 3.0]:
    Uex, m2, m4 = exact_moments(4, T)
    t0 = time.time()
    M, _ = ising.wolff_chain(4, T, 400000, 1000, seed=12345)
    dt = time.time()-t0
    Uf, Ujk, Ue = ising.binder_jk(M, 100)
    print(f"4x4 T={T}: exact U={Uex:.6f}  wolff U={Uf:.6f} +- {Ue:.6f} (jk-corr {Ujk:.6f}), time {dt:.1f}s")

# ---- benchmark on 16x16 ----
for T in [1.5, 2.269, 2.8]:
    t0 = time.time()
    M, _ = ising.wolff_chain(16, T, 20000, 1000, seed=7)
    dt = time.time()-t0
    print(f"16x16 T={T}: {dt/21000*1e6:.1f} us per update; <|m|>={np.mean(np.abs(M))/256:.4f}")
