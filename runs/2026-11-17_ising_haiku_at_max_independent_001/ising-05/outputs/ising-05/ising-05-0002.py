import os, sys, time
os.makedirs('/work', exist_ok=True)
src = r'''
import math, random

def make_neighbors(N):
    nbr = []
    for x in range(N):
        for y in range(N):
            nbr.append((((x + 1) % N) * N + y,
                        ((x - 1) % N) * N + y,
                        x * N + (y + 1) % N,
                        x * N + (y - 1) % N))
    return nbr

def wolff_chain(N, T, seed, n_burn, n_blocks, block, cold=False):
    """Wolff cluster algorithm for 2D Ising (J=1, beta=1/T), periodic lattice.
    Returns S2, S4 (lists, one entry per block of `block` measured cluster updates):
      S2[b] = sum of M^2, S4[b] = sum of M^4, M = total spin sum after each update.
    Also returns final M and the number of cluster-flip spins total (diagnostic)."""
    V = N * N
    nbr = make_neighbors(N)
    rnd = random.Random(seed).random
    padd = 1.0 - math.exp(-2.0 / T)
    if cold:
        sp = [1] * V
    else:
        sp = [1 if rnd() < 0.5 else -1 for _ in range(V)]
    M = sum(sp)
    for _ in range(n_burn):
        i0 = int(rnd() * V)
        s0 = sp[i0]
        sp[i0] = -s0
        stack = [i0]
        size = 1
        while stack:
            i = stack.pop()
            for j in nbr[i]:
                if sp[j] == s0 and rnd() < padd:
                    sp[j] = -s0
                    stack.append(j)
                    size += 1
        M -= 2 * s0 * size
    S2 = [0] * n_blocks
    S4 = [0] * n_blocks
    tot_size = 0
    for b in range(n_blocks):
        s2 = 0
        s4 = 0
        for _ in range(block):
            i0 = int(rnd() * V)
            s0 = sp[i0]
            sp[i0] = -s0
            stack = [i0]
            size = 1
            while stack:
                i = stack.pop()
                for j in nbr[i]:
                    if sp[j] == s0 and rnd() < padd:
                        sp[j] = -s0
                        stack.append(j)
                        size += 1
            M -= 2 * s0 * size
            tot_size += size
            M2 = M * M
            s2 += M2
            s4 += M2 * M2
        S2[b] = s2
        S4[b] = s4
    return S2, S4, M, tot_size / (n_blocks * block)

def U_from_sums(sum2, sum4, n):
    # U = 1 - <M^4>/(3 <M^2>^2), with <.> = sum/n ; scale-free in M
    return 1.0 - (sum4 / n) / (3.0 * (sum2 / n) ** 2)

def jackknife_U(S2, S4, G):
    """Delete-one-group jackknife over G groups of consecutive blocks.
    Returns (U_full, err, U_jk_mean)."""
    nb = len(S2)
    gsz = nb // G
    used = gsz * G
    S2g = [sum(S2[k * gsz:(k + 1) * gsz]) for k in range(G)]
    S4g = [sum(S4[k * gsz:(k + 1) * gsz]) for k in range(G)]
    tot2 = sum(S2g[:G]); tot4 = sum(S4g[:G])
    n = used * 1.0  # per-block count normalised below via block size (counts cancel out in ratio)
    # counts per group are identical, so use group counts = gsz (in units of blocks)
    U_full = U_from_sums(tot2, tot4, G * gsz)
    Uk = []
    for k in range(G):
        Uk.append(U_from_sums(tot2 - S2g[k], tot4 - S4g[k], (G - 1) * gsz))
    mean = sum(Uk) / G
    var = (G - 1) / G * sum((u - mean) ** 2 for u in Uk)
    return U_full, math.sqrt(var), mean
'''
with open('/work/wolff.py', 'w') as f:
    f.write(src)
sys.path.insert(0, '/work')
import numpy as np
from wolff import wolff_chain, jackknife_U

# ---------- exact enumeration on 4x4 ----------
N = 4; V = N * N
idx = np.arange(2 ** V)
bits = (idx[:, None] >> np.arange(V)[None, :]) & 1
spins = (bits * 2 - 1).reshape(-1, N, N).astype(np.int64)
Mtot = spins.sum(axis=(1, 2))
E = -((spins * np.roll(spins, -1, axis=1)).sum(axis=(1, 2)) + (spins * np.roll(spins, -1, axis=2)).sum(axis=(1, 2)))
Emin = E.min()
exact = {}
for T in [1.5, 2.2692, 2.8]:
    w = np.exp(-(E - Emin) / T)
    Z = w.sum()
    m2 = (w * Mtot ** 2).sum() / Z
    m4 = (w * Mtot ** 4).sum() / Z
    U = 1 - m4 / (3 * m2 ** 2)
    exact[T] = U
    print(f"exact N=4 T={T}: <M^2>={m2:.6f} <M^4>={m4:.4f} U={U:.6f}")

for T in [1.5, 2.2692, 2.8]:
    t0 = time.time()
    S2, S4, Mf, avgsz = wolff_chain(4, T, seed=12345, n_burn=2000, n_blocks=4000, block=100)
    dt = time.time() - t0
    Uf, err, _ = jackknife_U(S2, S4, 40)
    print(f"Wolff N=4 T={T}: U={Uf:.6f} +- {err:.6f}  (exact {exact[T]:.6f}, diff/err={(Uf-exact[T])/err:.2f}) time {dt:.1f}s avg cluster {avgsz:.2f}")

# ---------- timing on N=16 ----------
for T in [1.5, 2.2692, 2.8]:
    t0 = time.time()
    S2, S4, Mf, avgsz = wolff_chain(16, T, seed=777, n_burn=1000, n_blocks=200, block=100)
    dt = time.time() - t0
    print(f"N=16 T={T}: 20000 clusters in {dt:.2f}s -> {dt/20000*1e6:.1f} us/cluster, avg cluster size {avgsz:.1f}, U(quick)={jackknife_U(S2,S4,10)[0]:.4f}")
