import sys, math, time
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import wolff_series, U_of

def metropolis_run(N, T, R, nsweep, nequil, seed):
    rng = np.random.default_rng(seed)
    S = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
    acc_probs = {}
    Ms = []
    Es = []
    for sweep in range(nequil + nsweep):
        for par in range(2):
            nn = (np.roll(S, 1, axis=1) + np.roll(S, -1, axis=1) +
                  np.roll(S, 1, axis=2) + np.roll(S, -1, axis=2)).astype(np.int16)
            dE = 2*S.astype(np.int16)*nn  # energy change for flipping
            prob = np.exp(-np.clip(dE, 0, None)/T)
            u = rng.random(size=(R, N, N))
            flip = masks[par][None] & ((dE <= 0) | (u < prob))
            S = np.where(flip, -S, S).astype(np.int8)
        if sweep >= nequil:
            M = S.reshape(R, -1).sum(axis=1).astype(np.float64)
            Ms.append(M)
    Ms = np.array(Ms)  # (nsweep, R)
    return Ms

# Compare at T=2.8 and T=2.4 and T=5.0 for N=16
for T in [5.0, 2.8, 2.4]:
    t0 = time.time()
    Mmet = metropolis_run(16, T, R=64, nsweep=1500, nequil=500, seed=77)
    Mflat = Mmet.flatten()
    Mw = wolff_series(16, T, seed=5, n_updates=60000, n_equil=3000).astype(np.float64)
    m2_met = np.mean((Mflat/256)**2); m2_w = np.mean((Mw/256)**2)
    print(f"T={T}: Metropolis <m2>={m2_met:.5f} U={U_of(Mflat.astype(np.int64)):.4f} <|m|>={np.mean(np.abs(Mflat))/256:.4f} |  "
          f"Wolff <m2>={m2_w:.5f} U={U_of(Mw.astype(np.int64)):.4f} <|m|>={np.mean(np.abs(Mw))/256:.4f}   (t={time.time()-t0:.1f}s)")

# Wolff mean cluster size at several T (identity: <|C|> relates to <M^2>/L)
def wolff_csize(N, T, seed, n):
    import random, math
    from wolff_lib import build_nbrs
    nb = build_nbrs(N); L = N*N
    rnd = random.Random(seed).random
    p = 1 - math.exp(-2/T)
    spins = [1 if rnd() < 0.5 else -1 for _ in range(L)]
    stack = []; pop = stack.pop; push = stack.append
    tot = 0
    for k in range(n):
        i0 = int(rnd()*L); s = spins[i0]; spins[i0] = -s; push(i0); c = 1
        while stack:
            i = pop()
            for j in nb[i]:
                if spins[j] == s and rnd() < p:
                    spins[j] = -s; push(j); c += 1
        tot += c
    return tot/n
for T in [5.0, 2.8, 2.4, 2.269]:
    print("T", T, "mean Wolff cluster size", round(wolff_csize(16, T, 9, 20000), 3))
