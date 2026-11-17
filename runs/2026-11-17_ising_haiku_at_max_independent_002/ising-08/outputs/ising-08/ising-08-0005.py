import os, time, math, random, sys
sys.path.insert(0, '/work')
import numpy as np
import ising_wolff as iw

def batch(T, L, seeds, nflip, blk, discard):
    N = L*L; nb = iw.make_nb(L); padd = 1 - math.exp(-2.0/T)
    S2_all = []; S4_all = []
    for seed in seeds:
        rnd = random.Random(seed).random
        spins = [1 if rnd() < 0.5 else -1 for _ in range(N)]
        mark = [0]*N
        M = sum(spins)
        M, stamp, _, _ = iw.run_flips(spins, mark, 0, M, discard, padd, rnd, nb, N)
        nblk = nflip//blk
        S2b = np.empty(nblk); S4b = np.empty(nblk)
        for b in range(nblk):
            M, stamp, s2, s4 = iw.run_flips(spins, mark, stamp, M, blk, padd, rnd, nb, N)
            S2b[b] = s2; S4b[b] = s4
        assert sum(spins) == M
        S2_all.append(S2b); S4_all.append(S4b)
    return np.array(S2_all), np.array(S4_all)

# ---- 1. remaining production temperatures ----
t_start = time.time()
for T in [1.7, 1.6, 1.5]:
    fn = f'/work/res_{T:.2f}.npz'
    if os.path.exists(fn):
        continue
    t0 = time.time()
    idx = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8].index(T)
    seeds = [1000*(idx+1) + k for k in range(4)]
    S2a, S4a = batch(T, 16, seeds, 300000, 1000, 2000)
    np.savez(fn, S2b=S2a, S4b=S4a, seeds=np.array(seeds), T=T, nflip=300000, blk=1000, discard=2000)
    U, e = iw.jk_U(S2a.ravel(), S4a.ravel(), np.full(S2a.size, 1000.0))
    print(f'T={T:.2f} U={U:.6f} jk_err={e:.7f} time={time.time()-t0:.1f}s', flush=True)
print('production remaining done, elapsed', round(time.time()-t_start, 1), flush=True)

# ---- 2. second independent batch at T=2.4..2.8 (seeds 900000+) ----
for j, T in enumerate([2.8, 2.7, 2.6, 2.5, 2.4]):
    fn = f'/work/res2_{T:.2f}.npz'
    if os.path.exists(fn):
        continue
    t0 = time.time()
    seeds = [900000 + 100*j + k for k in range(4)]
    S2a, S4a = batch(T, 16, seeds, 300000, 1000, 2000)
    np.savez(fn, S2b=S2a, S4b=S4a, seeds=np.array(seeds), T=T, nflip=300000, blk=1000, discard=2000)
    U, e = iw.jk_U(S2a.ravel(), S4a.ravel(), np.full(S2a.size, 1000.0))
    print(f'batch2 T={T:.2f} U={U:.6f} jk_err={e:.7f} time={time.time()-t0:.1f}s', flush=True)
print('batch2 done, elapsed', round(time.time()-t_start, 1), flush=True)

# ---- 3. 4x4 exact check with many more flips at T=3.0 (bias test) ----
def exact_U_4x4(T):
    L = 4; N = 16
    bonds = []
    for i in range(L):
        for j in range(L):
            a = i*L+j
            bonds.append((a, ((i+1) % L)*L+j)); bonds.append((a, i*L+(j+1) % L))
    confs = np.arange(2**N, dtype=np.int64)
    bits = ((confs[:, None] >> np.arange(N)) & 1)
    s = (2*bits - 1).astype(np.int64)
    E = -np.sum(np.stack([s[:, a]*s[:, b] for a, b in bonds], axis=1), axis=1)
    m = s.mean(axis=1)
    w = np.exp(-(E - E.min())/T); Z = w.sum()
    m2 = np.sum(w*m**2)/Z; m4 = np.sum(w*m**4)/Z
    return 1 - m4/(3*m2**2)
for T in [3.0]:
    t0 = time.time()
    Uex = exact_U_4x4(T)
    S2a, S4a = batch(T, 4, [4242 + k for k in range(4)], 1000000, 2000, 2000)
    U, e = iw.jk_U(S2a.ravel(), S4a.ravel(), np.full(S2a.size, 2000.0))
    print(f'4x4 T={T}: exact={Uex:.6f} Wolff(4e6 flips)={U:.6f} jk_err={e:.6f} diff/err={(U-Uex)/e:.2f} time={time.time()-t0:.1f}s', flush=True)
print('total elapsed', round(time.time()-t_start, 1))
print(sorted(os.listdir('/work')))
