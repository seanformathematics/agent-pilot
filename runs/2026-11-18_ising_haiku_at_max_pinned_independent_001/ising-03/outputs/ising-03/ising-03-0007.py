import sys, os, time, json, math
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import wolff_series, U_of

t_start = time.time()
# ---- Part A: Metropolis cross-check at low T (independent algorithm) ----
def metropolis_run(N, T, R, nsweep, nequil, seed):
    rng = np.random.default_rng(seed)
    S = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, N, N))
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
    Ms = []
    for sweep in range(nequil + nsweep):
        for par in range(2):
            nn = (np.roll(S, 1, axis=1) + np.roll(S, -1, axis=1) +
                  np.roll(S, 1, axis=2) + np.roll(S, -1, axis=2)).astype(np.int16)
            dE = 2*S.astype(np.int16)*nn
            prob = np.exp(-np.clip(dE, 0, None)/T)
            u = rng.random(size=(R, N, N))
            flip = masks[par][None] & ((dE <= 0) | (u < prob))
            S = np.where(flip, -S, S).astype(np.int8)
        if sweep >= nequil:
            Ms.append(S.reshape(R, -1).sum(axis=1).astype(np.int64))
    return np.array(Ms).flatten()

cross = {}
for T in [1.5, 2.0, 2.2]:
    Mm = metropolis_run(16, T, R=64, nsweep=2000, nequil=1000, seed=4242)
    Uw = np.concatenate([np.load(f'/work/data/T{T:.1f}_c{k}.npy').astype(np.int64) for k in range(8)])
    cross[T] = (U_of(Mm), U_of(Uw))
    print(f"cross-check T={T}: Metropolis U={U_of(Mm):.5f} (R=64 x 2000 sweeps, correlated)  Wolff(batch1) U={U_of(Uw):.5f}")

# ---- Part B: second independent batch of Wolff chains ----
Ts = [round(1.5 + 0.1*i, 1) for i in range(14)]
K = 8; NUPD = 125000; NEQ = 5000; BUDGET = 420.0
seedlog = {}
done_now = 0
for iT, T in enumerate(Ts):
    for k in range(K):
        fname = f'/work/data/B2_T{T:.1f}_c{k}.npy'
        if os.path.exists(fname):
            continue
        if time.time() - t_start > BUDGET:
            break
        seed = 100000 + 1000*(iT+1) + k + 17
        M = wolff_series(16, T, seed, NUPD, NEQ)
        np.save(fname, M)
        seedlog[f'B2_T{T:.1f}_c{k}'] = seed
        done_now += 1
json.dump(seedlog, open(f'/work/data/seeds_B2_{int(time.time())}.json', 'w'), indent=1)
nfiles = len([f for f in os.listdir('/work/data') if f.startswith('B2_')])
print("batch2 chains done this call:", done_now, " total batch2 files:", nfiles, "of", len(Ts)*K)
print("elapsed", round(time.time()-t_start,1))
