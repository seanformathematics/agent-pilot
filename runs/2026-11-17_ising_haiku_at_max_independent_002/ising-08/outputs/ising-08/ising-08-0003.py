import os, time, math, random
mod_src = r'''
import math, random
import numpy as np

def make_nb(L):
    nb = []
    for i in range(L):
        for j in range(L):
            nb.append((((i+1) % L)*L + j, ((i-1) % L)*L + j, i*L + (j+1) % L, i*L + (j-1) % L))
    return nb

def run_flips(spins, mark, stamp, M, nfl, padd, rnd, nb, N):
    """Perform nfl Wolff cluster flips; after each flip record m^2 and m^4 (m=M/N).
    Returns (M, stamp, sum m^2, sum m^4)."""
    S2 = 0.0; S4 = 0.0
    Nf = float(N)
    for _ in range(nfl):
        seed = int(rnd()*N)
        s0 = spins[seed]
        stamp += 1
        mark[seed] = stamp
        cl = [seed]
        for c in cl:
            for q in nb[c]:
                if mark[q] != stamp and spins[q] == s0 and rnd() < padd:
                    mark[q] = stamp
                    cl.append(q)
        for c in cl:
            spins[c] = -s0
        M -= 2*s0*len(cl)
        m2 = (M/Nf)**2
        S2 += m2
        S4 += m2*m2
    return M, stamp, S2, S4

def jk_U(S2b, S4b, cnt):
    """Binder U = 1 - <m^4>/(3<m^2>^2) from block sums, with delete-one-block jackknife error."""
    S2 = S2b.sum(); S4 = S4b.sum(); n = cnt.sum()
    U = 1 - n*S4/(3*S2**2)
    Um = 1 - (n-cnt)*(S4-S4b)/(3*(S2-S2b)**2)
    B = len(S2b)
    err = math.sqrt((B-1)/B*np.sum((Um-Um.mean())**2))
    return float(U), float(err)
'''
os.makedirs('/work', exist_ok=True)
with open('/work/ising_wolff.py', 'w') as f:
    f.write(mod_src)

import sys
sys.path.insert(0, '/work')
import importlib
import ising_wolff as iw
importlib.reload(iw)
import numpy as np

L = 16; N = L*L
nb = iw.make_nb(L)
Ts = [2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
K = 4; NFL = 300000; DISC = 2000; BLK = 1000
SEED_BASE_FOR = {T: 1000*(i+1) for i, T in enumerate(Ts)}

budget = 400.0
t_start = time.time()
done = []
for T in Ts:
    fn = f'/work/res_{T:.2f}.npz'
    if os.path.exists(fn):
        continue
    if time.time() - t_start > budget - 90:
        break
    t0 = time.time()
    padd = 1 - math.exp(-2.0/T)
    S2_all = []; S4_all = []; seeds = []
    for k in range(K):
        seed = SEED_BASE_FOR[T] + k
        seeds.append(seed)
        rnd = random.Random(seed).random
        spins = [1 if rnd() < 0.5 else -1 for _ in range(N)]
        mark = [0]*N
        M = sum(spins)
        M, stamp, _, _ = iw.run_flips(spins, mark, 0, M, DISC, padd, rnd, nb, N)
        nblk = NFL//BLK
        S2b = np.empty(nblk); S4b = np.empty(nblk)
        for b in range(nblk):
            M, stamp, s2, s4 = iw.run_flips(spins, mark, stamp, M, BLK, padd, rnd, nb, N)
            S2b[b] = s2; S4b[b] = s4
        S2_all.append(S2b); S4_all.append(S4b)
    S2_all = np.array(S2_all); S4_all = np.array(S4_all)
    np.savez(fn, S2b=S2_all, S4b=S4_all, seeds=np.array(seeds), T=T, nflip=NFL, blk=BLK, discard=DISC)
    cnt = np.full(S2_all.size, float(BLK))
    U, eU = iw.jk_U(S2_all.ravel(), S4_all.ravel(), cnt)
    Uk = [iw.jk_U(S2_all[k], S4_all[k], np.full(S2_all.shape[1], float(BLK)))[0] for k in range(K)]
    print(f"T={T:.2f} U={U:.6f} jk_err={eU:.6f} chain U={np.round(Uk,5)} chain-scatter/sqrt(K)={np.std(Uk, ddof=1)/np.sqrt(K):.6f} time={time.time()-t0:.1f}s", flush=True)
    done.append(T)
print('done this call:', done, 'elapsed', round(time.time()-t_start,1))
print('existing:', sorted([f for f in os.listdir('/work') if f.startswith('res_')]))
