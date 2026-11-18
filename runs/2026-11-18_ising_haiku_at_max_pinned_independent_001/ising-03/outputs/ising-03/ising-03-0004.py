import os, time, json
os.makedirs('/work/data', exist_ok=True)
lib = r'''
import math, random
import numpy as np

def build_nbrs(N):
    nb = []
    for x in range(N):
        for y in range(N):
            r = x*N + (y+1) % N
            l = x*N + (y-1) % N
            d = ((x+1) % N)*N + y
            u = ((x-1) % N)*N + y
            nb.append((r, l, d, u))
    return nb

def wolff_series(N, T, seed, n_updates, n_equil=5000):
    """Wolff single-cluster updates; returns total magnetisation M = sum s_i after each measured update."""
    nb = build_nbrs(N)
    L = N*N
    rnd = random.Random(seed).random
    p = 1.0 - math.exp(-2.0/T)
    spins = [1 if rnd() < 0.5 else -1 for _ in range(L)]   # random start
    M = sum(spins)
    Ms = np.empty(n_updates, dtype=np.int16)
    stack = []
    pop = stack.pop; push = stack.append
    for k in range(n_equil + n_updates):
        i0 = int(rnd()*L)
        s = spins[i0]
        spins[i0] = -s
        push(i0)
        c = 1
        while stack:
            i = pop()
            for j in nb[i]:
                if spins[j] == s and rnd() < p:
                    spins[j] = -s
                    push(j)
                    c += 1
        M -= 2*s*c
        if k >= n_equil:
            Ms[k-n_equil] = M
    return Ms

def U_of(M):
    x = M.astype(np.float64)**2
    y = x*x
    return 1.0 - y.mean()/(3.0*x.mean()**2)

def jackknife(Mblocks, nb_blocks=None):
    """Mblocks: list of 1-D arrays (blocks). Delete-one jackknife over blocks."""
    sx = np.array([(M.astype(np.float64)**2).sum() for M in Mblocks])
    sy = np.array([((M.astype(np.float64)**2)**2).sum() for M in Mblocks])
    nn = np.array([len(M) for M in Mblocks], dtype=np.float64)
    Sx, Sy, Nt = sx.sum(), sy.sum(), nn.sum()
    Ui = 1.0 - ((Sy - sy)/(Nt-nn)) / (3.0*((Sx - sx)/(Nt-nn))**2)
    Ufull = 1.0 - (Sy/Nt)/(3.0*(Sx/Nt)**2)
    nb = len(Mblocks)
    err = math.sqrt((nb-1)/nb * np.sum((Ui - Ui.mean())**2))
    return Ufull, err
'''
open('/work/wolff_lib.py', 'w').write(lib)
import sys
sys.path.insert(0, '/work')
from wolff_lib import wolff_series
import numpy as np

Ts = [round(1.5 + 0.1*i, 1) for i in range(14)]
K = 8
NUPD = 125000
NEQ = 5000
BUDGET = 400.0
t_start = time.time()
seedlog = {}
done_now = []
for iT, T in enumerate(Ts):
    for k in range(K):
        fname = f'/work/data/T{T:.1f}_c{k}.npy'
        if os.path.exists(fname):
            continue
        if time.time() - t_start > BUDGET:
            break
        seed = 1000*(iT+1) + k + 17
        t0 = time.time()
        M = wolff_series(16, T, seed, NUPD, NEQ)
        np.save(fname, M)
        seedlog[f'T{T:.1f}_c{k}'] = seed
        done_now.append((T, k, round(time.time()-t0, 1)))
json.dump(seedlog, open('/work/data/seeds_call1.json', 'w'), indent=1)
print("chains finished this call:", len(done_now))
print("timing (T,chain,sec):", done_now[:3], '...', done_now[-3:])
print("elapsed", round(time.time()-t_start,1))
existing = sorted(os.listdir('/work/data'))
print("files now:", len([f for f in existing if f.endswith('.npy')]), "of", len(Ts)*K)
