import math, random, time, os
import numpy as np
os.makedirs('/work', exist_ok=True)

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

def wolff_series(N, T, seed, n_updates, n_equil=2000):
    nb = build_nbrs(N)
    L = N*N
    rnd = random.Random(seed).random
    p = 1.0 - math.exp(-2.0/T)
    spins = [1 if rnd() < 0.5 else -1 for _ in range(L)]
    M = sum(spins)
    Ms = np.empty(n_updates, dtype=np.int32)
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

def jackknife_err(M, nb_blocks):
    n = len(M)
    b = n // nb_blocks
    Mt = M[:b*nb_blocks].astype(np.float64)
    x = (Mt**2).reshape(nb_blocks, b)
    y = x**2
    sx = x.sum(axis=1); sy = y.sum(axis=1)
    Sx = sx.sum(); Sy = sy.sum(); nn = b*nb_blocks
    Ui = 1.0 - ((Sy - sy)/(nn-b)) / (3.0*((Sx - sx)/(nn-b))**2)
    Ufull = 1.0 - (Sy/nn)/(3.0*(Sx/nn)**2)
    err = math.sqrt((nb_blocks-1)/nb_blocks * np.sum((Ui - Ui.mean())**2))
    return Ufull, err

def tau_int(x, c=6.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf = acf/acf[0]
    tau = 0.5
    for t in range(1, n):
        tau += acf[t]
        if t >= c*tau:
            return tau, t
    return tau, n

# Pilot: 8 independent chains at T=2.269, 1.25e5 updates each
T = 2.269
t0 = time.time()
chains = []
for k in range(8):
    Ms = wolff_series(16, T, seed=1000+k, n_updates=125000, n_equil=3000)
    chains.append(Ms)
print("time for 8 chains:", round(time.time()-t0,1), "s")
Us = np.array([U_of(M) for M in chains])
print("per-chain U:", np.round(Us,5))
print("mean U over chains:", Us.mean(), " std across chains:", Us.std(ddof=1), " std of mean:", Us.std(ddof=1)/np.sqrt(len(Us)))
for nbk in [10, 20, 50, 100, 200, 500]:
    errs = [jackknife_err(M, nbk)[1] for M in chains]
    print(f"jackknife nblocks={nbk}: per-chain err mean={np.mean(errs):.5f}  (block size {125000//nbk} updates)")
tau_x, _ = tau_int((chains[0]**2).astype(float))
tau_y, _ = tau_int((chains[0]**4).astype(float))
print("tau_int of M^2:", tau_x, " tau_int of M^4:", tau_y)
np.save('/work/pilot_T2.269.npy', np.array(chains))
