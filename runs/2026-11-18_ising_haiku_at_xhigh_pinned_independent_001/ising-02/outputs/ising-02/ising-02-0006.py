import time, random, math, json, os, itertools
import numpy as np

def make_nbr(N):
    nb = []
    for x in range(N):
        for y in range(N):
            nb.append([((x + 1) % N) * N + y, ((x - 1) % N) * N + y, x * N + (y + 1) % N, x * N + (y - 1) % N])
    return nb

def run_wolff(T, nsteps, seed, N, nbr, burn=0):
    NN = N*N
    rng = random.Random(seed)
    padd = 1.0 - math.exp(-2.0 / T)
    spins = [1 if rng.random() < 0.5 else -1 for _ in range(NN)]
    M = sum(spins)
    Ms = np.empty(nsteps, dtype=np.int32)
    rnd = rng.random
    rr = rng.randrange
    for t in range(nsteps):
        i = rr(NN)
        s0 = spins[i]
        spins[i] = -s0
        stack = [i]
        c = 1
        while stack:
            j = stack.pop()
            for k in nbr[j]:
                if spins[k] == s0 and rnd() < padd:
                    spins[k] = -s0
                    stack.append(k)
                    c += 1
        M -= 2 * s0 * c
        Ms[t] = M
    return Ms[burn:] / NN

# --- validation on 4x4 lattice against exact enumeration ---
N4 = 4; nb4 = make_nbr(N4); NN4 = N4*N4
bonds = set()
for i in range(NN4):
    for k in nb4[i]:
        bonds.add(tuple(sorted((i, k))))
bonds = list(bonds)
confs = np.array(list(itertools.product([1,-1], repeat=NN4)), dtype=np.int8)
E = -np.sum(confs[:, [b[0] for b in bonds]] * confs[:, [b[1] for b in bonds]], axis=1)
for Tv in [1.5, 2.3, 3.5]:
    w = np.exp(-(E - E.min())/Tv); w /= w.sum()
    mm = confs.sum(1)/NN4
    ex_m2 = np.sum(w*mm**2); ex_m4 = np.sum(w*mm**4)
    ex_U = 1 - ex_m4/(3*ex_m2**2)
    Ms = run_wolff(Tv, 400000, 777, N4, nb4, burn=2000)
    wm2 = np.mean(Ms**2); wm4 = np.mean(Ms**4)
    print("N=4 check T=%.2f exact m2=%.6f wolff m2=%.6f exact m4=%.6f wolff m4=%.6f exactU=%.6f wolffU=%.6f" % (Tv, ex_m2, wm2, ex_m4, wm4, ex_U, 1-wm4/(3*wm2**2)))

# --- production: second half of temperatures ---
N = 16; NN = 256; nbr = make_nbr(N)
TS = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
NCHAIN = 4; NSTEP = 100000; BURN = 5000; BS = 1000

def autocorr_tau(x, wmax=500):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    for w in range(1, wmax):
        tau += acf[w]
        if w > 6*tau:
            break
    return tau

for ti, T in enumerate(TS):
    t0 = time.time()
    seeds = [20240601 + 1000*c + 37*(ti+7) for c in range(NCHAIN)]
    chains = [run_wolff(T, NSTEP, sd, N, nbr, burn=BURN) for sd in seeds]
    S2 = []; S4 = []; chain_U = []; taus = []
    for ch in chains:
        m2 = ch**2; m4 = ch**4
        nb = len(ch)//BS
        S2.append(m2[:nb*BS].reshape(nb, BS).sum(1))
        S4.append(m4[:nb*BS].reshape(nb, BS).sum(1))
        chain_U.append(1 - m4.mean()/(3*m2.mean()**2))
        taus.append(autocorr_tau(m2))
    S2 = np.concatenate(S2); S4 = np.concatenate(S4)
    nblk = len(S2); tot2 = S2.sum(); tot4 = S4.sum(); tot_n = nblk*BS
    U = 1 - (tot4/tot_n)/(3*(tot2/tot_n)**2)
    Uj = np.empty(nblk)
    for k in range(nblk):
        s2 = tot2 - S2[k]; s4 = tot4 - S4[k]; n = tot_n - BS
        Uj[k] = 1 - (s4/n)/(3*(s2/n)**2)
    err_jk = math.sqrt((nblk-1)/nblk * np.sum((Uj-Uj.mean())**2))
    err_chain = np.std(chain_U, ddof=1)/math.sqrt(NCHAIN)
    res = dict(T=T, U=float(U), err_jk=float(err_jk), err_chain=float(err_chain),
               chain_U=[float(x) for x in chain_U], tau_m2=[float(x) for x in taus],
               nsamples=int(tot_n), seeds=seeds, time=time.time()-t0)
    with open("/work/res_T%.2f.json" % T, "w") as f:
        json.dump(res, f)
    print(json.dumps(res))
