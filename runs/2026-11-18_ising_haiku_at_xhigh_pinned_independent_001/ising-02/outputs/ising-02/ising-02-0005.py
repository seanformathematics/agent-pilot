import time, random, math, json, os
import numpy as np

os.makedirs("/work", exist_ok=True)
N = 16
NN = N * N
nbr = []
for x in range(N):
    for y in range(N):
        nbr.append([((x + 1) % N) * N + y, ((x - 1) % N) * N + y, x * N + (y + 1) % N, x * N + (y - 1) % N])

def run_wolff(T, nsteps, seed):
    rng = random.Random(seed)
    padd = 1.0 - math.exp(-2.0 / T)
    spins = [1 if rng.random() < 0.5 else -1 for _ in range(NN)]   # random start
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
    return Ms

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

TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
NCHAIN = 4
NSTEP = 100000
BURN = 5000
BS = 1000
for ti, T in enumerate(TS):
    t0 = time.time()
    seeds = [20240601 + 1000*c + 37*ti for c in range(NCHAIN)]
    chains = []
    for sd in seeds:
        Ms = run_wolff(T, NSTEP, sd)
        chains.append(Ms[BURN:] / NN)
    # pooled block-jackknife over blocks (blocks never cross chain boundaries)
    S2 = []; S4 = []; cnt = []
    chain_U = []
    taus = []
    for ch in chains:
        m2 = ch**2; m4 = ch**4
        nb = len(ch)//BS
        S2.append(m2[:nb*BS].reshape(nb, BS).sum(1))
        S4.append(m4[:nb*BS].reshape(nb, BS).sum(1))
        chain_U.append(1 - m4.mean()/(3*m2.mean()**2))
        taus.append(autocorr_tau(m2))
    S2 = np.concatenate(S2); S4 = np.concatenate(S4)
    nblk = len(S2)
    n_per = BS
    tot2 = S2.sum(); tot4 = S4.sum(); tot_n = nblk*n_per
    U = 1 - (tot4/tot_n)/(3*(tot2/tot_n)**2)
    Uj = np.empty(nblk)
    for k in range(nblk):
        s2 = tot2 - S2[k]; s4 = tot4 - S4[k]; n = tot_n - n_per
        Uj[k] = 1 - (s4/n)/(3*(s2/n)**2)
    err_jk = math.sqrt((nblk-1)/nblk * np.sum((Uj-Uj.mean())**2))
    err_chain = np.std(chain_U, ddof=1)/math.sqrt(NCHAIN)
    res = dict(T=T, U=float(U), err_jk=float(err_jk), err_chain=float(err_chain),
               chain_U=[float(x) for x in chain_U], tau_m2=[float(x) for x in taus],
               nsamples=int(tot_n), seeds=seeds, time=time.time()-t0)
    with open("/work/res_T%.2f.json" % T, "w") as f:
        json.dump(res, f)
    print(json.dumps(res))
