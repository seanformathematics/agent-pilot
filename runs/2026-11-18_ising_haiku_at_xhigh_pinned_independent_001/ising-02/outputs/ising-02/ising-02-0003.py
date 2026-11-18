import time, random, math
import numpy as np

N = 16
NN = N * N
nbr = []
for x in range(N):
    for y in range(N):
        nbr.append([((x + 1) % N) * N + y, ((x - 1) % N) * N + y, x * N + (y + 1) % N, x * N + (y - 1) % N])

def run_wolff(T, nsteps, seed):
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
    return Ms

def U_of(m2, m4):
    return 1 - m4.mean() / (3 * m2.mean()**2)

def autocorr(x, maxlag=2000):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf /= acf[0]
    return acf

for T in [1.5, 2.269, 2.8]:
    t0 = time.time()
    Ms = run_wolff(T, 1000000, 2024)
    m = Ms / NN
    m2 = m**2; m4 = m**4
    burn = 5000
    m2 = m2[burn:]; m4 = m4[burn:]
    print("T", T, "time", round(time.time()-t0,1), "U", U_of(m2, m4))
    acf = autocorr(m2, 3000)
    # integrated tau with window
    tau = 0.5
    for w in range(1, 3000):
        tau += acf[w]
        if w > 6*tau: break
    print("  tau_int(m2) ~", round(tau,2), " window", w)
    # block jackknife for several block sizes
    for bs in [500, 2000, 10000, 50000]:
        nb = len(m2)//bs
        a = m2[:nb*bs].reshape(nb, bs); b = m4[:nb*bs].reshape(nb, bs)
        S2 = a.sum(1); S4 = b.sum(1)
        Ufull = 1 - (S4.sum()/(nb*bs)) / (3*(S2.sum()/(nb*bs))**2)
        Uj = np.empty(nb)
        for k in range(nb):
            s2 = (S2.sum()-S2[k]); s4=(S4.sum()-S4[k]); n = (nb-1)*bs
            Uj[k] = 1 - (s4/n)/(3*(s2/n)**2)
        err = math.sqrt((nb-1)/nb * np.sum((Uj - Uj.mean())**2))
        print("   bs", bs, "nb", nb, "U", round(Ufull,5), "jk err", round(err,5))
