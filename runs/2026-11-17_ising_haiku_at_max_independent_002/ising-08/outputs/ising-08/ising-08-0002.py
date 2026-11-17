import random, math, time
import numpy as np

def make_nb(L):
    nb = []
    for i in range(L):
        for j in range(L):
            nb.append((((i+1) % L)*L + j, ((i-1) % L)*L + j, i*L + (j+1) % L, i*L + (j-1) % L))
    return nb

def run_flips(spins, mark, stamp, M, nfl, padd, rnd, nb, N, Mrec=None):
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
        if Mrec is not None:
            Mrec.append(M)
        m2 = (M/Nf)**2
        S2 += m2
        S4 += m2*m2
    return M, stamp, S2, S4

def exact_U_4x4(T):
    L = 4; N = 16
    bonds = []
    for i in range(L):
        for j in range(L):
            a = i*L+j
            bonds.append((a, ((i+1) % L)*L+j))
            bonds.append((a, i*L+(j+1) % L))
    confs = np.arange(2**N, dtype=np.int64)
    bits = ((confs[:, None] >> np.arange(N)) & 1)
    s = (2*bits - 1).astype(np.int64)
    E = -np.sum(np.stack([s[:, a]*s[:, b] for a, b in bonds], axis=1), axis=1)
    m = s.mean(axis=1)
    w = np.exp(-(E - E.min())/T)
    Z = w.sum()
    m2 = np.sum(w*m**2)/Z
    m4 = np.sum(w*m**4)/Z
    return 1 - m4/(3*m2**2)

def jk_U(S2b, S4b, cnt):
    S2 = S2b.sum(); S4 = S4b.sum(); n = cnt.sum()
    U = 1 - n*S4/(3*S2**2)
    Um = 1 - (n-cnt)*(S4-S4b)/(3*(S2-S2b)**2)
    B = len(S2b)
    err = math.sqrt((B-1)/B*np.sum((Um-Um.mean())**2))
    return U, err

def tau_int(x):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f), 2*n)[:n]
    acf = acf/acf[0]
    tau = 0.5
    for t in range(1, n//2):
        tau += acf[t]
        if t >= 6*tau:
            break
    return tau

# ---- 1. validation on 4x4 ----
for T in [2.0, 3.0]:
    Uex = exact_U_4x4(T)
    L = 4; N = 16; nb = make_nb(L)
    rnd = random.Random(12345).random
    spins = [1 if rnd() < 0.5 else -1 for _ in range(N)]
    mark = [0]*N
    padd = 1 - math.exp(-2.0/T)
    M = sum(spins)
    M, stamp, _, _ = run_flips(spins, mark, 0, M, 2000, padd, rnd, nb, N)
    Mrec = []
    M, stamp, S2, S4 = run_flips(spins, mark, stamp, M, 200000, padd, rnd, nb, N, Mrec)
    Ms = np.array(Mrec, float)/N
    m2 = Ms**2; m4 = m2**2
    Uw = 1 - m4.mean()/(3*m2.mean()**2)
    cnt = np.full(100, len(m2)//100, float)
    S2b = m2[:len(cnt)*int(cnt[0])].reshape(100, -1).sum(1)
    S4b = m4[:len(cnt)*int(cnt[0])].reshape(100, -1).sum(1)
    Uj, ej = jk_U(S2b, S4b, cnt)
    print(f"4x4 T={T}: exact U={Uex:.6f}  Wolff U={Uw:.6f}  jackknife err={ej:.6f}  diff/err={(Uw-Uex)/ej:.2f}")

# ---- 2. timing and pilot on 16x16 ----
L = 16; N = 256; nb = make_nb(L)
for T, nfl in [(1.5, 20000), (2.269, 100000), (2.8, 40000)]:
    rnd = random.Random(777).random
    spins = [1 if rnd() < 0.5 else -1 for _ in range(N)]
    mark = [0]*N
    padd = 1 - math.exp(-2.0/T)
    M = sum(spins)
    t0 = time.perf_counter()
    M, stamp, _, _ = run_flips(spins, mark, 0, M, 2000, padd, rnd, nb, N)
    t1 = time.perf_counter()
    Mrec = []
    M, stamp, S2, S4 = run_flips(spins, mark, stamp, M, nfl, padd, rnd, nb, N, Mrec)
    t2 = time.perf_counter()
    Ms = np.array(Mrec, float)/N
    m2 = Ms**2; m4 = m2**2
    B = 100; blk = len(m2)//B
    S2b = m2[:B*blk].reshape(B, blk).sum(1); S4b = m4[:B*blk].reshape(B, blk).sum(1)
    cnt = np.full(B, float(blk))
    Uj, ej = jk_U(S2b, S4b, cnt)
    tau2 = tau_int(m2); tau4 = tau_int(m4)
    print(f"16x16 T={T}: discard 2000 in {t1-t0:.2f}s, {nfl} flips in {t2-t1:.2f}s ({(t2-t1)/nfl*1e6:.1f} us/flip)")
    print(f"   <|m|>={np.mean(np.abs(Ms)):.4f} <m2>={m2.mean():.5f} U={Uj:.5f} jk_err={ej:.5f} tau_int(m2)={tau2:.2f} tau_int(m4)={tau4:.2f}")
    print(f"   sign changes of M: {int(np.sum(np.sign(Ms[1:]) != np.sign(Ms[:-1])))}")
