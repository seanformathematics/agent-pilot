
import os, sys, time, math
os.makedirs('/work', exist_ok=True)
os.chdir('/work')
SRC = r'''
import math, random, time
import numpy as np

def make_nbrs(L):
    nbrs = []
    for x in range(L):
        for y in range(L):
            nbrs.append((((x+1) % L)*L + y, ((x-1) % L)*L + y, x*L + (y+1) % L, x*L + (y-1) % L))
    return nbrs

def wolff_run(L, T, seed, n_therm, n_blocks, block_len, nbrs=None, init_mode=0):
    """Wolff cluster Ising simulation. Returns per-block sums of M^2 and M^4 (M = total spin sum),
    and number of sign changes of M during measurement."""
    if nbrs is None:
        nbrs = make_nbrs(L)
    N = L*L
    rnd = random.Random(seed).random
    padd = 1.0 - math.exp(-2.0/T)
    if init_mode == 0:
        spins = [1 if rnd() < 0.5 else -1 for _ in range(N)]
    else:
        spins = [1]*N
    M = sum(spins)
    stamp = [0]*N
    cur = 0
    S2 = np.zeros(n_blocks, dtype=np.int64)
    S4 = np.zeros(n_blocks, dtype=np.int64)
    nflip = 0
    Mprev = M
    total = n_therm + n_blocks*block_len
    s2 = 0; s4 = 0; cnt = 0; b = 0
    for step in range(total):
        cur += 1
        i0 = int(rnd()*N)
        s0 = spins[i0]
        stamp[i0] = cur
        clust = [i0]
        for site in clust:
            for nb in nbrs[site]:
                if stamp[nb] != cur and spins[nb] == s0 and rnd() < padd:
                    stamp[nb] = cur
                    clust.append(nb)
        for site in clust:
            spins[site] = -s0
        M -= 2*s0*len(clust)
        if step >= n_therm:
            if (M > 0 and Mprev < 0) or (M < 0 and Mprev > 0):
                nflip += 1
            Mprev = M
            m2 = M*M
            s2 += m2
            s4 += m2*m2
            cnt += 1
            if cnt == block_len:
                S2[b] = s2; S4[b] = s4
                b += 1; s2 = 0; s4 = 0; cnt = 0
    return S2, S4, nflip

def jk_U(S2, S4, block_len, J=100):
    nb = len(S2); g = nb // J
    G2 = S2[:g*J].reshape(J, g).sum(axis=1).astype(float)
    G4 = S4[:g*J].reshape(J, g).sum(axis=1).astype(float)
    n_all = float(g*J*block_len); n_loo = float((J-1)*g*block_len)
    T2 = G2.sum(); T4 = G4.sum()
    U_full = 1 - (T4/n_all)/(3*(T2/n_all)**2)
    Ui = 1 - ((T4-G4)/n_loo)/(3*((T2-G2)/n_loo)**2)
    err = np.sqrt((J-1)/J*np.sum((Ui-Ui.mean())**2))
    return U_full, err

def exact_U(L, T):
    N = L*L
    nbrs = make_nbrs(L)
    idx = np.arange(2**N)
    conf = ((idx[:, None] >> np.arange(N)[None, :]) & 1)*2 - 1
    E = np.zeros(2**N)
    for i in range(N):
        for j in (nbrs[i][0], nbrs[i][2]):
            E -= conf[:, i]*conf[:, j]
    M = conf.sum(axis=1).astype(float)
    w = np.exp(-(E-E.min())/T)
    Z = w.sum()
    m2 = (w*M**2).sum()/Z
    m4 = (w*M**4).sum()/Z
    return 1 - m4/(3*m2*m2)

def autocorr_time(x, maxlag=20000):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    for t in range(1, min(maxlag, n-1)):
        tau += acf[t]
        if t >= 6*tau:
            return tau
    return tau
'''
open('/work/wolff_py.py', 'w').write(SRC)
sys.path.insert(0, '/work')
import numpy as np
import wolff_py as W

# 1) validation on 4x4 vs exact enumeration
for T in [2.0, 2.6, 3.5]:
    t1 = time.time()
    S2, S4, nf = W.wolff_run(4, T, 12345, 5000, 400000, 1)
    U, err = W.jk_U(S2, S4, 1, J=50)
    print(f'N=4 T={T}: Wolff U={U:.5f} +- {err:.5f}  exact={W.exact_U(4, T):.5f}  ({time.time()-t1:.1f}s)')

# 2) pilot N=16 at Tc
Tc = 2/math.log(1+math.sqrt(2))
t1 = time.time()
S2, S4, nf = W.wolff_run(16, Tc, 777, 10000, 300000, 1)
el = time.time()-t1
print(f'N=16 Tc={Tc:.5f}: 3e5 steps in {el:.1f}s -> {el/3e5*1e6:.1f} us/step; sign flips {nf}')
U, err = W.jk_U(S2, S4, 1, J=100)
print('U =', round(U, 5), ' jk err J=100:', round(err, 5))
for J in [10, 30, 100, 300, 1000]:
    print('  J', J, 'err', round(W.jk_U(S2, S4, 1, J=J)[1], 5))
print('tau_int M^2:', round(W.autocorr_time(S2), 2), ' tau_int M^4:', round(W.autocorr_time(S4), 2))
print('<M^2>/N^2 =', S2.mean()/256**2)

# 3) quick look at low and high T
for T in [1.5, 2.8]:
    t1 = time.time()
    S2b, S4b, nfb = W.wolff_run(16, T, 2024, 5000, 100000, 1)
    U, err = W.jk_U(S2b, S4b, 1, J=100)
    print(f'N=16 T={T}: U={U:.5f} +- {err:.5f}, sign flips {nfb}, tau M^2 {W.autocorr_time(S2b):.2f}, time {time.time()-t1:.1f}s')
