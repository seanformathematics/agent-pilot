
import os, sys, time
os.makedirs('/work', exist_ok=True)
os.chdir('/work')
SRC = r'''
import numpy as np
from numba import njit

def make_nbr(L):
    N = L*L
    nbr = np.empty((N, 4), dtype=np.int64)
    for x in range(L):
        for y in range(L):
            i = x*L + y
            nbr[i, 0] = ((x+1) % L)*L + y
            nbr[i, 1] = ((x-1) % L)*L + y
            nbr[i, 2] = x*L + (y+1) % L
            nbr[i, 3] = x*L + (y-1) % L
    return nbr

@njit(cache=True)
def wolff_blocks(nbr, N, beta, seed, n_therm, n_blocks, block_len, init_mode):
    np.random.seed(seed)
    spins = np.ones(N, dtype=np.int64)
    if init_mode == 0:
        for i in range(N):
            if np.random.random() < 0.5:
                spins[i] = -1
    M = 0
    for i in range(N):
        M += spins[i]
    padd = 1.0 - np.exp(-2.0*beta)
    stamp = np.zeros(N, dtype=np.int64)
    clust = np.empty(N, dtype=np.int64)
    S2 = np.zeros(n_blocks, dtype=np.int64)
    S4 = np.zeros(n_blocks, dtype=np.int64)
    cur = 0
    total = n_therm + n_blocks*block_len
    s2 = 0
    s4 = 0
    cnt = 0
    b = 0
    nflip = 0
    Mprev = M
    for step in range(total):
        cur += 1
        i0 = int(np.random.random()*N)
        s0 = spins[i0]
        stamp[i0] = cur
        clust[0] = i0
        n = 1
        head = 0
        while head < n:
            site = clust[head]
            head += 1
            for k in range(4):
                nb = nbr[site, k]
                if stamp[nb] != cur and spins[nb] == s0:
                    if np.random.random() < padd:
                        stamp[nb] = cur
                        clust[n] = nb
                        n += 1
        for k in range(n):
            spins[clust[k]] = -s0
        M -= 2*s0*n
        if step >= n_therm:
            if (M > 0 and Mprev < 0) or (M < 0 and Mprev > 0):
                nflip += 1
            Mprev = M
            m2 = M*M
            s2 += m2
            s4 += m2*m2
            cnt += 1
            if cnt == block_len:
                S2[b] = s2
                S4[b] = s4
                b += 1
                s2 = 0
                s4 = 0
                cnt = 0
    return S2, S4, nflip
'''
open('/work/ising_wolff.py', 'w').write(SRC)
sys.path.insert(0, '/work')

import numpy as np
try:
    import numba
    print('numba', numba.__version__, 'OK')
except Exception as e:
    print('numba import failed:', e)
    raise
import multiprocessing
print('cpus', multiprocessing.cpu_count(), 'numpy', np.__version__, 'python', sys.version.split()[0])
import ising_wolff as iw

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
    nbr = iw.make_nbr(L)
    idx = np.arange(2**N)
    conf = ((idx[:, None] >> np.arange(N)[None, :]) & 1)*2 - 1
    E = np.zeros(2**N)
    for i in range(N):
        for k in (0, 2):
            j = nbr[i, k]
            E -= conf[:, i]*conf[:, j]
    M = conf.sum(axis=1).astype(float)
    w = np.exp(-(E-E.min())/T)
    Z = w.sum()
    m2 = (w*M**2).sum()/Z
    m4 = (w*M**4).sum()/Z
    return 1 - m4/(3*m2*m2)

nb4 = iw.make_nbr(4)
for T in [2.0, 2.6, 3.5]:
    t1 = time.time()
    S2, S4, nfl = iw.wolff_blocks(nb4, 16, 1.0/T, 12345, 10000, 2000000, 1, 0)
    U, err = jk_U(S2, S4, 1, J=50)
    print(f'N=4 T={T}: Wolff U={U:.5f} +- {err:.5f}   exact U={exact_U(4, T):.5f}   time {time.time()-t1:.2f}s')

Tc = 2/np.log(1+np.sqrt(2))
nb16 = iw.make_nbr(16)
t1 = time.time()
S2, S4, nfl = iw.wolff_blocks(nb16, 256, 1.0/Tc, 777, 20000, 1000000, 1, 0)
el = time.time()-t1
print(f'N=16 pilot Tc={Tc:.5f}: 1e6 steps in {el:.2f}s (incl. compile)')
U, err = jk_U(S2, S4, 1, J=100)
print('U =', U, ' jk err (J=100) =', err)
for J in [10, 20, 50, 200, 1000]:
    print('  J', J, 'err', jk_U(S2, S4, 1, J=J)[1])
print('mean M^2/N^2 per step', S2.mean()/256**2, ' sign flips', nfl)

def autocorr_time(x, maxlag=5000):
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, 2*n)
    acf = np.fft.irfft(f*np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    for t in range(1, maxlag):
        tau += acf[t]
        if t >= 6*tau:
            return tau, t
    return tau, maxlag
print('tau_int M^2 :', autocorr_time(S2.astype(float)))
print('tau_int M^4 :', autocorr_time(S4.astype(float)))

for T in [1.5, 2.8]:
    t1 = time.time()
    S2b, S4b, nfl = iw.wolff_blocks(nb16, 256, 1.0/T, 2024, 20000, 200000, 1, 0)
    U, err = jk_U(S2b, S4b, 1, J=100)
    print(f'N=16 T={T}: U={U:.5f} +- {err:.5f}, sign flips {nfl}, time {time.time()-t1:.2f}s')
