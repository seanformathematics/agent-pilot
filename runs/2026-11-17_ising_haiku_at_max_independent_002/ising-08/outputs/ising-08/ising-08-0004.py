import sys, math, random, time
sys.path.insert(0, '/work')
import numpy as np
import ising_wolff as iw

def delta_err(S2b, S4b, blk):
    # S2b,S4b: arrays of block sums (each block has blk flips)
    a = S2b/blk; c = S4b/blk
    B = len(a)
    mu2 = a.mean(); mu4 = c.mean()
    U = 1 - mu4/(3*mu2**2)
    cov = np.cov(np.vstack([c, a]))/B  # order: mu4, mu2
    g = np.array([-1/(3*mu2**2), 2*mu4/(3*mu2**3)])
    return U, math.sqrt(g @ cov @ g)

def lag1(x):
    x = x - x.mean()
    return float(np.sum(x[:-1]*x[1:])/np.sum(x*x))

d = np.load('/work/res_2.80.npz')
S2 = d['S2b']; S4 = d['S4b']; blk = int(d['blk'])
print('shape', S2.shape, 'blk', blk)
print('pooled jk:', iw.jk_U(S2.ravel(), S4.ravel(), np.full(S2.size, float(blk))))
print('pooled delta:', delta_err(S2.ravel(), S4.ravel(), blk))
for k in range(S2.shape[0]):
    print(' chain', k, 'jk', iw.jk_U(S2[k], S4[k], np.full(S2.shape[1], float(blk))), 'delta', delta_err(S2[k], S4[k], blk))
print('lag1 autocorr of block means m2:', round(lag1(S2[0]),4), ' m4:', round(lag1(S4[0]),4))

# synthetic iid test: m Gaussian with variance 0.07 (approx paramagnet), n samples in blocks of 1000
rng = np.random.default_rng(1)
for n in [300000]:
    x = rng.normal(0, math.sqrt(0.0699), size=n)
    m2 = x**2; m4 = m2**2  # treat x as m directly (synthetic)
    blkn = 1000; nb_ = n//blkn
    S2s = m2[:nb_*blkn].reshape(nb_, blkn).sum(1); S4s = m4[:nb_*blkn].reshape(nb_, blkn).sum(1)
    Uj = iw.jk_U(S2s, S4s, np.full(nb_, float(blkn)))
    Ud = delta_err(S2s, S4s, blkn)
    # direct: spread of U over many independent synthetic chains of length n
    Us = []
    for r in range(200):
        xx = rng.normal(0, math.sqrt(0.0699), size=n)
        q2 = xx**2; q4 = q2**2
        Us.append(1 - q4.mean()/(3*q2.mean()**2))
    print('synthetic iid n=%d: jk=%s delta=%s  direct std over 200 reps=%.6f' % (n, Uj, Ud, np.std(Us, ddof=1)))

# fresh independent chains at T=2.8 (16 chains of 1e5 flips)
T = 2.8; N = 256; L = 16
nb = iw.make_nb(L)
padd = 1 - math.exp(-2.0/T)
K2 = 16; NF2 = 100000; BLK2 = 1000
S2n = []; S4n = []
t0 = time.time()
for k in range(K2):
    seed = 555000 + k
    rnd = random.Random(seed).random
    spins = [1 if rnd() < 0.5 else -1 for _ in range(N)]
    mark = [0]*N
    M = sum(spins)
    M, stamp, _, _ = iw.run_flips(spins, mark, 0, M, 2000, padd, rnd, nb, N)
    s2b = []; s4b = []
    for b in range(NF2//BLK2):
        M, stamp, s2, s4 = iw.run_flips(spins, mark, stamp, M, BLK2, padd, rnd, nb, N)
        s2b.append(s2); s4b.append(s4)
    # consistency check: recompute M from spins
    assert sum(spins) == M, (sum(spins), M)
    S2n.append(s2b); S4n.append(s4b)
S2n = np.array(S2n); S4n = np.array(S4n)
Uc = [iw.jk_U(S2n[k], S4n[k], np.full(S2n.shape[1], float(BLK2)))[0] for k in range(K2)]
Ec = [iw.jk_U(S2n[k], S4n[k], np.full(S2n.shape[1], float(BLK2)))[1] for k in range(K2)]
print('fresh 16 chains x 1e5 flips at T=2.8, time %.1fs' % (time.time()-t0))
print(' chain U:', np.round(Uc, 5))
print(' mean chain U = %.6f, std across chains = %.6f (dof %d)' % (np.mean(Uc), np.std(Uc, ddof=1), K2-1))
print(' mean per-chain jackknife err = %.6f' % np.mean(Ec))
print(' pooled 16 chains: jk', iw.jk_U(S2n.ravel(), S4n.ravel(), np.full(S2n.size, float(BLK2))))
