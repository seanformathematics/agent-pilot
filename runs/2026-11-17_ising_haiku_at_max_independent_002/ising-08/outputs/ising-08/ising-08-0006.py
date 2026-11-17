import numpy as np, time, math
L = 16; N = L*L
ii, jj = np.indices((L, L))
maskA = ((ii+jj) % 2 == 0)
maskB = ~maskA

def half_sweep(s, mask, ptab, rng):
    nbs = np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1) + np.roll(s, 1, axis=2) + np.roll(s, -1, axis=2)
    dE2 = (s*nbs).astype(np.int16) + 4          # index of dE2 in {-4..4}
    u = rng.random(s.shape, dtype=np.float32)
    flip = (u < ptab[dE2]) & mask
    return np.where(flip, -s, s).astype(np.int8)

def jk_U(S2b, S4b, cnt):
    S2 = S2b.sum(); S4 = S4b.sum(); n = cnt.sum()
    U = 1 - n*S4/(3*S2**2)
    Um = 1 - (n-cnt)*(S4-S4b)/(3*(S2-S2b)**2)
    B = len(S2b)
    return float(U), float(math.sqrt((B-1)/B*np.sum((Um-Um.mean())**2)))

def tau_int(x):
    x = np.asarray(x, float); x = x - x.mean(); n = len(x)
    f = np.fft.rfft(x, 2*n); acf = np.fft.irfft(f*np.conj(f), 2*n)[:n]; acf = acf/acf[0]
    tau = 0.5
    for t in range(1, n//2):
        tau += acf[t]
        if t >= 6*tau:
            break
    return tau

def metro(T, R, Lb, t_budget, disc, seed):
    rng = np.random.default_rng(seed)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(R, L, L))
    ptab = np.minimum(1.0, np.exp(-2.0*np.arange(-4, 5)/T))
    for _ in range(disc):
        s = half_sweep(s, maskA, ptab, rng); s = half_sweep(s, maskB, ptab, rng)
    B2 = []; B4 = []; avg2 = []
    t0 = time.time()
    while time.time() - t0 < t_budget:
        acc2 = np.zeros(R); acc4 = np.zeros(R)
        for _ in range(Lb):
            s = half_sweep(s, maskA, ptab, rng); s = half_sweep(s, maskB, ptab, rng)
            m = s.sum(axis=(1, 2))/N
            m2 = m*m
            acc2 += m2; acc4 += m2*m2
            avg2.append(m2.mean())
        B2.append(acc2); B4.append(acc4)
    B2 = np.array(B2); B4 = np.array(B4)   # shape (nblk, R)
    return B2, B4, np.array(avg2), time.time()-t0

results = {}
for T, tb, disc in [(2.8, 150.0, 200), (2.4, 200.0, 500)]:
    B2, B4, avg2, el = metro(T, 2000, 100, tb, disc, seed=31337 + int(T*10))
    nblk, R = B2.shape
    S2b = B2.ravel(); S4b = B4.ravel(); cnt = np.full(S2b.size, 100.0)
    U, e = jk_U(S2b, S4b, cnt)
    # replica-to-replica scatter check: U per replica (blocks summed over time)
    Ur = [jk_U(B2[:, r], B4[:, r], np.full(nblk, 100.0))[0] for r in range(R)]
    pooled_chain_err = np.std(Ur, ddof=1)/math.sqrt(R)
    tau = tau_int(avg2)
    # lag-1 correlation of block sums (blocks of 100 sweeps)
    bm = B2.sum(1) if False else B2.mean(1)
    print(f'Metropolis T={T}: sweeps/replica={nblk*100}, replicas={R}, time={el:.1f}s')
    print(f'   U={U:.6f}  jackknife err={e:.6f}  replica-scatter err={pooled_chain_err:.6f}  tau_int(m2, sweeps)={tau:.2f}')
    results[T] = (U, e)

# Wolff comparison values (pooled over all Wolff batches for each T)
import sys; sys.path.insert(0, '/work'); import ising_wolff as iw
import os
for T in [2.8, 2.4]:
    parts = []
    for fn in [f'/work/res_{T:.2f}.npz', f'/work/res2_{T:.2f}.npz']:
        if os.path.exists(fn):
            d = np.load(fn); parts.append((d['S2b'].ravel(), d['S4b'].ravel(), np.full(d['S2b'].size, 1000.0)))
    S2 = np.concatenate([p[0] for p in parts]); S4 = np.concatenate([p[1] for p in parts]); c = np.concatenate([p[2] for p in parts])
    Uw, ew = jk_U(S2, S4, c)
    Um, em = results[T]
    print(f'T={T}: Wolff pooled U={Uw:.6f} +- {ew:.6f} (n_flips={int(c.sum())});  Metropolis U={Um:.6f} +- {em:.6f};  diff={Um-Uw:+.6f}, sigma_diff={math.sqrt(ew**2+em**2):.6f}')
