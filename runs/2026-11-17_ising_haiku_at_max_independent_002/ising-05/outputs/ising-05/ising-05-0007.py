
import os, sys, time, math
import numpy as np
os.chdir('/work'); sys.path.insert(0, '/work')

def metro_check(L, T, R, nsweep, ntherm, seed, blk=200):
    rng = np.random.default_rng(seed)
    s = (rng.integers(0, 2, size=(R, L, L), dtype=np.int8)*2 - 1).astype(np.int8)
    ii, jj = np.indices((L, L))
    even = ((ii + jj) % 2 == 0)[None, :, :]
    odd = ~even
    beta = 1.0/T
    tab = np.array([min(1.0, math.exp(-2*beta*(k-4))) for k in range(9)])
    nblk = nsweep//blk
    S2 = np.zeros(nblk, dtype=np.int64); S4 = np.zeros(nblk, dtype=np.int64)
    for sweep in range(ntherm + nsweep):
        for m in (even, odd):
            nn = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
            a = (s*nn).astype(np.int8)
            p = tab[a + 4]
            r = rng.random(s.shape)
            flip = m & (r < p)
            s = np.where(flip, -s, s).astype(np.int8)
        if sweep >= ntherm:
            M = s.reshape(R, -1).sum(axis=1, dtype=np.int64)
            m2 = M*M
            k = (sweep - ntherm)//blk
            S2[k] += int(m2.sum()); S4[k] += int((m2*m2).sum())
    return S2, S4, blk*R

def jk_from_blocks(S2, S4, nper):
    J = len(S2)
    T2 = S2.sum(); T4 = S4.sum(); n = J*nper
    U = 1 - (T4/n)/(3*(T2/n)**2)
    Ui = np.empty(J)
    for j in range(J):
        a = (T2 - S2[j])/((J-1)*nper); b = (T4 - S4[j])/((J-1)*nper)
        Ui[j] = 1 - b/(3*a*a)
    err = np.sqrt((J-1)/J*np.sum((Ui-Ui.mean())**2))
    return U, err

t0 = time.time()
out = []
for T, seed in [(2.8, 31337), (2.5, 4242), (2.3, 8080)]:
    t1 = time.time()
    S2, S4, nper = metro_check(16, T, 512, 4000, 300, seed)
    U, err = jk_from_blocks(S2, S4, nper)
    d = np.load(f'/work/results/T{T:.2f}.npz')
    Uw = 1 - (d['S4'].sum()/(len(d['S4'])*100))/(3*(d['S2'].sum()/(len(d['S2'])*100))**2)
    ew = None
    line = f'T={T}: Metropolis U={U:.5f} +- {err:.5f} (512 reps x 4000 sweeps, seed {seed}) | Wolff U={Uw:.5f} | diff={U-Uw:+.5f} ({(U-Uw)/math.hypot(err,0.0):+.1f} sigma_metro) time {time.time()-t1:.1f}s'
    print(line, flush=True)
    out.append((T, U, err, Uw))
print('total', round(time.time()-t0, 1))
