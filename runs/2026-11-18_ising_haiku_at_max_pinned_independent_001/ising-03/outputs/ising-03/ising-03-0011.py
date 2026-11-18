import sys, os, time, json
sys.path.insert(0, '/work')
import numpy as np
from wolff_lib import wolff_series, U_of, jackknife

t_start = time.time()
K = 8; NUPD = 125000; NEQ = 5000
seedlog = {}
for T in [2.3, 2.6]:
    iT = int(round((T-1.5)*10))
    for k in range(K):
        fname = f'/work/data/B4_T{T:.1f}_c{k}.npy'
        if os.path.exists(fname):
            continue
        seed = 300000 + 1000*(iT+1) + k + 17
        M = wolff_series(16, T, seed, NUPD, NEQ)
        np.save(fname, M)
        seedlog[f'B4_T{T:.1f}_c{k}'] = seed
json.dump(seedlog, open('/work/data/seeds_B4.json', 'w'), indent=1)
print("batch4 time", round(time.time()-t_start,1))

# Autocorrelation analysis of x=M^2 and y=M^4 at T=2.3 and 2.6 (averaged over chains)
def acf_avg(chains, f, maxlag):
    acc = np.zeros(maxlag+1); cnt = 0
    for c in chains:
        v = f(c.astype(np.float64))
        v = v - v.mean()
        n = len(v)
        ff = np.fft.rfft(v, 2*n)
        ac = np.fft.irfft(ff*np.conj(ff))[:maxlag+1]
        acc += ac/ac[0]; cnt += 1
    return acc/cnt
for T in [2.3, 2.6]:
    chains = [np.load(f'/work/data/{p}{T:.1f}_c{k}.npy') for p in ['T','B2_T','B3_T'] for k in range(K)] if T==2.3 else None
    if T == 2.6:
        chains = [np.load(f'/work/data/{p}{T:.1f}_c{k}.npy') for p in ['T','B2_T','B3_T'] for k in range(K)]
    a2 = acf_avg(chains, lambda c: c**2, 60)
    a4 = acf_avg(chains, lambda c: c**4, 60)
    print(f"T={T}: ACF(M^2) lags 1..8:", np.round(a2[1:9],3), " ACF(M^4) lags 1..8:", np.round(a4[1:9],3))
    print(f"        ACF(M^2) at lags 10,20,40,60:", np.round(a2[[10,20,40,60]],4))
    # integrated tau (sum with cutoff at 6*tau window)
    tau = 0.5
    for t in range(1, 61):
        tau += a2[t]
    print(f"        integrated tau(M^2) summed to lag 60: {tau:.2f}")

# Batch-4 analysis at T=2.3, 2.6 combined with batches 1-3
def loadb(T, prefix):
    return [np.load(f'/work/data/{prefix}{T:.1f}_c{k}.npy').astype(np.int64) for k in range(K)]
def blocks_from(chains, bsize):
    out = []
    for c in chains:
        nb = len(c)//bsize
        out += [c[i*bsize:(i+1)*bsize] for i in range(nb)]
    return out
for T in [2.3, 2.6]:
    bl = {'b1': loadb(T, 'T'), 'b2': loadb(T, 'B2_T'), 'b3': loadb(T, 'B3_T'), 'b4': loadb(T, 'B4_T')}
    Ub = {}; eb = {}
    for name, ch in bl.items():
        Ub[name], eb[name] = jackknife(blocks_from(ch, 5000))
    Uall, eall = jackknife(blocks_from(sum(bl.values(), []), 5000))
    Uarr = np.array(list(Ub.values())); earr = np.array(list(eb.values()))
    # weighted mean across 4 batches and chi-square (3 dof)
    w = 1/earr**2
    Uw = np.sum(w*Uarr)/np.sum(w)
    chi2 = np.sum((Uarr-Uw)**2*w)
    print(f"T={T}: pooled U={Uall:.5f} err_jk={eall:.5f} | batch U: " + ", ".join(f"{k}={Ub[k]:.5f}(e={eb[k]:.5f})" for k in Ub) +
          f" | chi2(3dof)={chi2:.2f}")
    # per-chain U values scatter
    Uc = np.array([U_of(c) for c in sum(bl.values(), [])])
    print(f"        per-chain U: mean={Uc.mean():.5f} std={Uc.std(ddof=1):.5f} (std/sqrt(32)={Uc.std(ddof=1)/np.sqrt(len(Uc)):.5f})")
