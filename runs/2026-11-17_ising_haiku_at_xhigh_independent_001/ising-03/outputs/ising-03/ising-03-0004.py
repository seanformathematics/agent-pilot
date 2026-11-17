import sys, time, json
sys.path.insert(0,'/work')
import numpy as np
from sw import run

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R = 64; NEQ = 300; NMEAS = 20000
results = {}
t0 = time.time()
for k, T in enumerate(Ts):
    seed = 1000 + k
    t = time.time()
    m2s, m4s = run(16, R, T, seed, NEQ, NMEAS)
    s2 = m2s.sum(0); s4 = m4s.sum(0)   # per-chain sums
    n = NMEAS
    M2 = s2.sum()/(R*n); M4 = s4.sum()/(R*n)
    U = 1 - M4/(3*M2**2)
    # delete-one-chain jackknife
    tot2 = s2.sum(); tot4 = s4.sum()
    Uj = np.empty(R)
    for r in range(R):
        a2 = (tot2 - s2[r])/((R-1)*n); a4 = (tot4 - s4[r])/((R-1)*n)
        Uj[r] = 1 - a4/(3*a2**2)
    err_j = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    # block (time) check: 20 blocks of 1000 sweeps, U per block over all chains
    nb = 20; bl = NMEAS//nb
    Ub = []
    for b in range(nb):
        b2 = m2s[b*bl:(b+1)*bl].mean(); b4 = m4s[b*bl:(b+1)*bl].mean()
        Ub.append(1 - b4/(3*b2**2))
    Ub = np.array(Ub)
    # time-block jackknife error (20 blocks)
    Uh = np.array([1 - ((m4s.sum()-m4s[b*bl:(b+1)*bl].sum())/((nb-1)*bl*R)) /
                   (3*((m2s.sum()-m2s[b*bl:(b+1)*bl].sum())/((nb-1)*bl*R))**2) for b in range(nb)])
    err_t = np.sqrt((nb-1)/nb*np.sum((Uh-Uh.mean())**2))
    results[str(T)] = dict(T=T, U=float(U), err_chain_jack=float(err_j), err_block_jack=float(err_t),
                           block_U=[float(x) for x in Ub], seed=seed, M2=float(M2), M4=float(M4))
    print(f"T={T}: U={U:.5f}  err(chain-jack)={err_j:.5f}  err(time-block-jack)={err_t:.5f}  "
          f"block U range [{Ub.min():.5f},{Ub.max():.5f}]  ({time.time()-t:.1f}s)", flush=True)
    json.dump(results, open('/work/results.json','w'), indent=1)
    if time.time()-t0 > 480:
        print("stopping early for time; done up to", T); break
print("total", time.time()-t0)
