import sys, time
sys.path.insert(0, '/work')
import numpy as np
import ising

for T, nch, seed0 in [(2.6, 20, 8000000), (2.4, 12, 8100000)]:
    t0 = time.time()
    Us, jk = [], []
    for k in range(nch):
        M, _ = ising.wolff_chain(16, T, 500000, 5000, seed=seed0+k)
        U, Ujk, e = ising.binder_jk(M, 250)
        Us.append(U); jk.append(e)
    Us = np.array(Us); jk = np.array(jk)
    pooled_jk_equiv = np.mean(jk)/np.sqrt(4)   # expected error of a 4-chain pool (2e6 samples)
    print(f"T={T}: {nch} chains x 5e5: mean U={Us.mean():.6f}; chain std={Us.std(ddof=1):.6f} "
          f"-> 4-chain pool error from scatter={Us.std(ddof=1)/2:.6f}; "
          f"mean per-chain jackknife err={jk.mean():.6f} -> 4-chain pool={pooled_jk_equiv:.6f}; "
          f"time {time.time()-t0:.1f}s", flush=True)
    print("  chain U:", np.round(Us, 5).tolist(), flush=True)
