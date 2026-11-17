import sys, time, os, math
sys.path.insert(0, '/work')
import numpy as np
import ising
print("affinity cpus:", len(os.sched_getaffinity(0)))

out = {}
for T, seed0 in [(2.8, 50000), (2.5, 60000)]:
    t0 = time.time()
    Us, Jerr = [], []
    for k in range(20):
        M, _ = ising.wolff_chain(16, T, 200000, 5000, seed=seed0+k)
        U, Ujk, Ue = ising.binder_jk(M, 200)
        Us.append(U); Jerr.append(Ue)
    Us = np.array(Us)
    dt = time.time()-t0
    print(f"T={T}: chain U mean={Us.mean():.5f}, std across chains={Us.std(ddof=1):.5f}, "
          f"mean jackknife err per chain={np.mean(Jerr):.5f} (rms {np.sqrt(np.mean(np.square(Jerr))):.5f}), "
          f"std/sqrt(20)={Us.std(ddof=1)/np.sqrt(20):.5f}, time {dt:.1f}s", flush=True)
    print("  chain U values:", np.round(Us, 4).tolist())
