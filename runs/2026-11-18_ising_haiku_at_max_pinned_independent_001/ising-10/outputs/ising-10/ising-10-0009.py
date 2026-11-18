
import sys, time, numpy as np
sys.path.insert(0, '/work')
import sw, prod

Tlist = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
t0 = time.time()
for T in Tlist:
    seed = 60000 + int(round(T*10))
    B2, B4, NN, S = prod.run_T(T, seed)
    np.savez(f'/work/rep2_T{T:.2f}.npz', B2=B2, B4=B4, NN=NN, S=S, seed=seed, R=1000, burn=200)
    U, err = sw.binder_from_sums(B2, B4, S)
    print(f"REP2 T={T:.2f} seed={seed} U={U:.7f} err={err:.7f} elapsed={time.time()-t0:.1f}", flush=True)
