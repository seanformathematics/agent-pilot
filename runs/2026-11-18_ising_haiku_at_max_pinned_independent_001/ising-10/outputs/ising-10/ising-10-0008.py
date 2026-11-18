
import sys, time, numpy as np
sys.path.insert(0, '/work')
import sw, prod

Tlist = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]
t0 = time.time()
for T in Tlist:
    seed = 60000 + int(round(T*10))
    B2, B4, NN, S = prod.run_T(T, seed)
    np.savez(f'/work/rep2_T{T:.2f}.npz', B2=B2, B4=B4, NN=NN, S=S, seed=seed, R=1000, burn=200)
    U, err = sw.binder_from_sums(B2, B4, S)
    print(f"REP2 T={T:.2f} seed={seed} U={U:.7f} err={err:.7f} elapsed={time.time()-t0:.1f}", flush=True)
