import sys, time, os
sys.path.insert(0, '/work')
import numpy as np
from swlib import *
os.makedirs('/work/results', exist_ok=True)

N = 16; R = 256; NSW = 12000; NBURN = 300
Ts_all = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
batch = Ts_all[0:7]
t_start = time.time()
for T in batch:
    i = Ts_all.index(T)
    seed = 20250101 + i
    t0 = time.time()
    S1, S2, S4, _ = run(N, T, R=R, nsweep=NSW, nburn=NBURN, seed=seed, init='random')
    U, err = jackknife_U(S2, S4, NSW)
    m2 = S2.sum()/(NSW*R); m1 = S1.sum()/(NSW*R)
    np.savez(f'/work/results/prod_T{T:.2f}.npz', S1=S1, S2=S2, S4=S4, T=T, N=N, R=R, NSW=NSW, NBURN=NBURN, seed=seed, U=U, err=err)
    print(f"T={T:.2f} seed={seed} U={U:.6f} err={err:.6f} <|m|>={m1:.5f} <m2>={m2:.6f} time={time.time()-t0:.1f}s", flush=True)
    if time.time() - t_start > 480:
        print("stopping early for time budget", flush=True)
        break
print("batch total time", time.time()-t_start)
