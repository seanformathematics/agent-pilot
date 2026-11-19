
import sys, time, os
sys.path.insert(0,"/work")
import numpy as np
from ising import wolff_run

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R = 20
n_therm = 5000
n_meas = {1.5:20000, 1.6:20000, 1.7:20000, 1.8:20000, 1.9:20000, 2.0:20000,
          2.1:40000, 2.2:40000, 2.3:100000, 2.4:100000,
          2.5:200000, 2.6:200000, 2.7:200000, 2.8:200000}
t0 = time.time()
for iT, T in enumerate(Ts):
    fn = f"/work/wolffB_T{T}.npz"
    if os.path.exists(fn):
        print("skip", T); continue
    nm = n_meas[T]
    series = np.empty((R, nm), dtype=np.int16)
    th = np.empty((R, n_therm), dtype=np.int16)
    cbars = np.empty(R); seeds = np.empty(R, dtype=np.int64); inits = np.empty(R, dtype=np.int8)
    t = time.time()
    for r in range(R):
        seed = 920000 + 100*iT + r
        init = "hot" if r % 2 == 0 else "cold"      # reversed relative to batch A
        M, cbar, thm = wolff_run(16, T, seed, n_therm, nm, init)
        series[r] = M; th[r] = thm; cbars[r] = cbar; seeds[r] = seed; inits[r] = (init == "hot")
    np.savez(fn, M=series, th=th, T=T, seeds=seeds, cbar=cbars, hot=inits)
    print(f"T={T}: done {R}x{nm} flips in {time.time()-t:.1f}s (elapsed {time.time()-t0:.0f}s)", flush=True)
    if time.time() - t0 > 540:
        print("stopping early to respect time limit"); break
