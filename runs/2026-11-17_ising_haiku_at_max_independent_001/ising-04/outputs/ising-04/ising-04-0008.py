import sys, time
sys.path.insert(0, '/work')
import numpy as np
from swlib import *

N = 16
t0 = time.time()
out = {}
for T, base in [(2.5, 30000), (2.8, 31000)]:
    Us = []; Es = []
    for k in range(20):
        S1, S2, S4, _ = run(N, T, R=64, nsweep=3000, nburn=300, seed=base+k)
        U, e = jackknife_U(S2, S4, 3000)
        Us.append(U); Es.append(e)
    Us = np.array(Us); Es = np.array(Es)
    ratio = Us.std(ddof=1)/Es.mean()
    out[T] = (Us.mean(), Us.std(ddof=1), Es.mean(), ratio)
    print(f"T={T}: 20 seeds x 64 reps x 3000 sweeps: mean U={Us.mean():.5f}, std across seeds={Us.std(ddof=1):.5f}, "
          f"mean jackknife err={Es.mean():.5f}, ratio std/jk={ratio:.3f}, elapsed={time.time()-t0:.0f}s", flush=True)
