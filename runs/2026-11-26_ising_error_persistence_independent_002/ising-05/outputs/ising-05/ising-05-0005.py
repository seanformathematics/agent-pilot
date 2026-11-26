
import numpy as np, time, sys
sys.path.insert(0,'/work')
from ising import run, binder_jk
for T, seed in [(2.3, 105), (2.4, 106)]:
    t0 = time.time()
    A2, A4, B2, B4 = run(T, R=1000, nsweep=20000, ntherm=2000, seed=seed)
    U, err = binder_jk(A2, A4)
    Ub = [1 - b4.mean()/(3*b2.mean()**2) for b2, b4 in zip(B2, B4)]
    np.savez(f'/work/res_T{T}.npz', A2=A2, A4=A4, B2=B2, B4=B4)
    print(T, "U=%.5f err=%.5f" % (U, err), "blocks:", np.round(Ub, 4), "time %.1f" % (time.time()-t0), flush=True)
