import sys, time
sys.path.insert(0,'/work')
import numpy as np, ising
res={}
t0=time.time()
for T in [2.3, 2.4, 2.8]:
    Us=[]
    for s in range(8):
        out = ising.run(T, R=256, n_equil=2000, n_meas=4000, seed=9000+int(T*100)+s, start='random')
        U,_ = ising.binder_jk(out['A'],out['B'],out['K'])
        Us.append(U)
    Us=np.array(Us)
    sd256 = Us.std(ddof=1)
    print(f"T={T}: mean U(256 reps)={Us.mean():.5f}  empirical sd(256)={sd256:.5f} -> scaled to 1024 reps: {sd256/2:.5f}  time={time.time()-t0:.0f}s", flush=True)
