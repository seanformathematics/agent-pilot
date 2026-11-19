
import sys, time
sys.path.insert(0, "/work")
import numpy as np
from run import *

def produce(T, Rh=500, Rc=500, E=500, S=1000, L=50):
    seed_h = 10000 + int(round(T*100)); seed_c = 20000 + int(round(T*100))
    ah = run_T(16, T, R=Rh, E=E, S=S, L=L, seed=seed_h, start="hot")
    ac = run_T(16, T, R=Rc, E=E, S=S, L=L, seed=seed_c, start="cold")
    np.savez(f"/work/prod_T{T:.1f}.npz", hot=ah, cold=ac, seed_h=seed_h, seed_c=seed_c, E=E, S=S, L=L)
    return ah, ac

t0 = time.time()
for T in [2.2,2.3,2.4,2.5,2.6,2.7,2.8]:
    ah, ac = produce(T)
    acc = np.concatenate([ah, ac], axis=1)
    r = summarize(acc)
    print(f"T={T}: plain U={r['plain']['U']:.5f} +- {r['plain']['err_rep']:.5f} | improved U={r['improved']['U']:.5f} +- {r['improved']['err_rep']:.5f} | elapsed {time.time()-t0:.0f}s", flush=True)
