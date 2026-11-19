
prod = r'''
import sys, time, pickle, os
sys.path.insert(0, '/work')
import numpy as np
import sw

TS = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]

def produce(label, seed_base, Ts=TS, R=500, n_equil=200, n_meas=4000, B=100, e_every=5, tmax=500):
    t0 = time.time()
    for k, T in enumerate(Ts):
        fn = f'/work/prod{label}_k{k}.pkl'
        if os.path.exists(fn):
            continue
        if time.time() - t0 > tmax:
            print("time guard: stopping before T =", T, flush=True)
            break
        t1 = time.time()
        r = sw.run_T(16, R, T, seed=seed_base + k, n_equil=n_equil, n_meas=n_meas, B=B, e_every=e_every)
        pickle.dump(r, open(fn, 'wb'))
        blk = r['blk']
        s2 = blk[0].mean(axis=1); s4 = blk[1].mean(axis=1)
        U, err = sw.U_jack(s2, s4)
        print(f"set {label} T={T}: U={U:.5f} +- {err:.5f}  ({time.time()-t1:.1f}s, total {time.time()-t0:.0f}s)", flush=True)
'''
open('/work/prod.py','w').write(prod)
import sys
sys.path.insert(0,'/work')
import prod
prod.produce('A', 11000)
