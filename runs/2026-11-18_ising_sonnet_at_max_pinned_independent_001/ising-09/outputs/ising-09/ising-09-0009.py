
import sys, time
sys.path.insert(0, "/work")
import numpy as np
from metro import *
t0=time.time()
res = {}
for T, start in [(2.0,"cold"), (2.3,"hot"), (2.6,"hot")]:
    acc = run_metro(16, T, R=1000, E=1500, S=6000, L=100, seed=31337+int(T*100), start=start)
    rep2 = acc[0].mean(axis=1); rep4 = acc[1].mean(axis=1)
    U, err, Ujk = jackknife_U(rep2, rep4)
    # first-half vs second-half check
    h = acc.shape[2]//2
    Ua,_,_ = jackknife_U(acc[0][:, :h].mean(axis=1), acc[1][:, :h].mean(axis=1))
    Ub,_,_ = jackknife_U(acc[0][:, h:].mean(axis=1), acc[1][:, h:].mean(axis=1))
    res[T] = (U, err)
    print(f"Metropolis T={T} ({start} start): U={U:.5f} +- {err:.5f}  (m2={rep2.mean():.5f}, m4={rep4.mean():.5f});  1st half U={Ua:.5f}, 2nd half U={Ub:.5f}   elapsed {time.time()-t0:.0f}s")
np.savez("/work/metro_res.npz", **{f"T{T}": np.array(v) for T,v in res.items()})
