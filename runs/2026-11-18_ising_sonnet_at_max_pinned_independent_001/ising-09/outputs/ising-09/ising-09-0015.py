
import sys, time, json
sys.path.insert(0, "/work")
import numpy as np
from metro import *
d = json.load(open("/work/final_summary.json"))
t0=time.time()
out = {}
for T, start, E, S in [(1.5,"cold",500,3000), (1.8,"cold",800,3000), (2.5,"hot",1500,5000), (2.8,"hot",1000,5000)]:
    acc = run_metro(16, T, R=1000, E=E, S=S, L=100, seed=40000+int(round(T*100)), start=start)
    U, err, _ = jackknife_U(acc[0].mean(axis=1), acc[1].mean(axis=1))
    f = d[str(T)]
    z = (f['U']-U)/np.sqrt(f['err']**2+err**2)
    zi = (f['Ui']-U)/np.sqrt(f['erri']**2+err**2)
    out[T]=(U,err)
    print(f"T={T} Metropolis({start}): U={U:.6f} +- {err:.6f} | SW plain {f['U']:.6f} +- {f['err']:.6f}: z={z:+.2f} | SW improved {f['Ui']:.6f}: z={zi:+.2f}   [{time.time()-t0:.0f}s]", flush=True)
np.savez("/work/metro_res2.npz", **{f"T{T}": np.array(v) for T,v in out.items()})
