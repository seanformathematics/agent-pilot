prod = r'''
import sys, json, time, os
sys.path.insert(0, "/work")
import numpy as np
from sw import run_sw, binder_jackknife

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
L = 16
R = 1024
N_THERM = 200
N_MEAS = 1000
base_seed = 20240601

idxs = [int(a) for a in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(len(Ts)))
outfile = "/work/results_prod.json"
res = json.load(open(outfile)) if os.path.exists(outfile) else {}
for i in idxs:
    T = Ts[i]
    seed = base_seed + i
    t0 = time.time()
    M2s, M4s, nit = run_sw(L, T, R, N_THERM, N_MEAS, seed)
    dt = time.time() - t0
    U, err, Ujk = binder_jackknife(M2s, M4s, N_MEAS)
    n = N_MEAS
    Ur = 1.0 - (M4s / n) / (3.0 * (M2s / n) ** 2)
    NS = L * L
    m2 = (M2s.sum() / (R * n)) / NS**2
    m4 = (M4s.sum() / (R * n)) / NS**4
    res[str(T)] = dict(T=T, seed=seed, R=R, n_therm=N_THERM, n_meas=N_MEAS, U=float(U), U_err=float(err),
                       U_jk=float(Ujk), U_naive_err=float(Ur.std(ddof=1) / np.sqrt(R)),
                       m2=float(m2), m4=float(m4), time_s=dt, iters_per_step=nit / (N_THERM + N_MEAS))
    json.dump(res, open(outfile, "w"), indent=1)
    print(f"T={T} seed={seed}: U={U:.5f} +- {err:.5f} (naive {Ur.std(ddof=1)/np.sqrt(R):.5f}) <m2>={m2:.4f} time={dt:.1f}s", flush=True)
'''
open("/work/prod.py", "w").write(prod)
import subprocess
r = subprocess.run(["python3", "/work/prod.py", "0,1,2,3,4"], capture_output=True, text=True, timeout=560)
print(r.stdout)
print("STDERR:", r.stderr[-2000:])
