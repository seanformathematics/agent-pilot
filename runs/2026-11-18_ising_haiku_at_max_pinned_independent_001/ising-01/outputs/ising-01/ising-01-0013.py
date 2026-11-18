prod2 = r'''
import sys, json, time, os
sys.path.insert(0, "/work")
import numpy as np
from sw import run_sw, binder_jackknife

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
L = 16
R = 1024
N_THERM = 200
N_MEAS = 1000
base_seed2 = 730000

idxs = [int(a) for a in sys.argv[1].split(",")]
outfile = "/work/results_prod2.json"
res = json.load(open(outfile)) if os.path.exists(outfile) else {}
for i in idxs:
    T = Ts[i]
    seed = base_seed2 + i
    t0 = time.time()
    M2s, M4s, nit = run_sw(L, T, R, N_THERM, N_MEAS, seed)
    dt = time.time() - t0
    np.savez(f"/work/raw_prod2_T{T}.npz", M2s=M2s, M4s=M4s)
    U, err, Ujk = binder_jackknife(M2s, M4s, N_MEAS)
    res[str(T)] = dict(T=T, seed=seed, R=R, n_therm=N_THERM, n_meas=N_MEAS, U=float(U), U_err=float(err), time_s=dt)
    json.dump(res, open(outfile, "w"), indent=1)
    print(f"T={T} seed={seed}: U={U:.5f} +- {err:.5f} time={dt:.1f}s", flush=True)
'''
open("/work/prod2.py", "w").write(prod2)
import subprocess
r = subprocess.run(["python3", "/work/prod2.py", "0,1,2,3,4"], capture_output=True, text=True, timeout=590)
print(r.stdout)
print("STDERR:", r.stderr[-2000:])
