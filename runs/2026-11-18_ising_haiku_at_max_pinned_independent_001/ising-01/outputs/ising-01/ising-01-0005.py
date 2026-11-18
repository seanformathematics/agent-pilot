import sys, time
sys.path.insert(0, "/work")
import numpy as np
from sw import run_sw, binder_jackknife

for T in [2.269]:
    t0 = time.time()
    M2s, M4s, nit = run_sw(16, T, R=1024, n_therm=50, n_meas=200, seed=2024)
    dt = time.time() - t0
    U, err, Ujk = binder_jackknife(M2s, M4s, 200)
    steps = 250
    print(f"T={T} R=1024 steps={steps}: U={U:.5f} err={err:.5f} time={dt:.1f}s  per replica-step={dt/(1024*steps)*1e6:.1f} us  iters/step={nit/steps:.1f}")
    print("mean m2 per site-normalized:", (M2s.sum()/(1024*200))/256**2)
