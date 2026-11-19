
import sys, time
sys.path.insert(0, "/work")
import numpy as np
import sw

# timing
for R in [250, 1000]:
    t0 = time.time()
    o = sw.run_sw(16, 2.3, R=R, n_burn=0, n_meas=20, seed=5, start="hot")
    dt = time.time() - t0
    print(f"R={R}: {dt/20*1e3:.1f} ms per SW step -> {R*20/dt:.0f} replica-steps/s")

# equilibration trajectories (ensemble average of improved & plain m2 over R replicas vs sweep index)
R = 1000
for T in [1.5, 2.3, 2.8]:
    res = {}
    for start in ["cold", "hot"]:
        o = sw.run_sw(16, T, R=R, n_burn=0, n_meas=60, seed=11, start=start, traj=True)
        res[start] = o["traj"]
    print(f"T={T}: ensemble-mean m2 (improved / plain) vs sweep")
    for t in [0, 1, 2, 3, 5, 8, 12, 20, 30, 45, 59]:
        c = res["cold"][t]; h = res["hot"][t]
        print(f"   t={t:3d}  cold: {c[0]:.4f} / {c[1]:.4f}    hot: {h[0]:.4f} / {h[1]:.4f}")
