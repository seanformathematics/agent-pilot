import time, sys, json
import numpy as np
sys.path.insert(0, "/work")
from sw_ising import run_chain, U_jack

L = 16; R = 2000; NSW = 1000; BURN = 200
out = {"2.8": [], "2.5": []}
t_start = time.time()
plan = [(2.8, 30000201 + k) for k in range(8)] + [(2.5, 30000301 + k) for k in range(4)]
for T, seed in plan:
    t0 = time.time()
    Ms, Es = run_chain(L, R, T, NSW, seed, init="random")
    U, Ue = U_jack(Ms, 20, BURN)
    out[f"{T:.1f}"].append(dict(seed=seed, U=float(U), Uerr_jack=float(Ue)))
    print(f"T={T} seed={seed}: U={U:.5f} Uerr_jack={Ue:.5f} ({time.time()-t0:.1f}s)", flush=True)
    if time.time() - t_start > 480:
        print("time budget stop", flush=True)
        break
with open("/work/extra_runs.json", "w") as f:
    json.dump(out, f, indent=1)
for key in out:
    Us = np.array([r["U"] for r in out[key]])
    if len(Us) > 1:
        print(f"T={key}: n_extra={len(Us)} mean={Us.mean():.5f} std={Us.std(ddof=1):.5f} mean_jack={np.mean([r['Uerr_jack'] for r in out[key]]):.5f}")
print("total", time.time()-t_start)
