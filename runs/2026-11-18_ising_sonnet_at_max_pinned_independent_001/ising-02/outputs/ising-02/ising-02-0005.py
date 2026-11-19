
import sys, time, json
sys.path.insert(0,"/work")
import numpy as np
from metro import metropolis_batch
from analysis import jackknife_U

out = {}
plan = [(2.2, 40000, 11), (2.3, 40000, 12), (2.5, 20000, 13), (2.8, 20000, 14)]
for T, ns, seed in plan:
    t = time.time()
    S2, S4, cnt = metropolis_batch(16, T, 256, 2000, ns, seed=seed, init="hot", record_every=1)
    n = np.full(256, cnt)
    U, err, Ubc = jackknife_U(S2, S4, n)
    print(f"Metropolis T={T}: U={U:.5f} +- {err:.5f}  (bias-corrected {Ubc:.5f}), sweeps/replica={ns}, time={time.time()-t:.1f}s", flush=True)
    out[str(T)] = dict(U=U, err=err, Ubc=Ubc, seed=seed, nsweeps=ns, R=256)
json.dump(out, open("/work/metropolis_validation.json","w"), indent=1)
