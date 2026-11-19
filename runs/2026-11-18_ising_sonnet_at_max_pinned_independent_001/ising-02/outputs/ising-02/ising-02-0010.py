
import sys, time, json
sys.path.insert(0,"/work")
import numpy as np
from metro import metropolis_batch
from analysis import jackknife_U

out = {}
plan = [(1.5, 20000, 21), (1.7, 20000, 22), (1.9, 20000, 23), (2.1, 40000, 24)]
for T, ns, seed in plan:
    t = time.time()
    S2, S4, cnt = metropolis_batch(16, T, 256, 2000, ns, seed=seed, init="cold", record_every=1)
    U, err, Ubc = jackknife_U(S2, S4, np.full(256, cnt))
    print(f"Metropolis(cold) T={T}: U={U:.6f} +- {err:.6f}  sweeps/replica={ns}  time={time.time()-t:.1f}s", flush=True)
    out[str(T)] = dict(U=U, err=err, Ubc=Ubc, seed=seed, nsweeps=ns, R=256)
old = json.load(open("/work/metropolis_validation.json"))
old.update(out)
json.dump(old, open("/work/metropolis_validation.json","w"), indent=1)
