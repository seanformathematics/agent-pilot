
import sys, json
sys.path.insert(0,"/work")
import numpy as np
from metro import metropolis_batch
from analysis import *

N=4; n=N*N
confs = ((np.arange(2**n)[:,None] >> np.arange(n)[None,:]) & 1).astype(np.int8)*2-1
S = confs.reshape(-1,N,N).astype(np.int32)
E = -(S*np.roll(S,1,1)).sum((1,2)) - (S*np.roll(S,1,2)).sum((1,2))
Msum = S.sum((1,2))
def exact(T):
    w = np.exp(-(E-E.min())/T); w/=w.sum(); m = Msum/n
    m2=(w*m**2).sum(); m4=(w*m**4).sum(); return 1-m4/(3*m2**2)
zs=[]
for k,(T,seed,init) in enumerate([(1.5,41,"cold"),(2.3,42,"hot"),(2.8,43,"hot"),(2.0,44,"cold"),(2.5,45,"hot")]):
    S2m, S4m, cnt = metropolis_batch(N, T, 256, 1000, 20000, seed=seed, init=init)
    Um, em, _ = jackknife_U(S2m, S4m, np.full(256, cnt)); Ue = exact(T)
    zs.append((Um-Ue)/em)
    print(f"N=4 Metropolis({init}) T={T}: exact U={Ue:.6f}  MC U={Um:.6f}+-{em:.6f}  z={(Um-Ue)/em:+.2f}", flush=True)

# ---- build the submission payload
final = json.load(open("/work/final_results_v1.json"))
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
points=[]
for T in Ts:
    f = final[str(T)]
    points.append(dict(T=T, status="ok", U=round(f["U"], 7), U_err=float(f"{f['err_blk20']:.3g}"), n_samples=f["n"]))
seedsA = [910000 + 100*iT + r for iT in range(14) for r in range(20)]
seedsB = [920000 + 100*iT + r for iT in range(14) for r in range(20)]
extra = [11,12,13,14,21,22,23,24,7,31,41,42,43,44,45] + [777000+r for r in range(10)] + [1000+int(T*100) for T in Ts] + [2000+int(round(T*100)) for T in Ts]
seeds = seedsA + seedsB + sorted(set(extra))
assert len(set(seeds)) == len(seeds), "duplicate seeds"
payload = dict(points=points, seeds=seeds)
json.dump(payload, open("/work/submission_payload.json","w"))
print(json.dumps(points))
print(len(seeds), "seeds")
print("production seed rule: 910000+100*iT+r (batch A) and 920000+100*iT+r (batch B), iT=0..13 temperature index, r=0..19")
