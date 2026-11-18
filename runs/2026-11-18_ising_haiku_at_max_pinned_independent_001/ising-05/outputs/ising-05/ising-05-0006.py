import sys, json, numpy as np
sys.path.insert(0, '/work')
from prodlib import U_jack

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
out = []
for k, T in enumerate(Ts):
    seed = 1000 + k
    d = np.load(f'/work/res_T{T:.2f}_s{seed}.npz')
    S2 = d['S2'].copy(); S4 = d['S4'].copy(); nmeas = int(d['nmeas'])
    seeds = [seed]
    if k >= 7:
        seed2 = 2000 + (k - 7)
        c = np.load(f'/work/chk_T{T:.2f}_s{seed2}.npz')
        S2 = np.concatenate([S2, c['S2']])
        S4 = np.concatenate([S4, c['S4']])
        seeds.append(seed2)
    U, err = U_jack(S2, S4, nmeas)
    n = len(S2) * nmeas
    out.append(dict(T=T, status='ok', U=float(U), U_err=float(err), n_samples=int(n),
                    seeds=seeds, R=int(len(S2))))
    print(f"T={T:.2f}  U={U:.6f}  U_err={err:.6f}  n_samples={n}  replicas={len(S2)}  seeds={seeds}", flush=True)

json.dump(out, open('/work/final_results.json', 'w'), indent=1)
print("max U_err:", max(o['U_err'] for o in out))
print("all finite and <= 2/3:", all(np.isfinite(o['U']) and o['U'] <= 2/3 for o in out))
