import json, glob, math, sys
sys.path.insert(0, '/work')
import numpy as np
Ts = [1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8]
rows = []
for T in Ts:
    r = json.load(open(f'/work/res_T{T:.2f}.json'))
    rows.append(r)
    print(f"T={T:.2f}  U={r['U']:.8f}  err_jk={r['U_err_jackknife']:.3e}  err_chain={r['U_err_chain_scatter']:.3e}  "
          f"n={r['n_samples']}  seeds={r['seeds']}  bound_ok={r['U'] < 2/3}")
allseeds = [s for r in rows for s in r['seeds']]
print("num seeds:", len(allseeds), "unique:", len(set(allseeds)))
ok = all(math.isfinite(r['U']) and 0 < r['U_err_jackknife'] <= 0.005 and r['U'] < 2/3 for r in rows)
print("all checks pass:", ok)
json.dump([dict(T=r['T'], U=r['U'], U_err=r['U_err_jackknife'], n=r['n_samples'], seeds=r['seeds']) for r in rows],
          open('/work/final_summary.json','w'), indent=1)
