import sys, json
sys.path.insert(0, '/work')
import wolff_prod as wp
rows, ntot, nfiles = wp.analyze()
print("steps per chain:", ntot, "block files:", nfiles, "chains:", wp.NC)
out = []
for r in rows:
    out.append(dict(T=r['T'], U=float(f"{r['U']:.10g}"), U_err=float(f"{r['U_err']:.6g}"), n_samples=r['n_samples']))
    print(f"T={r['T']:.2f}  U={r['U']:.10e}  U_err={r['U_err']:.6e}  n={r['n_samples']}  U_half1={r['U_half1']:.7f} U_half2={r['U_half2']:.7f}  E/N={r['E']:.6f}+-{r['E_err']:.6f}")
print(json.dumps(out))
import numpy as np
print("all finite:", all(np.isfinite(o['U']) and np.isfinite(o['U_err']) and o['U_err']>0 for o in out))
print("max U:", max(o['U'] for o in out), "max U_err:", max(o['U_err'] for o in out))
