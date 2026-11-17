
import json, glob, numpy as np
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
rows = []
zs = []
for T in Ts:
    a = json.load(open(f'/work/results/T_{T:.2f}.json'))
    b = json.load(open(f'/work/results/rep_T_{T:.2f}.json'))
    U1, e1, U2, e2 = a['U'], a['U_err'], b['U'], b['U_err']
    z = (U1 - U2) / np.hypot(e1, e2)
    zs.append(z)
    U = 0.5 * (U1 + U2)
    e = 0.5 * np.hypot(e1, e2)
    rows.append((T, U, e, a['n_samples'] + b['n_samples'], a['seed'], b['seed'], U1, e1, U2, e2, z))
    print(f"T={T:.2f}: run1 U={U1:.7f}±{e1:.2e}  run2 U={U2:.7f}±{e2:.2e}  z={z:+.2f}  -> final U={U:.7f} ± {e:.2e}  n={a['n_samples']+b['n_samples']}")
zs = np.array(zs)
print("z-scores: mean %.2f, rms %.2f (expect ~0 and ~1); chi2/dof = %.2f over %d" % (zs.mean(), np.sqrt((zs**2).mean()), (zs**2).mean(), len(zs)))
print("max U_err:", max(r[2] for r in rows))
print("all U in (0, 2/3):", all(0 < r[1] < 2/3 for r in rows))
print("SEEDS run1:", [r[4] for r in rows])
print("SEEDS run2:", [r[5] for r in rows])
print("FINAL_TABLE")
for r in rows:
    print(f"{r[0]:.2f} {r[1]:.7f} {r[2]:.3e} {r[3]}")
