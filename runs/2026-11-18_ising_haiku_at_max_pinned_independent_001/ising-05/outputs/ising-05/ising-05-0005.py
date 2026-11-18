import sys, time, glob, numpy as np
sys.path.insert(0, '/work')
from prodlib import run_T, U_jack

Ts = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R_chk, nburn, nmeas = 1000, 100, 1000
rows = []
for k, T in enumerate(Ts):
    seed_orig = 1007 + k
    o = np.load(f'/work/res_T{T:.2f}_s{seed_orig}.npz')
    Uo, eo = U_jack(o['S2'], o['S4'], nmeas)
    # group-scatter check on original data: 20 groups of 100 replicas
    R0 = len(o['S2'])
    gU = []
    for g in range(20):
        sl = slice(g * 100, (g + 1) * 100)
        Ug, _ = U_jack(o['S2'][sl], o['S4'][sl], nmeas)
        gU.append(Ug)
    gU = np.array(gU)
    e_group = gU.std(ddof=1) / np.sqrt(20)
    seed_new = 2000 + k
    t0 = time.time()
    d = run_T(T, R_chk, nburn, nmeas, seed_new)
    np.savez(f'/work/chk_T{T:.2f}_s{seed_new}.npz', **{kk: np.asarray(v) for kk, v in d.items()})
    Un, en = U_jack(d['S2'], d['S4'], nmeas)
    z = (Un - Uo) / np.sqrt(en**2 + eo**2)
    rows.append((T, Uo, eo, e_group, Un, en, z))
    print(f"T={T:.2f}: orig U={Uo:.6f} +- {eo:.6f} (group-scatter err {e_group:.6f}) | fresh U={Un:.6f} +- {en:.6f} | z={z:+.2f}  ({time.time()-t0:.1f}s)", flush=True)
zs = np.array([r[-1] for r in rows])
print("chi2/dof of z-scores:", np.mean(zs**2), " max|z|:", np.max(np.abs(zs)))
