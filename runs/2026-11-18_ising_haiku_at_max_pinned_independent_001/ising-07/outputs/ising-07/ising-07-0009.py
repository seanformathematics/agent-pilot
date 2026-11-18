import sys, json
import numpy as np
sys.path.insert(0, "/work")
from sw_ising import U_jack, U_from_series, jack_U

BURN = 200; NSW = 1000
Tlist = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
rows = []
zs = []
print(f"{'T':>5} {'U_A':>9} {'sA':>9} {'U_B':>9} {'sB':>9} {'z':>6} | {'U_pool':>9} {'s_pool(B20)':>11} {'s_pool(B100)':>12} {'s_pool(rep)':>11} {'n_rep':>6}")
for i, T in enumerate(Tlist):
    seedA = 20250000 + i
    seedB = 30000100 + i
    dA = np.load(f"/work/prod_T{T:.1f}_s{seedA}.npz")
    dB = np.load(f"/work/setB_T{T:.1f}_s{seedB}.npz")
    MA = dA["Ms"]; MB = dB["Ms"]
    assert MA.shape == (NSW, 2000) and MB.shape == (NSW, 2000)
    UA, sA = U_jack(MA, 20, BURN)
    UB, sB = U_jack(MB, 20, BURN)
    z = (UA - UB)/np.sqrt(sA**2 + sB**2) if (sA**2+sB**2) > 0 else 0.0
    zs.append(z)
    MP = np.concatenate([MA, MB], axis=1)
    UP, sP20 = U_jack(MP, 20, BURN)
    _, sP100 = U_jack(MP, 100, BURN)
    # replica-level bins: one bin per replica (whole post-burn-in chain)
    x = MP[BURN:].astype(float)
    s2 = (x**2).sum(axis=0); s4 = (x**4).sum(axis=0)
    _, sPrep = jack_U(s2, s4, x.shape[0])
    print(f"{T:5.2f} {UA:9.5f} {sA:9.5f} {UB:9.5f} {sB:9.5f} {z:6.2f} | {UP:9.5f} {sP20:11.6f} {sP100:12.6f} {sPrep:11.6f} {MP.shape[1]:6d}", flush=True)
    rows.append(dict(T=T, seedA=seedA, seedB=seedB, U_A=float(UA), sA=float(sA), U_B=float(UB), sB=float(sB),
                     z=float(z), U_pool=float(UP), s_pool_B20=float(sP20), s_pool_B100=float(sP100),
                     s_pool_rep=float(sPrep), n_chains=int(MP.shape[1]), n_sweeps_kept=int(NSW-BURN)))
zs = np.array(zs)
print("RMS z over all T:", np.sqrt(np.mean(zs**2)).round(3), " mean z:", zs.mean().round(3))
print("RMS z excluding T<=2.1 (nontrivial errors only):", np.sqrt(np.mean(zs[7:]**2)).round(3))
with open("/work/final_table.json", "w") as f:
    json.dump(rows, f, indent=1)
print("saved")
