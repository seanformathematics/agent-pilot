
import numpy as np, json
TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def jk_from_sums(a2, a4, K):
    R = a2.size
    S2 = a2.sum(); S4 = a4.sum(); Nt = R*K
    U = 1 - (S4/Nt)/(3*(S2/Nt)**2)
    S2j = S2 - a2; S4j = S4 - a4; Nj = (R-1)*K
    Uj = 1 - (S4j/Nj)/(3*(S2j/Nj)**2)
    err = np.sqrt((R-1)/R*np.sum((Uj-Uj.mean())**2))
    return float(U), float(err)

rows = []
for T in TS:
    # run 1: raw data saved
    M1 = np.load(f"/work/prod_M_T{T:.2f}.npy").astype(np.float64)/256.0
    K1 = M1.shape[1]
    a2_1 = (M1*M1).sum(axis=1); a4_1 = (M1**4).sum(axis=1)
    U1, e1 = jk_from_sums(a2_1, a4_1, K1)
    # run 2: saved per-replica sums
    d2 = np.load(f"/work/prod2_sums_T{T:.2f}.npz")
    a2_2 = d2['a2']; a4_2 = d2['a4']; K2 = int(d2['K'])
    U2, e2 = jk_from_sums(a2_2, a4_2, K2)
    # combined 2000 replicas
    a2c = np.concatenate([a2_1, a2_2]); a4c = np.concatenate([a4_1, a4_2])
    Uc, ec = jk_from_sums(a2c, a4c, K1)
    z = (U1-U2)/np.sqrt(e1**2+e2**2) if (e1**2+e2**2) > 0 else np.nan
    rows.append(dict(T=T, U1=U1, e1=e1, U2=U2, e2=e2, Uc=Uc, ec=ec, z=z))
    print(f"T={T:.2f}: run1 U={U1:.5f} (jk {e1:.5f}) | run2 U={U2:.5f} (jk {e2:.5f}) | combined U={Uc:.5f} jk_err={ec:.5f} | z(diff)={z:+.2f}")

zs = np.array([r['z'] for r in rows if np.isfinite(r['z']) and r['e1']>1e-4])
print(f"\nCalibration over T with e>1e-4 (n={zs.size}): mean z^2 = {np.mean(zs**2):.3f} (expect 1), mean z = {zs.mean():+.3f}")
# Also: ratio of observed run-difference to predicted, using only temperatures with appreciable errors
big = [r for r in rows if r['e1']>1e-4]
for r in big:
    print(f"  T={r['T']:.2f}: |U1-U2|={abs(r['U1']-r['U2']):.5f}, predicted sd of difference={np.sqrt(r['e1']**2+r['e2']**2):.5f}")
json.dump(rows, open("/work/combined_results.json", "w"), indent=1)
