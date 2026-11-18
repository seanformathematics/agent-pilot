import json, numpy as np
r1 = json.load(open("/work/results_prod.json"))
r2 = json.load(open("/work/results_prod2.json"))
r3 = json.load(open("/work/results_prod3.json"))
ex = json.load(open("/work/extra_check.json"))
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R, n = 1024, 1000

runs = {}
for T in Ts:
    k = str(T)
    lst = [(r1[k]["U"], r1[k]["U_err"], r1[k]["seed"]),
           (r2[k]["U"], r2[k]["U_err"], r2[k]["seed"]),
           (r3[k]["U"], r3[k]["U_err"], r3[k]["seed"])]
    if k in ex:
        lst.append((ex[k]["U"], ex[k]["U_err"], ex[k]["seed"]))
    runs[k] = lst

# calibration: pooled chi2 of run-to-run differences, using only temperatures with non-negligible errors
chi2 = 0.0; dof = 0
for k, lst in runs.items():
    U = np.array([a for a, _, _ in lst]); e = np.array([b for _, b, _ in lst])
    if e.max() > 1e-4:
        Um = U.mean()
        chi2 += np.sum((U - Um) ** 2 / e ** 2)
        dof += len(U) - 1
ratio = chi2 / dof
r_cal = np.sqrt(max(1.0, ratio))
print(f"pooled chi2/dof = {chi2:.2f}/{dof} = {ratio:.3f}; calibration factor r = {r_cal:.3f}")

final = []
for T in Ts:
    k = str(T)
    lst = runs[k]
    U = np.array([a for a, _, _ in lst]); e = np.array([b for _, b, _ in lst])
    kk = len(U)
    Umean = U.mean()
    Uerr = r_cal * np.sqrt(np.sum(e ** 2)) / kk
    emp = U.std(ddof=1) / np.sqrt(kk)
    final.append(dict(T=T, U=float(Umean), U_err=float(Uerr), n_samples=int(kk * R * n), k=kk,
                      jk_err_per_run=[float(x) for x in e], run_U=[float(x) for x in U],
                      empirical_se=float(emp), seeds=[int(s) for _, _, s in lst]))
    print(f"T={T}: U={Umean:.6f}  U_err={Uerr:.3e}  (k={kk} runs, empirical se={emp:.2e}, jk per-run={np.round(e,6)}) n={kk*R*n}")

json.dump(final, open("/work/final.json", "w"), indent=1)
