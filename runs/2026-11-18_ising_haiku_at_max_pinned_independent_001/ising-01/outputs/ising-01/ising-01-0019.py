import json, numpy as np
r1 = json.load(open("/work/results_prod.json"))
r2 = json.load(open("/work/results_prod2.json"))
r3 = json.load(open("/work/results_prod3.json"))
ex = json.load(open("/work/extra_check.json"))
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
rows = []
chi2_tot = 0.0; dof_tot = 0
print(f"{'T':>4} {'U1':>9} {'e1':>8} {'U2':>9} {'e2':>8} {'U3':>9} {'e3':>8} {'mean':>9} {'chi2(2dof)':>10}")
for T in Ts:
    k = str(T)
    U = np.array([r1[k]["U"], r2[k]["U"], r3[k]["U"]])
    e = np.array([r1[k]["U_err"], r2[k]["U_err"], r3[k]["U_err"]])
    Um = U.mean()
    chi2 = np.sum((U - Um) ** 2 / e ** 2)
    rows.append((T, U, e, Um, chi2))
    if e.max() > 2e-5:
        chi2_tot += chi2; dof_tot += 2
    print(f"{T:>4} {U[0]:9.5f} {e[0]:8.5f} {U[1]:9.5f} {e[1]:8.5f} {U[2]:9.5f} {e[2]:8.5f} {Um:9.5f} {chi2:10.2f}")
print("extra runs (R=1024): T=2.3 U=%.5f e=%.5f ; T=2.8 U=%.5f e=%.5f" % (ex["2.3"]["U"], ex["2.3"]["U_err"], ex["2.8"]["U"], ex["2.8"]["U_err"]))
print("pooled chi2/dof over T with nontrivial errors:", round(chi2_tot, 2), dof_tot, " ratio=", round(chi2_tot/dof_tot, 3), " r=", round(np.sqrt(chi2_tot/dof_tot), 3))
