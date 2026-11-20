import numpy as np
Ts = [1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
MEAS = 4000
for T in Ts:
    d = np.load(f"/work/raw_T{T}.npz")
    S2, S4 = d["S2"] / MEAS, d["S4"] / MEAS   # per-replica time averages
    R = len(S2)
    m2 = S2.mean(); m4 = S4.mean()
    U = 1 - m4 / (3 * m2 * m2)
    # jackknife over replicas (delete-one)
    tot2, tot4 = S2.sum(), S4.sum()
    jk = np.empty(R)
    for r in range(R):
        a2 = (tot2 - S2[r]) / (R - 1); a4 = (tot4 - S4[r]) / (R - 1)
        jk[r] = 1 - a4 / (3 * a2 * a2)
    Ujk = R * U - (R - 1) * jk.mean()
    err = np.sqrt((R - 1) / R * np.sum((jk - jk.mean()) ** 2))
    # replica-level bootstrap-free check: split replicas into 10 groups
    g = np.array_split(np.arange(R), 10)
    Ug = []
    for idx in g:
        a2 = S2[idx].mean(); a4 = S4[idx].mean()
        Ug.append(1 - a4 / (3 * a2 * a2))
    err10 = np.std(Ug, ddof=1) / np.sqrt(10)
    print(f"T={T}: U={Ujk:.5f}  jackknife_err={err:.5f}  group10_err={err10:.5f}  <m2>={m2:.4f}")
