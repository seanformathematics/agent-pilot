
import sys, json
import numpy as np
sys.path.insert(0, "/work")
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def U_of(S2, S4, n):
    return 1.0 - (S4 / n) / (3.0 * (S2 / n) ** 2)

def jk_err(S2r, S4r, nper, G):
    R = S2r.size; k = R // G
    s2 = S2r.reshape(G, k).sum(1); s4 = S4r.reshape(G, k).sum(1)
    ng = nper * k; n = ng * G
    Uloo = U_of(s2.sum() - s2, s4.sum() - s4, n - ng)
    return np.sqrt((G - 1) / G * np.sum((Uloo - Uloo.mean()) ** 2))

G = 100
chi_all = {"i": [], "p": []}
for T in Ts:
    d = np.load(f"/work/prod_T{T:.1f}.npz")
    blen = int(d["blen"]); nb, R = d["i2"].shape
    line = f"T={T}: "
    for pre in ("i", "p"):
        Ub = np.array([U_of(d[pre + "2"][b].sum(), d[pre + "4"][b].sum(), R * blen) for b in range(nb)])
        # per-block error estimated via jackknife over replica groups within each block
        eb = np.array([jk_err(d[pre + "2"][b], d[pre + "4"][b], blen, G) for b in range(nb)])
        chi2 = np.sum((Ub - np.average(Ub, weights=1 / eb ** 2)) ** 2 / eb ** 2)
        chi_all[pre].append(chi2)
        line += f"[{pre}] chi2(9dof)={chi2:5.1f}  "
        if T in (1.5, 1.6):
            print(f"   T={T} {pre} block U - mean (x1e6):", np.round((Ub - Ub.mean()) * 1e6, 2), " block err x1e6:", np.round(eb * 1e6, 2))
    print(line)
for pre in ("i", "p"):
    c = np.array(chi_all[pre]); print(pre, "sum chi2 =", c.sum().round(1), "for", 9 * len(c), "dof; mean chi2/dof", (c.mean() / 9).round(2))

# Look at T=1.5 in detail: per-replica time-block means of improved m2
d = np.load("/work/prod_T1.5.npz")
i2 = d["i2"] / int(d["blen"])   # (nb, R) block-mean of improved m2 per replica
print("T=1.5 improved m2: mean %.6f; per-block means:" % i2.mean(), np.round(i2.mean(axis=1), 6))
print("per-replica-block std %.5f ; min %.5f max %.5f" % (i2.std(), i2.min(), i2.max()))
# any outlier replica-blocks?
q = np.sort(i2.ravel())
print("lowest 8 replica-block means:", np.round(q[:8], 5))
