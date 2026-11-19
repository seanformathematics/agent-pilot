
import sys
sys.path.insert(0, "/work")
import numpy as np
import sw

TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
NT = 1000
def load(tag, T):
    d = np.load(f"/work/{tag}_T{T:.1f}.npz")
    return {k: d[k] for k in d.files}

G, per = 20, 100
ratios, ratios_p = [], []
print("T    pooled U(imp)   jk err    | group scatter std(U_g)  mean jk err_g  ratio | pooled jk err vs std/sqrt(G)")
for T in TS:
    d1, d2 = load("run1", T), load("run2", T)
    S2 = np.concatenate([d1["S2i"].sum(1), d2["S2i"].sum(1)])
    S4 = np.concatenate([d1["S4i"].sum(1), d2["S4i"].sum(1)])
    U, e, Ub = sw.jackknife_U(S2, S4, NT)
    Ug, eg = [], []
    for g in range(G):
        sl = slice(g * per, (g + 1) * per)
        u, er, _ = sw.jackknife_U(S2[sl], S4[sl], NT)
        Ug.append(u); eg.append(er)
    Ug = np.array(Ug); eg = np.array(eg)
    sd = Ug.std(ddof=1); me = np.sqrt((eg ** 2).mean())
    ratios.append(sd / me)
    print(f"{T:.1f}  {U:.6f}  {e:.6f}  | {sd:.6f}  {me:.6f}  {sd/me:.2f} | {e:.6f} vs {sd/np.sqrt(G):.6f}")
ratios = np.array(ratios)
print("ratio (actual scatter / jackknife-predicted) per T:", np.round(ratios, 2))
print("mean ratio = %.3f  (expected 1 +- %.3f if jackknife is honest;  each ratio has ~ +-%.2f)" % (ratios.mean(), 1 / np.sqrt(2 * (G - 1) * len(TS)), 1 / np.sqrt(2 * (G - 1))))
print("rms of ratio = %.3f" % np.sqrt((ratios ** 2).mean()))
