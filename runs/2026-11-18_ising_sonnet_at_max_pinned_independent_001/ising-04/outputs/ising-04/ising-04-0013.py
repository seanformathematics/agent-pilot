
import sys, json
import numpy as np
sys.path.insert(0, "/work")
Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def lnZ_kaufman(N, K):
    s2k, c2k = np.sinh(2 * K), np.cosh(2 * K)
    l = np.arange(2 * N)
    cg = c2k ** 2 / s2k - np.cos(np.pi * l / N)
    gam = np.arccosh(np.maximum(cg, 1.0)); gam[0] = 2 * K + np.log(np.tanh(K))
    odd = gam[1::2]; even = gam[0::2]
    lncosh2 = lambda x: np.abs(x) + np.log1p(np.exp(-2 * np.abs(x)))
    lnsinh2 = lambda x: np.abs(x) + np.log1p(-np.exp(-2 * np.abs(x)))
    terms = np.array([lncosh2(N * odd / 2).sum(), lnsinh2(N * odd / 2).sum(), lncosh2(N * even / 2).sum(), lnsinh2(N * even / 2).sum()])
    signs = np.array([1, 1, 1, np.sign(even[0])]); mx = terms.max()
    return np.log(0.5) + 0.5 * N * N * np.log(2 * np.sinh(2 * K)) + mx + np.log((signs * np.exp(terms - mx)).sum())
def e_exact(N, T, h=1e-5):
    K = 1.0 / T
    return -(lnZ_kaufman(N, K + h) - lnZ_kaufman(N, K - h)) / (2 * h) / (N * N)

def U_of(S2, S4, n): return 1.0 - (S4 / n) / (3.0 * (S2 / n) ** 2)
def jk(S2r, S4r, nper, G):
    R = S2r.size; k = R // G
    s2 = S2r.reshape(G, k).sum(1); s4 = S4r.reshape(G, k).sum(1)
    ng = nper * k; n = ng * G
    Uall = U_of(s2.sum(), s4.sum(), n)
    Uloo = U_of(s2.sum() - s2, s4.sum() - s4, n - ng)
    return Uall, G * Uall - (G - 1) * Uloo.mean(), np.sqrt((G - 1) / G * np.sum((Uloo - Uloo.mean()) ** 2))

G = 100
zi, zp, ze = [], [], []
final = {}
print(f"{'T':>4} | {'U1 (hot)':>9} {'e1':>7} | {'U2 (cold)':>9} {'e2':>7} | z_impr z_plain | pooled U_impr  err | pooled plain err | e_pool-exact z")
for T in Ts:
    d1 = np.load(f"/work/prod_T{T:.1f}.npz"); d2 = np.load(f"/work/prod2_T{T:.1f}.npz")
    blen = int(d1["blen"]); nb, R = d1["i2"].shape; nper = nb * blen
    S1 = {k: d1[k].sum(axis=0) for k in ("p2", "p4", "i2", "i4", "e")}
    S2 = {k: d2[k].sum(axis=0) for k in ("p2", "p4", "i2", "i4", "e")}
    u1, _, e1 = jk(S1["i2"], S1["i4"], nper, G); u2, _, e2 = jk(S2["i2"], S2["i4"], nper, G)
    p1, _, f1 = jk(S1["p2"], S1["p4"], nper, G); p2, _, f2 = jk(S2["p2"], S2["p4"], nper, G)
    zi.append((u1 - u2) / np.hypot(e1, e2)); zp.append((p1 - p2) / np.hypot(f1, f2))
    # pooled over both runs: 2000 independent replicas, 200 groups
    Sp = {k: np.concatenate([S1[k], S2[k]]) for k in S1}
    Gp = 200
    up, upbc, ep = jk(Sp["i2"], Sp["i4"], nper, Gp)
    pp, ppbc, fp = jk(Sp["p2"], Sp["p4"], nper, Gp)
    ee = Sp["e"] / nper
    e_mc = ee.mean(); e_err = ee.reshape(Gp, -1).mean(1).std(ddof=1) / np.sqrt(Gp)
    e_ex = e_exact(16, T)
    ze.append((e_mc - e_ex) / e_err)
    final[T] = dict(U=up, U_bc=upbc, err=ep, Up=pp, ep=fp, n=int(2 * R * nper), e_mc=e_mc, e_err=e_err, e_exact=e_ex, U1=u1, e1=e1, U2=u2, e2=e2)
    print(f"{T:4.1f} | {u1:9.5f} {e1:7.5f} | {u2:9.5f} {e2:7.5f} | {zi[-1]:6.2f} {zp[-1]:6.2f} | {up:9.5f} {ep:8.5f} | {pp:8.5f} {fp:8.5f} | {(e_mc-e_ex):8.1e} {ze[-1]:5.2f}")
zi, zp, ze = map(np.array, (zi, zp, ze))
print("chi2 over 14 T: impr %.1f, plain %.1f (expect ~14 +- 5.3); energy chi2 %.1f" % ((zi**2).sum(), (zp**2).sum(), (ze**2).sum()))
json.dump({str(k): v for k, v in final.items()}, open("/work/results_pooled.json", "w"), indent=1)
