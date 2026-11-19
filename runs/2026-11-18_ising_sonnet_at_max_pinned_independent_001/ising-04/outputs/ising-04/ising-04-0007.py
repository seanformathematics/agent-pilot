
import sys, json
import numpy as np
sys.path.insert(0, "/work")

Ts = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

def lnZ_kaufman(N, K):
    s2k, c2k = np.sinh(2 * K), np.cosh(2 * K)
    l = np.arange(2 * N)
    cg = c2k ** 2 / s2k - np.cos(np.pi * l / N)
    gam = np.arccosh(np.maximum(cg, 1.0))
    gam[0] = 2 * K + np.log(np.tanh(K))
    odd = gam[1::2]; even = gam[0::2]
    lncosh2 = lambda x: np.abs(x) + np.log1p(np.exp(-2 * np.abs(x)))
    lnsinh2 = lambda x: np.abs(x) + np.log1p(-np.exp(-2 * np.abs(x)))
    terms = np.array([lncosh2(N * odd / 2).sum(), lnsinh2(N * odd / 2).sum(), lncosh2(N * even / 2).sum(), lnsinh2(N * even / 2).sum()])
    signs = np.array([1, 1, 1, np.sign(even[0])])
    mx = terms.max()
    return np.log(0.5) + 0.5 * N * N * np.log(2 * s2k) + mx + np.log((signs * np.exp(terms - mx)).sum())

def e_exact(N, T, h=1e-5):
    K = 1.0 / T
    return -(lnZ_kaufman(N, K + h) - lnZ_kaufman(N, K - h)) / (2 * h) / (N * N)

def U_of(S2, S4, n):
    return 1.0 - (S4 / n) / (3.0 * (S2 / n) ** 2)

def jk(S2r, S4r, nper, G):
    """S2r,S4r: per-replica sums (R,), each over nper measurements. delete-one-group jackknife."""
    R = S2r.size; k = R // G
    s2 = S2r.reshape(G, k).sum(1); s4 = S4r.reshape(G, k).sum(1)
    ng = nper * k; n = ng * G
    Uall = U_of(s2.sum(), s4.sum(), n)
    Uloo = U_of(s2.sum() - s2, s4.sum() - s4, n - ng)
    var = (G - 1) / G * np.sum((Uloo - Uloo.mean()) ** 2)
    return Uall, G * Uall - (G - 1) * Uloo.mean(), np.sqrt(var)

def delta(S2r, S4r, nper):
    R = S2r.size
    x2, x4 = S2r / nper, S4r / nper
    M2, M4 = x2.mean(), x4.mean()
    z = -x4 / (3 * M2 ** 2) + 2 * M4 * x2 / (3 * M2 ** 3)
    return np.sqrt(z.var(ddof=1) / R)

G = 100
results = {}
print(f"{'T':>4} | {'U_impr':>8} {'jk_err':>8} {'delta':>8} {'Ujk-U':>8} | {'U_plain':>8} {'err':>8} {'diff/sd':>7} | half1-half2 (z) impr/plain | e: exact, MC, z")
for T in Ts:
    d = np.load(f"/work/prod_T{T:.1f}.npz")
    blen = int(d["blen"]); nb, R = d["i2"].shape
    nper = nb * blen
    S = {k: d[k].sum(axis=0) for k in ("p2", "p4", "i2", "i4", "e")}
    Ui, Uji, ei = jk(S["i2"], S["i4"], nper, G)
    di = delta(S["i2"], S["i4"], nper)
    Up, Ujp, ep = jk(S["p2"], S["p4"], nper, G)
    # consistency plain vs improved: jackknife of the difference
    k = R // G
    def Udiff_loo():
        a2 = S["i2"].reshape(G, k).sum(1); a4 = S["i4"].reshape(G, k).sum(1)
        b2 = S["p2"].reshape(G, k).sum(1); b4 = S["p4"].reshape(G, k).sum(1)
        ng = nper * k; n = ng * G
        ui = U_of(a2.sum() - a2, a4.sum() - a4, n - ng); up = U_of(b2.sum() - b2, b4.sum() - b4, n - ng)
        dd = ui - up
        return np.sqrt((G - 1) / G * np.sum((dd - dd.mean()) ** 2))
    sd_diff = Udiff_loo()
    # stationarity: first half vs second half of the measurement blocks (improved)
    h = nb // 2
    S1 = {kk: d[kk][:h].sum(axis=0) for kk in ("i2", "i4", "p2", "p4")}
    S2_ = {kk: d[kk][h:].sum(axis=0) for kk in ("i2", "i4", "p2", "p4")}
    zs = []
    for pre in ("i", "p"):
        U1, _, e1 = jk(S1[pre + "2"], S1[pre + "4"], h * blen, G)
        U2, _, e2 = jk(S2_[pre + "2"], S2_[pre + "4"], (nb - h) * blen, G)
        zs.append((U1 - U2) / np.hypot(e1, e2))
    # energy check
    ee = (S["e"] / nper)
    e_mc = ee.mean(); e_err = ee.reshape(G, k).mean(1).std(ddof=1) / np.sqrt(G)
    e_ex = e_exact(16, T)
    results[T] = dict(U=Ui, U_jackbc=Uji, err=ei, err_delta=di, Up=Up, ep=ep, n=int(R * nper), e_mc=e_mc, e_err=e_err, e_exact=e_ex)
    print(f"{T:4.1f} | {Ui:8.5f} {ei:8.5f} {di:8.5f} {Uji-Ui:8.1e} | {Up:8.5f} {ep:8.5f} {(Ui-Up)/sd_diff:7.2f} | {zs[0]:5.2f} {zs[1]:5.2f} | {e_ex:9.5f} {e_mc:9.5f} {(e_mc-e_ex)/e_err:5.2f}")
json.dump({str(k): v for k, v in results.items()}, open("/work/results_main.json", "w"), indent=1)
