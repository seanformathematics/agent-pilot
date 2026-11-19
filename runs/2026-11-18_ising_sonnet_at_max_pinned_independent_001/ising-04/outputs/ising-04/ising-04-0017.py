
import sys, time, json
import numpy as np

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

def metropolis_run(T, seed, R=2000, N=16, neq=500, nblocks=10, blen=1000, start="hot"):
    rng = np.random.default_rng(seed)
    s = (2 * rng.integers(0, 2, size=(R, N, N), dtype=np.int8) - 1).astype(np.int8) if start == "hot" else np.ones((R, N, N), dtype=np.int8)
    ii, jj = np.indices((N, N))
    masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
    table = np.minimum(1.0, np.exp(-2.0 * np.arange(-4, 5, 2) / T))
    def sweep(s):
        for mask in masks:
            nn = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
            prob = table[((s * nn + 4) >> 1).astype(np.intp)]
            s = np.where((rng.random(s.shape) < prob) & mask, -s, s)
        return s
    for _ in range(neq):
        s = sweep(s)
    V = N * N
    S2 = np.zeros((nblocks, R)); S4 = np.zeros((nblocks, R)); SE = np.zeros((nblocks, R)); ne = 0
    for b in range(nblocks):
        for t in range(blen):
            s = sweep(s)
            m = s.reshape(R, -1).sum(axis=1, dtype=np.int64) / V
            S2[b] += m * m; S4[b] += m ** 4
            if t % 10 == 0:
                SE[b] += -((s * np.roll(s, -1, 1)).sum(axis=(1, 2), dtype=np.int64) + (s * np.roll(s, -1, 2)).sum(axis=(1, 2), dtype=np.int64)) / V
    return S2, S4, SE, blen

def U_of(S2, S4, n): return 1.0 - (S4 / n) / (3.0 * (S2 / n) ** 2)
def jk(S2r, S4r, nper, G):
    R = S2r.size; k = R // G
    s2 = S2r.reshape(G, k).sum(1); s4 = S4r.reshape(G, k).sum(1)
    ng = nper * k; n = ng * G
    Uall = U_of(s2.sum(), s4.sum(), n)
    Uloo = U_of(s2.sum() - s2, s4.sum() - s4, n - ng)
    return Uall, np.sqrt((G - 1) / G * np.sum((Uloo - Uloo.mean()) ** 2))

ref = json.load(open("/work/final_candidates.json"))
old = json.load(open("/work/metropolis_check2.json"))
t00 = time.time(); met = {}
for T in (2.8, 2.7, 2.6, 2.5):
    seed = 420000 + int(round(T * 10))
    S2, S4, SE, blen = metropolis_run(T, seed)
    nb, R = S2.shape; nper = nb * blen
    u, e = jk(S2.sum(0), S4.sum(0), nper, 100)
    r = ref[str(T)]
    z = (u - r["U"]) / np.hypot(e, r["jk200"]); zp = (u - r["Up"]) / np.hypot(e, r["ep"])
    ee = SE.sum(0) / (nb * (blen // 10)); e_mc = ee.mean(); e_err = ee.reshape(100, -1).mean(1).std(ddof=1) / 10
    ex = e_exact(16, T)
    msg = f"T={T}: Metropolis U={u:.5f} +- {e:.5f} | SW impr {r['U']:.5f}+-{r['jk200']:.5f} z={z:5.2f} | SW plain {r['Up']:.5f}+-{r['ep']:.5f} z={zp:5.2f} | energy MC {e_mc:.5f}+-{e_err:.5f} exact {ex:.5f} z={(e_mc-ex)/e_err:5.2f}"
    if str(T) in old:
        o = old[str(T)]; w1, w2 = 1 / e ** 2, 1 / o["err"] ** 2
        uc = (u * w1 + o["U"] * w2) / (w1 + w2); ec = (w1 + w2) ** -0.5
        zc = (uc - r["U"]) / np.hypot(ec, r["jk200"])
        msg += f" | combined with previous Metropolis run: {uc:.5f}+-{ec:.5f} z_vs_SW_impr={zc:5.2f}"
    print(msg + f" | {time.time()-t00:.0f}s", flush=True)
    met[T] = dict(U=u, err=e, seed=seed)
json.dump({str(k): v for k, v in met.items()}, open("/work/metropolis_check3.json", "w"))
