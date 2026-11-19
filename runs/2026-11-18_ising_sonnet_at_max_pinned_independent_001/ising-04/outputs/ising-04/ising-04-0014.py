
import sys, time, json
import numpy as np
sys.path.insert(0, "/work")

def metropolis_run(T, seed, R=1000, N=16, neq=500, nblocks=10, blen=400, start="hot"):
    rng = np.random.default_rng(seed)
    if start == "hot":
        s = (2 * rng.integers(0, 2, size=(R, N, N), dtype=np.int8) - 1).astype(np.int8)
    else:
        s = np.ones((R, N, N), dtype=np.int8)
    ii, jj = np.indices((N, N))
    masks = [((ii + jj) % 2 == 0), ((ii + jj) % 2 == 1)]
    table = np.minimum(1.0, np.exp(-2.0 * np.arange(-4, 5, 2) / T))   # x = s_i * sum_nn in {-4,-2,0,2,4}
    def sweep(s):
        for mask in masks:
            nn = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
            x = s * nn
            prob = table[((x + 4) >> 1).astype(np.intp)]
            acc = (rng.random(s.shape) < prob) & mask
            s = np.where(acc, -s, s)
        return s
    for _ in range(neq):
        s = sweep(s)
    V = N * N
    S2 = np.zeros((nblocks, R)); S4 = np.zeros((nblocks, R))
    for b in range(nblocks):
        for t in range(blen):
            s = sweep(s)
            m = s.reshape(R, -1).sum(axis=1, dtype=np.int64) / V
            S2[b] += m * m; S4[b] += m ** 4
    return S2, S4, blen

def U_of(S2, S4, n): return 1.0 - (S4 / n) / (3.0 * (S2 / n) ** 2)
def jk(S2r, S4r, nper, G):
    R = S2r.size; k = R // G
    s2 = S2r.reshape(G, k).sum(1); s4 = S4r.reshape(G, k).sum(1)
    ng = nper * k; n = ng * G
    Uall = U_of(s2.sum(), s4.sum(), n)
    Uloo = U_of(s2.sum() - s2, s4.sum() - s4, n - ng)
    return Uall, np.sqrt((G - 1) / G * np.sum((Uloo - Uloo.mean()) ** 2))

ref = json.load(open("/work/results_pooled.json"))
t00 = time.time()
met = {}
for T, start in ((2.0, "cold"), (2.3, "hot"), (2.7, "hot")):
    seed = 400000 + int(round(T * 10))
    t0 = time.time()
    S2, S4, blen = metropolis_run(T, seed, start=start)
    nb, R = S2.shape; nper = nb * blen
    u, e = jk(S2.sum(0), S4.sum(0), nper, 50)
    u1, e1 = jk(S2[:nb // 2].sum(0), S4[:nb // 2].sum(0), nper // 2, 50)
    u2, e2 = jk(S2[nb // 2:].sum(0), S4[nb // 2:].sum(0), nper // 2, 50)
    r = ref[str(T)]
    z_imp = (u - r["U"]) / np.hypot(e, r["err"]); z_pl = (u - r["Up"]) / np.hypot(e, r["ep"])
    met[T] = dict(U=u, err=e, seed=seed)
    print(f"T={T} Metropolis({start} start, {nper} sweeps x {R} replicas): U={u:.5f} +- {e:.5f}  | halves: {u1:.5f}+-{e1:.5f}, {u2:.5f}+-{e2:.5f}"
          f" | SW impr {r['U']:.5f}+-{r['err']:.5f} z={z_imp:5.2f} | SW plain {r['Up']:.5f}+-{r['ep']:.5f} z={z_pl:5.2f} | {time.time()-t0:.0f}s", flush=True)
json.dump({str(k): v for k, v in met.items()}, open("/work/metropolis_check.json", "w"))
