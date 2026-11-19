
import sys, time
sys.path.insert(0, '/work')
import numpy as np
import sw, kauf
import prod

TS = prod.TS
# ---------- (1) calibration from production data ----------
chi2_tot = 0.0; dof_tot = 0
zs_e = []
print("T      U_plain  err_jk   | sub-block scatter(std of 10 U_k)  mean jk err of sub-blocks (per block)  chi2/dof(9) | ratio err_sub*sqrt(10)/err_full")
for T in TS:
    d = np.load(f'/work/prod_T{T}.npz')
    pr = {k: d[k].mean(axis=1) for k in ["m2", "m4", "m2i", "m4i", "e"]}
    R = len(pr["m2"])
    Uf, ef = sw.jackknife_U(pr["m2"], pr["m4"], 100)
    Uk = []; ek = []
    for k in range(10):
        sl = slice(k, None, 10)
        u, e = sw.jackknife_U(pr["m2"][sl], pr["m4"][sl], 20)
        Uk.append(u); ek.append(e)
    Uk = np.array(Uk); ek = np.array(ek)
    # test: the sub-block values scatter around the full value with sigma_k = ek ; chi2 about mean of Uk
    chi2 = np.sum((Uk - Uk.mean()) ** 2 / ek ** 2)
    chi2_tot += chi2; dof_tot += 9
    print(f"{T}  {Uf:.5f}  {ef:.6f} | std(U_k)={Uk.std(ddof=1):.6f}  rms jk err={np.sqrt(np.mean(ek**2)):.6f}  chi2={chi2:.1f}/9 | {np.sqrt(np.mean(ek**2))/np.sqrt(10)/ef:.3f}")
    # energy z
    em = pr["e"].mean()
    G = 100; k = R // G
    xg = pr["e"].reshape(G, k).sum(axis=1)
    xj = (pr["e"].sum() - xg) / (R - k)
    eerr = np.sqrt((G - 1) / G * np.sum((xj - xj.mean()) ** 2))
    zs_e.append((em - kauf.e_exact(T, 16)) / eerr)
print(f"pooled chi2 = {chi2_tot:.1f} for {dof_tot} dof (expect ~{dof_tot} +- {np.sqrt(2*dof_tot):.1f})")
print("energy z-scores:", np.round(zs_e, 2), " mean z^2 =", np.mean(np.square(zs_e)))

# ---------- (2) heat-bath cross-check ----------
def heatbath(N, R, T, seed, neq, nmeas):
    rng = np.random.default_rng(seed)
    s = (rng.integers(0, 2, size=(R, N, N), dtype=np.int8) * 2 - 1).astype(np.int8)
    s[:R // 2] = 1
    ii, jj = np.indices((N, N))
    masks = [((ii + jj) % 2 == 0)[None, :, :], ((ii + jj) % 2 == 1)[None, :, :]]
    table = 1.0 / (1.0 + np.exp(-2.0 * np.arange(-4, 5) / T))
    def half(par):
        nonlocal s
        nn = np.roll(s, 1, 1) + np.roll(s, -1, 1) + np.roll(s, 1, 2) + np.roll(s, -1, 2)
        prob = table[nn + 4]
        new = np.where(rng.random(s.shape) < prob, 1, -1).astype(np.int8)
        s = np.where(masks[par], new, s)
    for _ in range(neq):
        half(0); half(1)
    a2 = np.zeros(R); a4 = np.zeros(R)
    for _ in range(nmeas):
        half(0); half(1)
        m = s.reshape(R, -1).sum(axis=1, dtype=np.int64) / (N * N)
        m2 = m * m
        a2 += m2; a4 += m2 * m2
    return a2 / nmeas, a4 / nmeas

t0 = time.time()
for T, seed in [(2.0, 9001), (2.3, 9002), (2.6, 9003), (2.8, 9004)]:
    a2, a4 = heatbath(16, 400, T, seed, 1000, 5000)
    Uh, eh = sw.jackknife_U(a2, a4, 40)
    d = np.load(f'/work/prod_T{T}.npz')
    pr = {k: d[k].mean(axis=1) for k in ["m2", "m4"]}
    Us, es = sw.jackknife_U(pr["m2"], pr["m4"], 100)
    print(f"T={T}: heat-bath U={Uh:.5f}+-{eh:.5f}   SW U={Us:.5f}+-{es:.5f}   z={(Uh-Us)/np.hypot(eh,es):+.2f}   [{time.time()-t0:.0f}s]", flush=True)
