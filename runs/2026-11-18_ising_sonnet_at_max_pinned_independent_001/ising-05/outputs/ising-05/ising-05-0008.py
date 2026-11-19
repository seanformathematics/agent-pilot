
import sys, time
sys.path.insert(0, '/work')
import numpy as np
import sw, kauf, prod

N = 16
# (a) independent SW rerun at T=1.5, different seed
T = 1.5; R = 1000
sim = sw.SwendsenWang(N, R, T, 8001, start="hot")
sim.s[:R // 2] = 1
for _ in range(300):
    sim.sweep(measure=False)
acc = {k: np.zeros(R) for k in ["m2", "m4", "e"]}
nm = 2000
for _ in range(nm):
    o = sim.sweep(measure=True, energy=True)
    for k in acc:
        acc[k] += o[k]
for k in acc:
    acc[k] /= nm
U, Ue = sw.jackknife_U(acc["m2"], acc["m4"], 100)
em, eerr = prod.jk_mean(acc["e"], 100)
ex = kauf.e_exact(T, N)
d = np.load('/work/prod_T1.5.npz')
pr = {k: d[k].mean(axis=1) for k in ["m2", "m4", "e"]}
U0, U0e = sw.jackknife_U(pr["m2"], pr["m4"], 100)
print(f"SW rerun T=1.5: U={U:.6f}+-{Ue:.6f} (production {U0:.6f}+-{U0e:.6f}, z={(U-U0)/np.hypot(Ue,U0e):+.2f}); e={em:.5f}+-{eerr:.5f} exact={ex:.5f} z={(em-ex)/eerr:+.2f}", flush=True)
# pooled energy over both runs
e0, e0err = prod.jk_mean(pr["e"], 100)
ep = 0.5 * (e0 + em); eperr = 0.5 * np.hypot(e0err, eerr)
print(f"pooled e = {ep:.5f}+-{eperr:.5f}, z vs exact = {(ep-ex)/eperr:+.2f}")

# (b) cold-start heat-bath at low T
def heatbath_cold(N, R, T, seed, neq, nmeas):
    rng = np.random.default_rng(seed)
    s = np.ones((R, N, N), dtype=np.int8)
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
    a2 = np.zeros(R); a4 = np.zeros(R); ae = np.zeros(R)
    for _ in range(nmeas):
        half(0); half(1)
        m = s.reshape(R, -1).sum(axis=1, dtype=np.int64) / (N * N)
        m2 = m * m
        a2 += m2; a4 += m2 * m2
        ae += -((s * np.roll(s, -1, 2)).sum(axis=(1, 2), dtype=np.int64) + (s * np.roll(s, -1, 1)).sum(axis=(1, 2), dtype=np.int64)) / (N * N)
    return a2 / nmeas, a4 / nmeas, ae / nmeas

t0 = time.time()
for T, seed in [(1.5, 9011), (1.8, 9012)]:
    a2, a4, ae = heatbath_cold(N, 400, T, seed, 500, 6000)
    Uh, eh = sw.jackknife_U(a2, a4, 40)
    emh, eerrh = prod.jk_mean(ae, 40)
    d = np.load(f'/work/prod_T{T}.npz')
    pr = {k: d[k].mean(axis=1) for k in ["m2", "m4"]}
    Us, es = sw.jackknife_U(pr["m2"], pr["m4"], 100)
    ex = kauf.e_exact(T, N)
    print(f"T={T}: heat-bath U={Uh:.6f}+-{eh:.6f}  SW U={Us:.6f}+-{es:.6f}  z={(Uh-Us)/np.hypot(eh,es):+.2f} | heat-bath e={emh:.5f}+-{eerrh:.5f} exact={ex:.5f} z={(emh-ex)/eerrh:+.2f}   [{time.time()-t0:.0f}s]", flush=True)
