
import sys, time, os
sys.path.insert(0, "/work")
import numpy as np
import sw
from scipy import stats

N = 16
TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R, n_burn, n_meas, NB = 1000, 300, 1000, 4

def load(tag, T):
    d = np.load(f"/work/{tag}_T{T:.1f}.npz")
    return {k: d[k] for k in d.files}

# ---- (a) drift check on run1: U from each of the 4 time blocks ----
print("Block (stationarity) check, run1, improved estimator: chi2 of 4 block-U values about their weighted mean (dof=3)")
chis = []
for T in TS:
    d = load("run1", T)
    blk = n_meas // NB
    Ub, eb = [], []
    for b in range(NB):
        u, e, _ = sw.jackknife_U(d["S2i"][:, b], d["S4i"][:, b], blk)
        Ub.append(u); eb.append(max(e, 1e-12))
    Ub = np.array(Ub); eb = np.array(eb)
    w = 1 / eb ** 2
    mu = (w * Ub).sum() / w.sum()
    chi2 = (w * (Ub - mu) ** 2).sum()
    chis.append(chi2)
    print(f"T={T:.1f}: block U = {np.array2string(Ub, precision=5)}  errs~{eb.mean():.5f}  chi2={chi2:.2f}")
print("sum chi2 =", sum(chis), "dof =", 3 * len(TS), " p =", stats.chi2.sf(sum(chis), 3 * len(TS)))

# ---- (b) run2: independent seeds, hot start ----
def production(T, seed, start):
    rng = np.random.default_rng(seed)
    sim = sw.SW(N, R, T, rng)
    s = sim.init_state(start)
    S = {k: np.zeros((R, NB)) for k in ["2i", "4i", "2p", "4p"]}
    blk = n_meas // NB
    for t in range(n_burn + n_meas):
        s, m2i, m4i = sim.step(s)
        if t >= n_burn:
            b = (t - n_burn) // blk
            m2p, m4p = sw.plain_moments(s)
            S["2i"][:, b] += m2i; S["4i"][:, b] += m4i
            S["2p"][:, b] += m2p; S["4p"][:, b] += m4p
    return S

t_start = time.time()
zs = []
for T in TS:
    fn = f"/work/run2_T{T:.1f}.npz"
    seed = 2000 + int(round(T * 10))
    if not os.path.exists(fn):
        S = production(T, seed, "hot")
        np.savez(fn, T=T, seed=seed, R=R, n_burn=n_burn, n_meas=n_meas, NB=NB, **{("S" + k_): v for k_, v in S.items()})
    d1, d2 = load("run1", T), load("run2", T)
    U1, e1, _ = sw.jackknife_U(d1["S2i"].sum(1), d1["S4i"].sum(1), n_meas)
    U2, e2, _ = sw.jackknife_U(d2["S2i"].sum(1), d2["S4i"].sum(1), n_meas)
    z = (U1 - U2) / np.hypot(e1, e2)
    zs.append(z)
    print(f"T={T:.1f}: run1 U={U1:.5f}+-{e1:.5f}   run2(hot,seed {seed}) U={U2:.5f}+-{e2:.5f}   z={z:+.2f}   [{time.time()-t_start:.0f}s]", flush=True)
zs = np.array(zs)
print("z-scores run1-run2: mean %.2f, rms %.2f (expect ~0, ~1); chi2=%.1f for %d dof, p=%.3f" % (zs.mean(), np.sqrt((zs**2).mean()), (zs**2).sum(), len(zs), stats.chi2.sf((zs**2).sum(), len(zs))))
