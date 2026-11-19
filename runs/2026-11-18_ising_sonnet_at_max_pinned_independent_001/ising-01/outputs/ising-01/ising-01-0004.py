
import sys, time, os
sys.path.insert(0, "/work")
import numpy as np
import sw

N = 16
TS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
R, n_burn, n_meas, NB = 1000, 300, 1000, 4

def production(T, seed, R=R, n_burn=n_burn, n_meas=n_meas, NB=NB, start="cold"):
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

tag = "run1"
t_start = time.time()
for k, T in enumerate(TS):
    fn = f"/work/{tag}_T{T:.1f}.npz"
    if os.path.exists(fn):
        continue
    seed = 1000 + int(round(T * 10))
    t0 = time.time()
    S = production(T, seed)
    np.savez(fn, T=T, seed=seed, R=R, n_burn=n_burn, n_meas=n_meas, NB=NB, **{("S" + k_): v for k_, v in S.items()})
    Ui, eri, Uib = sw.jackknife_U(S["2i"].sum(1), S["4i"].sum(1), n_meas)
    Up, erp, Upb = sw.jackknife_U(S["2p"].sum(1), S["4p"].sum(1), n_meas)
    print(f"T={T:.1f} seed={seed}: improved U={Ui:.5f} +- {eri:.5f} (bias-corr {Uib:.5f}) | plain U={Up:.5f} +- {erp:.5f} | diff/err={(Ui-Up)/np.hypot(eri,erp):+.2f} | {time.time()-t0:.0f}s", flush=True)
print("total", time.time() - t_start)
