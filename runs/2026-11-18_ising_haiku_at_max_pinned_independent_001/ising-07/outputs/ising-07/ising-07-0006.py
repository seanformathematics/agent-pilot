import time, sys
import numpy as np
sys.path.insert(0, "/work")
from sw_ising import run_chain, U_from_series, U_jack

BURN = 200
def tau_int(x_series, maxlag=60):
    # x_series: (nt, R) float; average normalized ACF over replicas, Madras-Sokal window
    x = x_series - x_series.mean(axis=0)
    nt, R = x.shape
    f = np.fft.rfft(x, n=2*nt, axis=0)
    ac = np.fft.irfft(f*np.conj(f), axis=0)[:nt]
    ac = ac / ac[0:1]
    rho = ac.mean(axis=1)
    tau = 0.5
    W = 1
    for W in range(1, maxlag):
        tau += rho[W]
        if W >= 6*tau:
            break
    return tau, W

print("== autocorrelation & block-size dependence (production data) ==")
for T, seed in [(2.3, 20250008), (2.8, 20250013), (2.5, 20250010)]:
    d = np.load(f"/work/prod_T{T:.1f}_s{seed}.npz")
    Ms = d["Ms"]
    x = Ms[BURN:].astype(float)
    t2, W2 = tau_int(x**2)
    t4, W4 = tau_int(x**4)
    out = []
    for B in [1, 2, 5, 10, 20, 50, 100]:
        U, Ue = U_jack(Ms, B, BURN)
        out.append(f"B={B}:{Ue:.6f}")
    # replica-level bins (most conservative)
    R = Ms.shape[1]
    Ureps = []
    s2 = (x**2).sum(axis=0); s4 = (x**4).sum(axis=0)
    Urep, Uerr_rep = jack_U = None, None
    from sw_ising import jack_U as jk
    Urep, Uerr_rep = jk(s2, s4, x.shape[0])
    print(f"T={T}: tau_int(M^2)={t2:.2f}, tau_int(M^4)={t4:.2f}; jackknife err vs block: " + " ".join(out) + f" | replica-bins err={Uerr_rep:.6f}")

print("== independent-seed repeats ==")
L = 16; R = 2000; NSW = 1000
results = {}
for T, prod_seed in [(2.3, 20250008), (2.8, 20250013)]:
    runs = []
    d = np.load(f"/work/prod_T{T:.1f}_s{prod_seed}.npz")
    Up, Uep = U_jack(d["Ms"], 20, BURN)
    runs.append(("prod", prod_seed, "random", Up, Uep))
    for seed, init in [(30000001 + int(round(T*10)) , "random"), (30000002 + int(round(T*10)), "random"),
                       (30000003 + int(round(T*10)), "ordered"), (30000004 + int(round(T*10)), "ordered")]:
        t0 = time.time()
        Ms, Es = run_chain(L, R, T, NSW, seed, init=init)
        U, Ue = U_jack(Ms, 20, BURN)
        runs.append(("rep", seed, init, U, Ue))
        print(f"  T={T} seed={seed} init={init}: U={U:.5f} Uerr_jack={Ue:.5f} ({time.time()-t0:.1f}s)", flush=True)
    Us = np.array([r[3] for r in runs])
    Ues = np.array([r[4] for r in runs])
    print(f"T={T}: production U={Up:.5f} (jack err {Uep:.5f}); 5-run mean U={Us.mean():.5f}, sample std across runs={Us.std(ddof=1):.5f}, mean jack err={Ues.mean():.5f}")
    results[T] = dict(runs=[(r[1], r[2], float(r[3]), float(r[4])) for r in runs], std=float(Us.std(ddof=1)), mean_jack=float(Ues.mean()))
import json
with open("/work/repeat_check.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=1)
print("done")
