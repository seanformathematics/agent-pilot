
import sys, os, time, json
sys.path.insert(0, '/work')
import numpy as np
import ising
temps = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
out = {}
rs = np.random.default_rng(199)
for k, T in enumerate(temps):
    seed = 100 * (k + 8) + 7
    t0 = time.time()
    S2, S4, nm, series = ising.run(T, nburn=2000, nmeas_new=3000, seed=seed)
    U, err = ising.ubinder_stats(S2, S4, nm)
    Us = []
    for _ in range(200):
        idx = rs.choice(len(S2), 100, replace=False)
        Us.append(ising.ubinder_stats(S2[idx], S4[idx], nm)[0])
    sub_std = float(np.std(Us, ddof=1))
    UA = ising.ubinder_stats(S2[:500], S4[:500], nm)[0]
    UB = ising.ubinder_stats(S2[500:], S4[500:], nm)[0]
    x = series[:, 0] - series[:, 0].mean()
    ac = np.correlate(x, x, 'full')[len(x)-1:] / (x @ x)
    tau = 1 + 2 * np.sum(ac[1:200] * (ac[1:200] > 0.05))
    out[T] = dict(seed=seed, U=float(U), err=float(err), nmeas=int(nm), sub100_std=sub_std,
                  err_expected_from_sub=sub_std / np.sqrt(10), halfA=float(UA), halfB=float(UB), tau_m2=float(tau))
    print("T=%.2f U=%.5f err=%.5f sub100std/sqrt10=%.5f halves=(%.5f,%.5f) tau~%.1f  time=%.0fs" % (
        T, U, err, sub_std / np.sqrt(10), UA, UB, tau, time.time() - t0), flush=True)
with open('/work/results_part2.json', 'w') as f:
    json.dump(out, f, indent=1)
