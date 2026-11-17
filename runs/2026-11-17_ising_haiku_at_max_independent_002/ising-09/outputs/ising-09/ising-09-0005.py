import sys, time, json
sys.path.insert(0, '/work')
import ising_lib as L
import numpy as np

out = {}
t_all = time.time()
for T in [2.8, 2.3]:
    Us, errs, times = [], [], []
    for k in range(6):
        seed = 9000 + 10 * k + (0 if T == 2.8 else 1)
        m2, m4, hist, el, itm = L.run_chain(16, 500, T, 300, 2000, seed)
        U, e = L.jack_U_rep(m2.sum(1), m4.sum(1), 2000)
        Us.append(float(U)); errs.append(float(e)); times.append(el)
        print(f"  T={T} seed={seed}: U={U:.5f} err_rep={e:.5f} time={el:.1f}s", flush=True)
    Us = np.array(Us)
    sd = Us.std(ddof=1)
    print(f"T={T}: mean U over seeds={Us.mean():.5f}, scatter sd={sd:.5f} (+-{sd/np.sqrt(2*(len(Us)-1)):.5f}), mean jackknife err={np.mean(errs):.5f}", flush=True)
    out[str(T)] = dict(U=Us.tolist(), errs=errs, sd=float(sd), mean_err=float(np.mean(errs)))
json.dump(out, open('/work/calib.json', 'w'), indent=1)
print("total", time.time() - t_all, flush=True)
